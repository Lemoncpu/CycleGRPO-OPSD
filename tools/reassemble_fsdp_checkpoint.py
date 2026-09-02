#!/usr/bin/env python3
"""Export an FSDP checkpoint with CPU/Gloo instead of its training GPUs.

The repository checkpoint manager stores one torch.save state dict per FSDP
rank.  The normal exporter must recreate the original CUDA world size because
each process loads its own filename.  This tool instead launches that source
world size on CPU, gathers one parameter at a time, reconstructs the full
tensor on rank 0, and writes a standard Hugging Face safetensors directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


_MODEL_FILE_PATTERN = re.compile(r"^model_world_size_(\d+)_rank_(\d+)\.pt$")


def discover_checkpoint_world_size(actor_dir: Path) -> int:
    """Return and validate the world size encoded by actor checkpoint files."""
    ranks_by_world_size: dict[int, set[int]] = {}
    for path in actor_dir.iterdir():
        match = _MODEL_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        world_size, rank = (int(value) for value in match.groups())
        ranks_by_world_size.setdefault(world_size, set()).add(rank)
    if len(ranks_by_world_size) != 1:
        found = {world_size: sorted(ranks) for world_size, ranks in ranks_by_world_size.items()}
        raise ValueError(f"Expected exactly one checkpoint world size under {actor_dir}; found {found}.")
    world_size, ranks = next(iter(ranks_by_world_size.items()))
    expected = set(range(world_size))
    if ranks != expected:
        raise ValueError(
            f"Checkpoint rank files are incomplete for world_size={world_size}: "
            f"expected {sorted(expected)}, found {sorted(ranks)}."
        )
    return world_size


def _tensor_metadata(tensor: Any) -> tuple[tuple[int, ...], str]:
    return tuple(int(value) for value in tensor.shape), str(tensor.dtype)


def _state_value_payload(value: Any) -> dict[str, Any]:
    """Serialize one local state-dict value for gloo gather_object."""
    class_name = type(value).__name__
    if class_name == "ShardedTensor" and hasattr(value, "local_shards"):
        global_shape = tuple(int(value) for value in value.metadata().size)
        local_shards = []
        for local_shard in value.local_shards():
            metadata = local_shard.metadata
            local_shards.append(
                {
                    "offsets": tuple(int(value) for value in metadata.shard_offsets),
                    "sizes": tuple(int(value) for value in metadata.shard_sizes),
                    "tensor": local_shard.tensor.detach().cpu().contiguous(),
                }
            )
        dtype = str(value.dtype)
        return {
            "kind": "sharded_tensor",
            "global_shape": global_shape,
            "dtype": dtype,
            "shards": local_shards,
        }
    if class_name == "DTensor" and hasattr(value, "to_local"):
        placements = []
        for placement in value.placements:
            if type(placement).__name__ == "Shard":
                placements.append(("Shard", int(placement.dim)))
            elif type(placement).__name__ == "Replicate":
                placements.append(("Replicate", None))
            else:
                raise ValueError(f"Unsupported DTensor placement {placement!r}.")
        return {
            "kind": "dtensor",
            "global_shape": tuple(int(item) for item in value.shape),
            "dtype": str(value.dtype),
            "placements": tuple(placements),
            "tensor": value.to_local().detach().cpu().contiguous(),
        }
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return {
            "kind": "tensor",
            "shape": _tensor_metadata(value)[0],
            "dtype": _tensor_metadata(value)[1],
            "tensor": value.detach().cpu().contiguous(),
        }
    raise ValueError(f"Unsupported checkpoint value type: {type(value)!r}.")


def _restore_sharded_tensor(torch: Any, payloads: list[dict[str, Any]], key: str):
    first = payloads[0]
    global_shape = first["global_shape"]
    if any(
        payload["kind"] != "sharded_tensor"
        or payload["global_shape"] != global_shape
        or payload["dtype"] != first["dtype"]
        for payload in payloads
    ):
        raise ValueError(f"Inconsistent ShardedTensor metadata for {key!r}.")
    full = torch.empty(global_shape, dtype=payloads[0]["shards"][0]["tensor"].dtype, device="cpu")
    seen: set[tuple[int, ...]] = set()
    covered = 0
    for payload in payloads:
        for shard in payload["shards"]:
            offsets, sizes, tensor = shard["offsets"], shard["sizes"], shard["tensor"]
            shard_id = (*offsets, *sizes)
            if shard_id in seen:
                raise ValueError(f"Duplicate ShardedTensor shard for {key!r}: offsets={offsets}, sizes={sizes}.")
            seen.add(shard_id)
            if tuple(tensor.shape) != sizes:
                raise ValueError(
                    f"ShardedTensor local shape mismatch for {key!r}: expected {sizes}, got {tuple(tensor.shape)}."
                )
            full[tuple(slice(offset, offset + size) for offset, size in zip(offsets, sizes))].copy_(tensor)
            covered += tensor.numel()
    if covered != full.numel():
        raise ValueError(
            f"ShardedTensor {key!r} covers {covered} elements, expected {full.numel()}; checkpoint is incomplete."
        )
    return full


def _restore_dtensor(torch: Any, payloads: list[dict[str, Any]], key: str):
    first = payloads[0]
    if any(
        payload["kind"] != "dtensor"
        or payload["global_shape"] != first["global_shape"]
        or payload["dtype"] != first["dtype"]
        or payload["placements"] != first["placements"]
        for payload in payloads
    ):
        raise ValueError(f"Inconsistent DTensor metadata for {key!r}.")
    placements = first["placements"]
    if placements == (("Replicate", None),):
        tensor = first["tensor"]
        if tuple(tensor.shape) != first["global_shape"]:
            raise ValueError(f"Replicated DTensor shape mismatch for {key!r}.")
        return tensor
    if len(placements) != 1 or placements[0][0] != "Shard":
        raise ValueError(
            f"Only one-dimensional replicated or sharded DTensors are supported; {key!r} has {placements}."
        )
    dimension = placements[0][1]
    full = torch.cat([payload["tensor"] for payload in payloads], dim=dimension)
    if tuple(full.shape) != first["global_shape"]:
        raise ValueError(
            f"DTensor reconstruction shape mismatch for {key!r}: expected {first['global_shape']}, got {tuple(full.shape)}."
        )
    return full


def _restore_tensor(payloads: list[dict[str, Any]], key: str):
    first = payloads[0]
    if any(
        payload["kind"] != "tensor"
        or payload["shape"] != first["shape"]
        or payload["dtype"] != first["dtype"]
        for payload in payloads
    ):
        raise ValueError(f"Inconsistent replicated tensor metadata for {key!r}.")
    return first["tensor"]


def _restore_value(torch: Any, payloads: list[dict[str, Any]], key: str):
    kind = payloads[0]["kind"]
    if kind == "sharded_tensor":
        return _restore_sharded_tensor(torch, payloads, key)
    if kind == "dtensor":
        return _restore_dtensor(torch, payloads, key)
    if kind == "tensor":
        return _restore_tensor(payloads, key)
    raise ValueError(f"Unsupported gathered payload kind {kind!r} for {key!r}.")


def _copy_checkpoint_processor(actor_dir: Path, output_dir: Path) -> None:
    source = actor_dir / "huggingface"
    if not source.is_dir():
        raise ValueError(f"Checkpoint processor/config directory not found: {source}.")
    for path in source.iterdir():
        target = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)


def _save_huggingface_model(state_dict: dict[str, Any], model_path: str, output_dir: Path, max_shard_size: str) -> None:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_class = (
        AutoModelForImageTextToText
        if type(config) in AutoModelForImageTextToText._model_mapping.keys()
        else AutoModelForCausalLM
    )
    with init_empty_weights():
        model = model_class.from_config(config, trust_remote_code=True)
    model.tie_weights()
    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="global_step_* checkpoint directory")
    parser.add_argument("--model-path", required=True, help="Original SAMTok Hugging Face model directory")
    parser.add_argument("--output", required=True, type=Path, help="New or empty Hugging Face export directory")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actor_dir = args.checkpoint / "actor"
    if not actor_dir.is_dir():
        raise ValueError(f"Actor checkpoint directory not found: {actor_dir}.")
    expected_world_size = discover_checkpoint_world_size(actor_dir)

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != expected_world_size:
        raise ValueError(
            f"Launch exactly {expected_world_size} CPU ranks for this checkpoint, got {world_size}."
        )
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"Refusing to overwrite non-empty output directory: {args.output}.")
        args.output.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    shard_path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
    local_state = torch.load(shard_path, map_location="cpu", weights_only=False)
    keys = list(local_state.keys())
    gathered_keys: list[list[str] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(keys, gathered_keys, dst=0)
    if rank == 0:
        assert gathered_keys is not None
        if any(candidate != keys for candidate in gathered_keys):
            raise ValueError("Checkpoint ranks do not contain the same model-state keys.")
    dist.barrier()

    full_state: dict[str, Any] = {}
    for index, key in enumerate(keys, start=1):
        payload = _state_value_payload(local_state[key])
        gathered_payloads: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered_payloads, dst=0)
        if rank == 0:
            assert gathered_payloads is not None
            full_state[key] = _restore_value(torch, list(gathered_payloads), key)
            if index % 50 == 0 or index == len(keys):
                print(f"Reassembled {index}/{len(keys)} state-dict entries.", flush=True)

    if rank == 0:
        _copy_checkpoint_processor(actor_dir, args.output)
        _save_huggingface_model(full_state, args.model_path, args.output, args.max_shard_size)
        manifest = {
            "checkpoint": str(args.checkpoint.resolve()),
            "source_world_size": world_size,
            "backend": "gloo",
            "cuda_used": False,
            "state_dict_entries": len(full_state),
        }
        (args.output / "fsdp_reassembly_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Exported Hugging Face model to {args.output.resolve()}.", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FSDP reassembly failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
