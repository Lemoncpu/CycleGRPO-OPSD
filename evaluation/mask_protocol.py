"""Shared SAMTok mask-generation protocols for offline segmentation evaluation."""

import re


MASK_PROTOCOLS = ("legacy_union", "first_mask")
_MASK_GROUP_PATTERN = re.compile(
    r"<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>"
)
_RAW_MASK_TOKEN_PATTERN = re.compile(r"<\|mt_(\d{4})\|>")


def validate_mask_protocol(protocol: str) -> str:
    if protocol not in MASK_PROTOCOLS:
        raise ValueError(f"Unknown mask protocol {protocol!r}; choose one of {MASK_PROTOCOLS}.")
    return protocol


def complete_mask_group_count(text: str) -> int:
    return len(_MASK_GROUP_PATTERN.findall(text))


def parse_mask_groups(
    text: str,
    *,
    codebook_size: int,
    protocol: str,
) -> list[list[int]]:
    """Return codebook-remapped legal depth-2 SAMTok groups for a protocol.

    ``legacy_union`` retains every complete legal group for historical SAMTok
    comparability. ``first_mask`` keeps only the first complete group, matching
    strict single-mask serialization evaluation.
    """
    validate_mask_protocol(protocol)
    matches = _MASK_GROUP_PATTERN.findall(text)
    if protocol == "first_mask":
        matches = matches[:1]

    groups = []
    for first_text, second_text in matches:
        first, second = int(first_text), int(second_text)
        if 0 <= first < codebook_size and codebook_size <= second < 2 * codebook_size:
            groups.append([first, second - codebook_size])
    return groups


def cyclegrpo_legacy_raw_mask_token_count(text: str) -> int:
    """Count raw mask tokens using the public CycleGRPO RefCOCO regex."""
    return len(_RAW_MASK_TOKEN_PATTERN.findall(text))


def _repair_cyclegrpo_legacy_format(text: str) -> str:
    """Reproduce the public CycleGRPO odd-mask-token repair exactly."""
    text = re.sub(
        r"(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>",
        r"\1\2\3<|mt_end|>",
        text,
    )
    text = re.sub(
        r"(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)",
        r"\1\2<|mt_9999|><|mt_end|>",
        text,
    )
    return re.sub(
        r"(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)",
        r"\1\2<|mt_9999|><|mt_end|>",
        text,
    )


def parse_cyclegrpo_legacy_mask_groups(text: str, *, codebook_size: int) -> list[list[int]]:
    """Parse masks exactly as the public CycleGRPO RefCOCO evaluator does.

    This intentionally accepts bare ``mt`` token pairs and uses the original
    odd-token repair before falling back to ``-1`` for an invalid second code.
    It is a RefCOCO compatibility helper, not a generic evaluation protocol.
    """
    quant_ids = [int(value) for value in _RAW_MASK_TOKEN_PATTERN.findall(text)]
    if not quant_ids:
        return []
    if len(quant_ids) % 2:
        repaired = _repair_cyclegrpo_legacy_format(text)
        quant_ids = [
            int(value)
            for match in _MASK_GROUP_PATTERN.findall(repaired)
            for value in match
        ]
    if len(quant_ids) % 2:
        return []

    groups = []
    for first, second in zip(quant_ids[::2], quant_ids[1::2]):
        if not 0 <= first < codebook_size:
            continue
        remapped_second = second - codebook_size
        groups.append(
            [
                first,
                remapped_second if 0 <= remapped_second < codebook_size else -1,
            ]
        )
    return groups


def generation_eos_token_id(model, processor, protocol: str):
    """Return an EOS override only for strict first-mask generation."""
    validate_mask_protocol(protocol)
    if protocol == "legacy_union":
        return None

    base_eos = model.generation_config.eos_token_id
    eos_token_id = list(base_eos) if isinstance(base_eos, (list, tuple)) else [base_eos]
    eos_token_id.append(processor.tokenizer.convert_tokens_to_ids("<|mt_end|>"))
    return list(dict.fromkeys(token_id for token_id in eos_token_id if token_id is not None))
