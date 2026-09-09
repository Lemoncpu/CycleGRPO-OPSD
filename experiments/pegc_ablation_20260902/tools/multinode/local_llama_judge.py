#!/usr/bin/env python3
"""Run a detached local vLLM Llama judge on physical GPU 7 of each Ray node.

Ray exposes only the seven training GPUs per node to veRL. This helper uses
CPU-only, node-affine detached actors to manage a vLLM child process that is
explicitly bound to the otherwise unregistered physical GPU 7.  It never asks
Ray to schedule the judge as a GPU actor, so all advertised training GPUs
remain available to the trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="llama3.1-8b")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--gpu-device", default="7")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--name-prefix", default="cyclegrpo-local-llama")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def healthy(
    port: int,
    served_model_name: str | None = None,
    timeout_seconds: float = 2.0,
) -> bool:
    """Return true only for the intended OpenAI-compatible judge service.

    A reachable TCP port is not enough: an unrelated vLLM instance on 8007
    would otherwise be silently reused and receive DLC-QA requests.  When a
    served model name is supplied, require it to be present in `/v1/models`.
    """
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=timeout_seconds
        ) as response:
            if not 200 <= response.status < 300:
                return False
            if not served_model_name:
                return True
            payload = json.load(response)
            model_ids = {
                str(model.get("id"))
                for model in payload.get("data", [])
                if isinstance(model, dict) and model.get("id") is not None
            }
            return served_model_name in model_ids
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return False


@ray.remote(num_cpus=1, max_restarts=0)
class LocalLlamaJudge:
    def __init__(
        self,
        *,
        env_dir: str,
        model_path: str,
        served_model_name: str,
        port: int,
        gpu_device: str,
        gpu_memory_utilization: float,
        chat_template: str,
        log_path: str,
    ) -> None:
        self.env_dir = env_dir
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.port = port
        self.gpu_device = gpu_device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.chat_template = chat_template
        self.log_path = log_path
        self.process: subprocess.Popen | None = None

    def start(self, timeout_seconds: int) -> dict:
        if healthy(self.port, self.served_model_name):
            return {"pid": None, "already_healthy": True, "port": self.port}
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError(f"vLLM process is running but unhealthy on port {self.port}.")

        python_bin = str(Path(self.env_dir) / "bin" / "python3")
        if not os.path.isfile(python_bin):
            raise FileNotFoundError(f"Project Python not found: {python_bin}")
        for required_path in (self.model_path, self.chat_template):
            if not os.path.exists(required_path):
                raise FileNotFoundError(required_path)

        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self.gpu_device
        environment["TOKENIZERS_PARALLELISM"] = "true"
        command = [
            python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--served-model-name",
            self.served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--chat-template",
            self.chat_template,
        ]
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM exited with code {return_code}; inspect {self.log_path}."
                )
            if healthy(self.port, self.served_model_name):
                return {"pid": self.process.pid, "already_healthy": False, "port": self.port}
            time.sleep(2)
        raise TimeoutError(f"vLLM did not become healthy within {timeout_seconds}s; inspect {self.log_path}.")

    def status(self) -> dict:
        return {
            "pid": None if self.process is None else self.process.pid,
            "returncode": None if self.process is None else self.process.poll(),
            "healthy": healthy(self.port, self.served_model_name),
            "port": self.port,
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def stop(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        return self.status()


def alive_nodes() -> list[dict]:
    return sorted(
        (node for node in ray.nodes() if node.get("Alive")),
        key=lambda node: (node.get("NodeManagerAddress", ""), node["NodeID"]),
    )


def actor_name(prefix: str, node_id: str) -> str:
    return f"{prefix}-{node_id[:12]}"


def get_actor_or_none(name: str, namespace: str):
    try:
        return ray.get_actor(name, namespace=namespace)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.expected_nodes <= 0:
        raise ValueError("--expected-nodes must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")

    ray.init(address=args.ray_address, namespace=args.namespace, logging_level="ERROR")
    nodes = alive_nodes()
    if len(nodes) != args.expected_nodes:
        raise RuntimeError(
            f"Expected exactly {args.expected_nodes} alive Ray node(s), found {len(nodes)}."
        )

    results = []
    for node in nodes:
        node_id = node["NodeID"]
        name = actor_name(args.name_prefix, node_id)
        actor = get_actor_or_none(name, args.namespace)
        if args.action == "start":
            if actor is None:
                node_suffix = node_id[:12]
                log_path = str(Path(args.log_dir) / f"llama_judge_{node_suffix}.log")
                actor = LocalLlamaJudge.options(
                    name=name,
                    lifetime="detached",
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(
                    env_dir=args.env_dir,
                    model_path=args.model_path,
                    served_model_name=args.served_model_name,
                    port=args.port,
                    gpu_device=args.gpu_device,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    chat_template=args.chat_template,
                    log_path=log_path,
                )
            result = ray.get(actor.start.remote(args.timeout_seconds))
        elif args.action == "status":
            result = {"missing": True} if actor is None else ray.get(actor.status.remote())
        else:
            if actor is None:
                result = {"missing": True}
            else:
                result = ray.get(actor.stop.remote())
                ray.kill(actor, no_restart=True)
        results.append(
            {
                "node_id": node_id,
                "node_ip": node.get("NodeManagerAddress"),
                "actor": name,
                **result,
            }
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    ray.shutdown()


if __name__ == "__main__":
    main()
