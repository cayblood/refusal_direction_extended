"""Run project tasks on Runpod.

Two workflows are supported:

1. Existing pod: set RUNPOD_SSH_TARGET/RUNPOD_SSH_PORT and use runpod-* tasks.
2. Ephemeral pod: set RUNPOD_API_KEY plus pod spec env vars, then run the
   ephemeral task to create a pod, execute the baseline, and terminate it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REMOTE_DIR = "/workspace/refusal_direction_extended"
DEFAULT_POD_NAME = "refusal-direction-baseline"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_VOLUME_GB = 80
DEFAULT_CONTAINER_DISK_GB = 40
DEFAULT_MIN_VCPU = 4
DEFAULT_MIN_MEMORY_GB = 24
DEFAULT_SSH_PRIVATE_PORT = 22
# These must match Runpod GPU IDs exactly:
# https://docs.runpod.io/references/gpu-types
DEFAULT_GPU_TYPE_IDS = [
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
]
REST_URL = "https://rest.runpod.io/v1"
RSYNC_EXCLUDES = [
    ".git/",
    ".ipynb_checkpoints/",
    ".ruff_cache/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
]


@dataclass(frozen=True)
class RunpodConfig:
    ssh_target: str
    ssh_port: str
    remote_dir: str
    hf_token: str | None

    @classmethod
    def from_env(cls) -> RunpodConfig:
        ssh_target = os.environ.get("RUNPOD_SSH_TARGET")
        if not ssh_target:
            raise RuntimeError(
                "RUNPOD_SSH_TARGET is required. Example: "
                "RUNPOD_SSH_TARGET=root@203.0.113.10"
            )

        return cls(
            ssh_target=ssh_target,
            ssh_port=os.environ.get("RUNPOD_SSH_PORT", "22"),
            remote_dir=os.environ.get("RUNPOD_PROJECT_DIR", DEFAULT_REMOTE_DIR),
            hf_token=os.environ.get("RUNPOD_HF_TOKEN")
            or os.environ.get("HF_TOKEN"),
        )

    @classmethod
    def from_ssh(cls, ssh_target: str, ssh_port: str) -> RunpodConfig:
        return cls(
            ssh_target=ssh_target,
            ssh_port=ssh_port,
            remote_dir=os.environ.get("RUNPOD_PROJECT_DIR", DEFAULT_REMOTE_DIR),
            hf_token=os.environ.get("RUNPOD_HF_TOKEN")
            or os.environ.get("HF_TOKEN"),
        )


@dataclass(frozen=True)
class EphemeralPodSpec:
    api_key: str
    gpu_type_ids: tuple[str, ...]
    name: str
    image_name: str
    cloud_type: str
    volume_gb: int
    container_disk_gb: int
    min_vcpu: int
    min_memory_gb: int
    ssh_private_port: int
    wait_timeout_seconds: int
    keep_pod: bool

    @classmethod
    def from_env(cls) -> EphemeralPodSpec:
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise RuntimeError("RUNPOD_API_KEY is required for ephemeral pods")

        gpu_type_ids = parse_gpu_type_ids(
            os.environ.get("RUNPOD_GPU_TYPE_IDS")
            or os.environ.get("RUNPOD_GPU_TYPE_ID")
        )

        return cls(
            api_key=api_key,
            gpu_type_ids=gpu_type_ids,
            name=os.environ.get("RUNPOD_POD_NAME", DEFAULT_POD_NAME),
            image_name=os.environ.get("RUNPOD_IMAGE_NAME", DEFAULT_IMAGE),
            cloud_type=os.environ.get("RUNPOD_CLOUD_TYPE", DEFAULT_CLOUD_TYPE),
            volume_gb=int(
                os.environ.get("RUNPOD_VOLUME_GB", DEFAULT_VOLUME_GB)
            ),
            container_disk_gb=int(
                os.environ.get(
                    "RUNPOD_CONTAINER_DISK_GB",
                    DEFAULT_CONTAINER_DISK_GB,
                )
            ),
            min_vcpu=int(os.environ.get("RUNPOD_MIN_VCPU", DEFAULT_MIN_VCPU)),
            min_memory_gb=int(
                os.environ.get("RUNPOD_MIN_MEMORY_GB", DEFAULT_MIN_MEMORY_GB)
            ),
            ssh_private_port=int(
                os.environ.get(
                    "RUNPOD_SSH_PRIVATE_PORT", DEFAULT_SSH_PRIVATE_PORT
                )
            ),
            wait_timeout_seconds=int(
                os.environ.get("RUNPOD_WAIT_TIMEOUT_SECONDS", "900")
            ),
            keep_pod=os.environ.get("RUNPOD_KEEP_POD", "").lower()
            in {"1", "true", "yes"},
        )


def parse_gpu_type_ids(raw_value: str | None) -> tuple[str, ...]:
    if raw_value:
        gpu_type_ids = tuple(
            value.strip() for value in raw_value.split(",") if value.strip()
        )
        if gpu_type_ids:
            return gpu_type_ids
    return tuple(DEFAULT_GPU_TYPE_IDS)


def run(args: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def ssh(config: RunpodConfig, command: str) -> None:
    remote_command = f"bash -lc {shlex.quote(command)}"
    run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            config.ssh_port,
            config.ssh_target,
            remote_command,
        ]
    )


def remote_env_prefix(config: RunpodConfig) -> str:
    if not config.hf_token:
        return ""
    return f"export HF_TOKEN={shlex.quote(config.hf_token)}; "


def in_project(config: RunpodConfig, command: str) -> str:
    remote_dir = shlex.quote(config.remote_dir)
    return f"cd {remote_dir} && {remote_env_prefix(config)}{command}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rest_request(
    api_key: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{REST_URL}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "refusal-direction-extended/0.1",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode()
        raise RuntimeError(
            f"Runpod REST API HTTP {exc.code}: {response_body}"
        ) from exc

    if not response_body:
        return {}
    return json.loads(response_body)


def create_pod(spec: EphemeralPodSpec) -> str:
    print("Trying Runpod GPU types:", flush=True)
    for gpu_type_id in spec.gpu_type_ids:
        print(f"  - {gpu_type_id}", flush=True)

    pod_input = {
        "cloudType": spec.cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(spec.gpu_type_ids),
        "gpuTypePriority": "custom",
        "name": spec.name,
        "imageName": spec.image_name,
        "containerDiskInGb": spec.container_disk_gb,
        "volumeInGb": spec.volume_gb,
        "volumeMountPath": "/workspace",
        "minVCPUPerGPU": spec.min_vcpu,
        "minRAMPerGPU": spec.min_memory_gb,
        "ports": [f"{spec.ssh_private_port}/tcp"],
        "supportPublicIp": True,
    }
    pod = rest_request(spec.api_key, "POST", "/pods", pod_input)
    pod_id = pod["id"]
    gpu = pod.get("gpu") or {}
    print(
        f"Created Runpod pod: {pod_id} using {gpu.get('id', 'unknown GPU')}",
        flush=True,
    )
    return pod_id


def terminate_pod(api_key: str, pod_id: str) -> None:
    rest_request(api_key, "DELETE", f"/pods/{pod_id}")
    print(f"Terminated Runpod pod: {pod_id}", flush=True)


def get_pod(api_key: str, pod_id: str) -> dict[str, Any]:
    return rest_request(api_key, "GET", f"/pods/{pod_id}")


def ssh_connection_from_pod(
    pod: dict[str, Any], private_port: int
) -> tuple[str, str] | None:
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    public_port = port_mappings.get(str(private_port))
    if public_ip and public_port:
        return f"root@{public_ip}", str(public_port)
    return None


def wait_for_ssh(spec: EphemeralPodSpec, pod_id: str) -> RunpodConfig:
    deadline = time.monotonic() + spec.wait_timeout_seconds
    last_status = None
    while time.monotonic() < deadline:
        pod = get_pod(spec.api_key, pod_id)
        last_status = pod.get("desiredStatus")
        if connection := ssh_connection_from_pod(pod, spec.ssh_private_port):
            ssh_target, ssh_port = connection
            config = RunpodConfig.from_ssh(ssh_target, ssh_port)
            try:
                ssh(config, "echo ssh-ready")
            except subprocess.CalledProcessError:
                print("SSH port is mapped; waiting for sshd...", flush=True)
            else:
                print(f"Runpod SSH ready: {ssh_target}:{ssh_port}", flush=True)
                return config
        else:
            print(f"Waiting for SSH mapping; status={last_status}", flush=True)
        time.sleep(10)

    raise RuntimeError(
        "Timed out waiting for Runpod SSH. Last desiredStatus="
        f"{last_status!r}. Ensure the image exposes SSH on port "
        f"{spec.ssh_private_port}."
    )


def check_config(config: RunpodConfig) -> None:
    print(f"RUNPOD_SSH_TARGET={config.ssh_target}")
    print(f"RUNPOD_SSH_PORT={config.ssh_port}")
    print(f"RUNPOD_PROJECT_DIR={config.remote_dir}")
    print(f"HF_TOKEN configured={bool(config.hf_token)}")


def check_ephemeral_config(spec: EphemeralPodSpec) -> None:
    print("RUNPOD_GPU_TYPE_IDS=")
    for gpu_type_id in spec.gpu_type_ids:
        print(f"  - {gpu_type_id}")
    print(f"RUNPOD_POD_NAME={spec.name}")
    print(f"RUNPOD_IMAGE_NAME={spec.image_name}")
    print(f"RUNPOD_CLOUD_TYPE={spec.cloud_type}")
    print(f"RUNPOD_VOLUME_GB={spec.volume_gb}")
    print(f"RUNPOD_CONTAINER_DISK_GB={spec.container_disk_gb}")
    print(f"RUNPOD_MIN_VCPU={spec.min_vcpu}")
    print(f"RUNPOD_MIN_MEMORY_GB={spec.min_memory_gb}")
    print(f"RUNPOD_KEEP_POD={spec.keep_pod}")


def ensure_remote_rsync(config: RunpodConfig) -> None:
    ssh(
        config,
        "command -v rsync >/dev/null || "
        "(command -v apt-get >/dev/null && "
        "DEBIAN_FRONTEND=noninteractive apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y rsync) || "
        "(command -v apk >/dev/null && apk add --no-cache rsync) || "
        "(command -v dnf >/dev/null && dnf install -y rsync) || "
        "(command -v yum >/dev/null && yum install -y rsync)",
    )


def sync(config: RunpodConfig) -> None:
    ssh(config, f"mkdir -p {shlex.quote(config.remote_dir)}")
    ensure_remote_rsync(config)

    rsync_args = [
        "rsync",
        "-rz",
        "--delete",
        "--no-owner",
        "--no-group",
        "--no-perms",
        "-e",
        f"ssh -o StrictHostKeyChecking=accept-new -p {config.ssh_port}",
    ]
    for pattern in RSYNC_EXCLUDES:
        rsync_args.extend(["--exclude", pattern])

    destination = f"{config.ssh_target}:{config.remote_dir}/"
    run([*rsync_args, "./", destination], cwd=repo_root())


def setup(config: RunpodConfig) -> None:
    ssh(
        config,
        in_project(
            config,
            "command -v uv >/dev/null || python3 -m pip install uv; uv sync",
        ),
    )


def gpu_check(config: RunpodConfig) -> None:
    ssh(
        config,
        in_project(
            config,
            'uv run python -c "import torch; '
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            'print(torch.cuda.get_device_name(0))"',
        ),
    )


def download_models(config: RunpodConfig) -> None:
    ssh(
        config,
        in_project(
            config,
            "uv run hf download Qwen/Qwen2.5-1.5B-Instruct --repo-type model "
            "&& uv run hf download Qwen/Qwen2.5-3B-Instruct --repo-type model",
        ),
    )


def baseline(config: RunpodConfig, extra_args: Sequence[str]) -> None:
    joined_args = " ".join(shlex.quote(arg) for arg in extra_args)
    command = f"uv run python scripts/baseline.py --device cuda {joined_args}"
    ssh(config, in_project(config, command.strip()))


def exec_remote(config: RunpodConfig, command: str) -> None:
    ssh(config, in_project(config, command))


def run_all(config: RunpodConfig, extra_args: Sequence[str]) -> None:
    sync(config)
    setup(config)
    gpu_check(config)
    download_models(config)
    baseline(config, extra_args)


def run_ephemeral(extra_args: Sequence[str]) -> None:
    spec = EphemeralPodSpec.from_env()
    pod_id = create_pod(spec)
    try:
        config = wait_for_ssh(spec, pod_id)
        run_all(config, extra_args)
    finally:
        if spec.keep_pod:
            print(f"Keeping Runpod pod for debugging: {pod_id}", flush=True)
        else:
            terminate_pod(spec.api_key, pod_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-config")
    subparsers.add_parser("check-ephemeral-config")
    subparsers.add_parser("sync")
    subparsers.add_parser("setup")
    subparsers.add_parser("gpu-check")
    subparsers.add_parser("download-models")

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    ephemeral_parser = subparsers.add_parser("ephemeral")
    ephemeral_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("remote_command")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        match args.command:
            case "check-ephemeral-config":
                check_ephemeral_config(EphemeralPodSpec.from_env())
            case "ephemeral":
                run_ephemeral(args.extra_args)
            case "check-config":
                check_config(RunpodConfig.from_env())
            case "sync":
                sync(RunpodConfig.from_env())
            case "setup":
                setup(RunpodConfig.from_env())
            case "gpu-check":
                gpu_check(RunpodConfig.from_env())
            case "download-models":
                download_models(RunpodConfig.from_env())
            case "baseline":
                baseline(RunpodConfig.from_env(), args.extra_args)
            case "all":
                run_all(RunpodConfig.from_env(), args.extra_args)
            case "exec":
                exec_remote(RunpodConfig.from_env(), args.remote_command)
            case _:
                raise ValueError(f"Unknown command: {args.command}")
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"runpod: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
