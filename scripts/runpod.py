"""Run project tasks on Runpod.

Workflows:

1. Existing pod: set RUNPOD_SSH_TARGET/RUNPOD_SSH_PORT and use runpod-* tasks.
2. Persistent H100 pod: create or reuse a network volume, then retry H100 pod
   creation until a usable SSH/CUDA pod is running.
3. Ephemeral pod: create a temporary pod, run the baseline, and terminate it.
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
DEFAULT_HF_HOME = "/workspace/.cache/huggingface"
DEFAULT_POD_NAME = "refusal-direction-baseline"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_CLOUD_TYPE = "SECURE"
DEFAULT_VOLUME_GB = 80
DEFAULT_CONTAINER_DISK_GB = 40
DEFAULT_MIN_VCPU = 4
DEFAULT_MIN_MEMORY_GB = 24
DEFAULT_SSH_PRIVATE_PORT = 22
DEFAULT_GPU_VERIFY_TIMEOUT_SECONDS = 180
DEFAULT_NETWORK_VOLUME_NAME = "refusal-direction-cache"
DEFAULT_NETWORK_VOLUME_SIZE_GB = 200
DEFAULT_NETWORK_VOLUME_MOUNT_PATH = "/workspace"
DEFAULT_H100_GPU_TYPE_ID = "NVIDIA H100 80GB HBM3"
DEFAULT_DEDICATED_H100_POD_NAME = "refusal-direction-h100"
# These must match Runpod GPU IDs exactly:
# https://docs.runpod.io/references/gpu-types
DEFAULT_GPU_TYPE_IDS = [
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
    "NVIDIA H100 80GB HBM3",
]
REST_URL = "https://rest.runpod.io/v1"
LOCAL_ENV_FILE = ".runpod.env"
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
        load_local_env()
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
class CreatedPod:
    id: str
    requested_gpu_type_ids: tuple[str, ...]


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
    gpu_verify_timeout_seconds: int
    keep_pod: bool

    @classmethod
    def from_env(cls) -> EphemeralPodSpec:
        load_local_env()
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
            gpu_verify_timeout_seconds=int(
                os.environ.get(
                    "RUNPOD_GPU_VERIFY_TIMEOUT_SECONDS",
                    DEFAULT_GPU_VERIFY_TIMEOUT_SECONDS,
                )
            ),
            keep_pod=os.environ.get("RUNPOD_KEEP_POD", "").lower()
            in {"1", "true", "yes"},
        )


@dataclass(frozen=True)
class PersistentH100PodSpec:
    api_key: str
    data_center_id: str
    network_volume_id: str | None
    network_volume_name: str
    network_volume_size_gb: int
    name: str
    image_name: str
    cloud_type: str
    container_disk_gb: int
    min_vcpu: int
    min_memory_gb: int
    ssh_private_port: int
    wait_timeout_seconds: int
    gpu_verify_timeout_seconds: int
    retry_sleep_seconds: int
    max_attempts: int

    @classmethod
    def from_env(cls) -> PersistentH100PodSpec:
        load_local_env()
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            raise RuntimeError("RUNPOD_API_KEY is required")

        data_center_id = os.environ.get("RUNPOD_DATACENTER_ID")
        if not data_center_id:
            raise RuntimeError(
                "RUNPOD_DATACENTER_ID is required so the network volume and "
                "H100 pod are created in the same datacenter"
            )

        return cls(
            api_key=api_key,
            data_center_id=data_center_id,
            network_volume_id=os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
            or None,
            network_volume_name=os.environ.get(
                "RUNPOD_NETWORK_VOLUME_NAME", DEFAULT_NETWORK_VOLUME_NAME
            ),
            network_volume_size_gb=int(
                os.environ.get(
                    "RUNPOD_NETWORK_VOLUME_SIZE_GB",
                    DEFAULT_NETWORK_VOLUME_SIZE_GB,
                )
            ),
            name=os.environ.get(
                "RUNPOD_POD_NAME", DEFAULT_DEDICATED_H100_POD_NAME
            ),
            image_name=os.environ.get("RUNPOD_IMAGE_NAME", DEFAULT_IMAGE),
            cloud_type=os.environ.get("RUNPOD_CLOUD_TYPE", DEFAULT_CLOUD_TYPE),
            container_disk_gb=int(
                os.environ.get(
                    "RUNPOD_CONTAINER_DISK_GB", DEFAULT_CONTAINER_DISK_GB
                )
            ),
            min_vcpu=int(os.environ.get("RUNPOD_MIN_VCPU", "16")),
            min_memory_gb=int(os.environ.get("RUNPOD_MIN_MEMORY_GB", "120")),
            ssh_private_port=int(
                os.environ.get(
                    "RUNPOD_SSH_PRIVATE_PORT", DEFAULT_SSH_PRIVATE_PORT
                )
            ),
            wait_timeout_seconds=int(
                os.environ.get("RUNPOD_WAIT_TIMEOUT_SECONDS", "900")
            ),
            gpu_verify_timeout_seconds=int(
                os.environ.get(
                    "RUNPOD_GPU_VERIFY_TIMEOUT_SECONDS",
                    DEFAULT_GPU_VERIFY_TIMEOUT_SECONDS,
                )
            ),
            retry_sleep_seconds=int(
                os.environ.get("RUNPOD_RETRY_SLEEP_SECONDS", "60")
            ),
            max_attempts=int(os.environ.get("RUNPOD_MAX_ATTEMPTS", "0")),
        )


def local_env_path() -> Path:
    return repo_root() / LOCAL_ENV_FILE


def parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    parts = shlex.split(value, comments=False, posix=True)
    return key, parts[0] if parts else ""


def load_local_env() -> None:
    path = local_env_path()
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        assignment = parse_env_assignment(line)
        if assignment is None:
            continue
        key, value = assignment
        os.environ.setdefault(key, value)


def shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def write_local_env(values: dict[str, str]) -> None:
    path = local_env_path()
    lines = [
        "# Local Runpod state for refusal-direction-extended.",
        "# Generated by scripts/runpod.py. Do not commit this file.",
    ]
    for key in sorted(values):
        lines.append(shell_export(key, values[key]))
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path.relative_to(repo_root())}", flush=True)


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


def ssh_args(config: RunpodConfig, command: str) -> list[str]:
    remote_command = f"bash -lc {shlex.quote(command)}"
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        config.ssh_port,
        config.ssh_target,
        remote_command,
    ]


def ssh(config: RunpodConfig, command: str) -> None:
    run(ssh_args(config, command))


def ssh_output(config: RunpodConfig, command: str) -> str:
    completed = subprocess.run(
        ssh_args(config, command), check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def remote_env_prefix(config: RunpodConfig) -> str:
    exports = {
        "HF_HOME": os.environ.get("RUNPOD_HF_HOME", DEFAULT_HF_HOME),
    }
    exports["HF_HUB_CACHE"] = f"{exports['HF_HOME']}/hub"
    exports["TRANSFORMERS_CACHE"] = exports["HF_HOME"]
    if config.hf_token:
        exports["HF_TOKEN"] = config.hf_token
    return "".join(
        f"export {key}={shlex.quote(value)}; " for key, value in exports.items()
    )


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
    *,
    timeout: int = 60,
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
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode()
        raise RuntimeError(
            f"Runpod REST API HTTP {exc.code}: {response_body}"
        ) from exc

    if not response_body:
        return {}
    return json.loads(response_body)


def create_network_volume(spec: PersistentH100PodSpec) -> str:
    body = {
        "name": spec.network_volume_name,
        "size": spec.network_volume_size_gb,
        "dataCenterId": spec.data_center_id,
    }
    volume = rest_request(spec.api_key, "POST", "/networkvolumes", body)
    volume_id = str(volume["id"])
    print(f"Created Runpod network volume: {volume_id}", flush=True)
    write_local_env(
        {
            "RUNPOD_DATACENTER_ID": spec.data_center_id,
            "RUNPOD_NETWORK_VOLUME_ID": volume_id,
        }
    )
    return volume_id


def ensure_network_volume(spec: PersistentH100PodSpec) -> str:
    if spec.network_volume_id:
        print(
            f"Using existing Runpod network volume: {spec.network_volume_id}",
            flush=True,
        )
        return spec.network_volume_id
    return create_network_volume(spec)


def create_pod(spec: EphemeralPodSpec) -> CreatedPod:
    print("Requesting Runpod GPU types:", flush=True)
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
    observed_gpu = pod_gpu_type_id(pod)
    if observed_gpu and not any(
        same_gpu_type_id(observed_gpu, gpu_type_id)
        for gpu_type_id in spec.gpu_type_ids
    ):
        terminate_pod(spec.api_key, pod_id)
        raise RuntimeError(
            "Runpod created a pod with an unexpected GPU type: "
            f"{observed_gpu!r}; requested one of {list(spec.gpu_type_ids)!r}"
        )

    print(f"Created Runpod pod: {pod_id}; verifying assigned GPU", flush=True)
    return CreatedPod(id=pod_id, requested_gpu_type_ids=spec.gpu_type_ids)


def create_persistent_h100_pod(
    spec: PersistentH100PodSpec, network_volume_id: str
) -> CreatedPod:
    pod_input = {
        "cloudType": spec.cloud_type,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": [DEFAULT_H100_GPU_TYPE_ID],
        "gpuTypePriority": "custom",
        "name": spec.name,
        "imageName": spec.image_name,
        "containerDiskInGb": spec.container_disk_gb,
        "networkVolumeId": network_volume_id,
        "volumeMountPath": DEFAULT_NETWORK_VOLUME_MOUNT_PATH,
        "dataCenterIds": [spec.data_center_id],
        "dataCenterPriority": "custom",
        "minVCPUPerGPU": spec.min_vcpu,
        "minRAMPerGPU": spec.min_memory_gb,
        "ports": [f"{spec.ssh_private_port}/tcp"],
        "supportPublicIp": True,
    }
    pod = rest_request(spec.api_key, "POST", "/pods", pod_input)
    pod_id = str(pod["id"])
    observed_gpu = pod_gpu_type_id(pod)
    if observed_gpu and not same_gpu_type_id(
        observed_gpu, DEFAULT_H100_GPU_TYPE_ID
    ):
        terminate_pod(spec.api_key, pod_id)
        raise RuntimeError(
            "Runpod created a pod with an unexpected GPU type: "
            f"{observed_gpu!r}; requested {DEFAULT_H100_GPU_TYPE_ID!r}"
        )

    print(f"Created persistent Runpod H100 pod: {pod_id}", flush=True)
    return CreatedPod(
        id=pod_id, requested_gpu_type_ids=(DEFAULT_H100_GPU_TYPE_ID,)
    )


def terminate_pod(api_key: str, pod_id: str) -> None:
    rest_request(api_key, "DELETE", f"/pods/{pod_id}")
    print(f"Terminated Runpod pod: {pod_id}", flush=True)


def get_pod(api_key: str, pod_id: str) -> dict[str, Any]:
    return rest_request(api_key, "GET", f"/pods/{pod_id}")


def normalized_gpu_type_id(value: str) -> str:
    return " ".join(value.split()).casefold()


def same_gpu_type_id(left: str, right: str) -> bool:
    return normalized_gpu_type_id(left) == normalized_gpu_type_id(right)


def pod_gpu_type_id(pod: dict[str, Any]) -> str | None:
    machine = pod.get("machine") or {}
    machine_gpu_type = machine.get("gpuType") or {}
    gpu = pod.get("gpu") or {}

    for value in (
        machine.get("gpuTypeId"),
        machine_gpu_type.get("id"),
        gpu.get("id"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def gpu_in_requested_list(
    observed_gpu: str, requested_gpu_type_ids: Sequence[str]
) -> bool:
    return any(
        same_gpu_type_id(observed_gpu, gpu_type_id)
        for gpu_type_id in requested_gpu_type_ids
    )


def verify_requested_gpu_from_api(
    spec: EphemeralPodSpec | PersistentH100PodSpec,
    pod_id: str,
    requested_gpu_type_ids: Sequence[str],
) -> bool:
    deadline = time.monotonic() + spec.gpu_verify_timeout_seconds

    while time.monotonic() < deadline:
        pod = get_pod(spec.api_key, pod_id)
        observed_gpu = pod_gpu_type_id(pod)
        if observed_gpu and gpu_in_requested_list(
            observed_gpu, requested_gpu_type_ids
        ):
            print(f"Verified Runpod API GPU: {observed_gpu}", flush=True)
            return True
        if observed_gpu:
            raise RuntimeError(
                "Runpod assigned an unexpected GPU type: "
                f"{observed_gpu!r}; requested one of "
                f"{list(requested_gpu_type_ids)!r}"
            )

        print(
            "Waiting for Runpod API GPU assignment to be reported", flush=True
        )
        time.sleep(10)

    print(
        "Runpod API did not report the GPU type; will verify over SSH",
        flush=True,
    )
    return False


def verify_requested_gpu_from_ssh(
    config: RunpodConfig, requested_gpu_type_ids: Sequence[str]
) -> None:
    observed_gpu = ssh_output(
        config, "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1"
    )
    if gpu_in_requested_list(observed_gpu, requested_gpu_type_ids):
        print(f"Verified pod GPU: {observed_gpu}", flush=True)
        return

    raise RuntimeError(
        "Pod has an unexpected GPU type: "
        f"{observed_gpu!r}; requested one of {list(requested_gpu_type_ids)!r}"
    )


def ssh_connection_from_pod(
    pod: dict[str, Any], private_port: int
) -> tuple[str, str] | None:
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    public_port = port_mappings.get(str(private_port))
    if public_ip and public_port:
        return f"root@{public_ip}", str(public_port)
    return None


def wait_for_ssh(
    spec: EphemeralPodSpec | PersistentH100PodSpec, pod_id: str
) -> RunpodConfig:
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


def check_persistent_h100_config(spec: PersistentH100PodSpec) -> None:
    print(f"RUNPOD_DATACENTER_ID={spec.data_center_id}")
    print(f"RUNPOD_NETWORK_VOLUME_ID={spec.network_volume_id}")
    print(f"RUNPOD_NETWORK_VOLUME_NAME={spec.network_volume_name}")
    print(f"RUNPOD_NETWORK_VOLUME_SIZE_GB={spec.network_volume_size_gb}")
    print(f"RUNPOD_POD_ID={os.environ.get('RUNPOD_POD_ID')}")
    print(f"RUNPOD_POD_NAME={spec.name}")
    print(f"RUNPOD_IMAGE_NAME={spec.image_name}")
    print(f"RUNPOD_CLOUD_TYPE={spec.cloud_type}")
    print(f"RUNPOD_CONTAINER_DISK_GB={spec.container_disk_gb}")
    print(f"RUNPOD_MIN_VCPU={spec.min_vcpu}")
    print(f"RUNPOD_MIN_MEMORY_GB={spec.min_memory_gb}")
    print(f"RUNPOD_RETRY_SLEEP_SECONDS={spec.retry_sleep_seconds}")
    max_attempts = "infinite" if spec.max_attempts == 0 else spec.max_attempts
    print(f"RUNPOD_MAX_ATTEMPTS={max_attempts}")


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
            "uv run hf download meta-llama/Llama-3.2-1B-Instruct "
            "--repo-type model "
            "&& uv run hf download meta-llama/Llama-3.2-3B-Instruct "
            "--repo-type model",
        ),
    )


def baseline(config: RunpodConfig, extra_args: Sequence[str]) -> None:
    joined_args = " ".join(shlex.quote(arg) for arg in extra_args)
    command = f"uv run python scripts/baseline.py --device cuda {joined_args}"
    ssh(config, in_project(config, command.strip()))


def prepare_datasets(config: RunpodConfig, extra_args: Sequence[str]) -> None:
    joined_args = " ".join(shlex.quote(arg) for arg in extra_args)
    command = f"uv run python scripts/prepare_datasets.py {joined_args}"
    ssh(config, in_project(config, command.strip()))


def exec_remote(config: RunpodConfig, command: str) -> None:
    ssh(config, in_project(config, command))


def run_all(config: RunpodConfig, extra_args: Sequence[str]) -> None:
    sync(config)
    setup(config)
    gpu_check(config)
    download_models(config)
    baseline(config, extra_args)


def save_persistent_pod_env(
    spec: PersistentH100PodSpec,
    network_volume_id: str,
    pod_id: str,
    config: RunpodConfig,
) -> None:
    values = {
        "RUNPOD_DATACENTER_ID": spec.data_center_id,
        "RUNPOD_NETWORK_VOLUME_ID": network_volume_id,
        "RUNPOD_POD_ID": pod_id,
        "RUNPOD_PROJECT_DIR": config.remote_dir,
        "RUNPOD_SSH_PORT": config.ssh_port,
        "RUNPOD_SSH_TARGET": config.ssh_target,
    }
    write_local_env(values)
    print("\nPersistent Runpod H100 pod is ready:", flush=True)
    for key in sorted(values):
        print(shell_export(key, values[key]), flush=True)
    print(
        "\nSubsequent runpod tasks will read these values from "
        f"{LOCAL_ENV_FILE} automatically.",
        flush=True,
    )


def connect_existing_persistent_pod(
    spec: PersistentH100PodSpec, network_volume_id: str
) -> bool:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        return False
    print(f"Checking existing Runpod pod from {LOCAL_ENV_FILE}: {pod_id}")
    try:
        get_pod(spec.api_key, pod_id)
        if os.environ.get("RUNPOD_SSH_TARGET"):
            config = RunpodConfig.from_env()
            verify_requested_gpu_from_ssh(config, (DEFAULT_H100_GPU_TYPE_ID,))
        else:
            api_verified = verify_requested_gpu_from_api(
                spec, pod_id, (DEFAULT_H100_GPU_TYPE_ID,)
            )
            config = wait_for_ssh(spec, pod_id)
            if not api_verified:
                verify_requested_gpu_from_ssh(
                    config, (DEFAULT_H100_GPU_TYPE_ID,)
                )
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        urllib.error.URLError,
    ) as exc:
        print(f"Existing pod is not usable yet: {exc}", flush=True)
        return False

    save_persistent_pod_env(spec, network_volume_id, pod_id, config)
    return True


def run_persistent_h100() -> None:
    spec = PersistentH100PodSpec.from_env()
    network_volume_id = ensure_network_volume(spec)
    if connect_existing_persistent_pod(spec, network_volume_id):
        return

    attempt = 0
    while spec.max_attempts == 0 or attempt < spec.max_attempts:
        attempt += 1
        print(f"Starting H100 pod attempt {attempt}", flush=True)
        pod: CreatedPod | None = None
        try:
            pod = create_persistent_h100_pod(spec, network_volume_id)
            api_verified = verify_requested_gpu_from_api(
                spec, pod.id, pod.requested_gpu_type_ids
            )
            config = wait_for_ssh(spec, pod.id)
            if not api_verified:
                verify_requested_gpu_from_ssh(
                    config, pod.requested_gpu_type_ids
                )
            gpu_check(config)
            save_persistent_pod_env(spec, network_volume_id, pod.id, config)
            return
        except (
            RuntimeError,
            subprocess.CalledProcessError,
            urllib.error.URLError,
        ) as exc:
            print(f"H100 pod attempt {attempt} failed: {exc}", flush=True)
            if pod is not None:
                try:
                    terminate_pod(spec.api_key, pod.id)
                except RuntimeError as terminate_exc:
                    print(
                        "Failed to terminate unsuccessful pod "
                        f"{pod.id}: {terminate_exc}",
                        flush=True,
                    )
            if spec.max_attempts != 0 and attempt >= spec.max_attempts:
                raise
            print(
                f"Retrying in {spec.retry_sleep_seconds} seconds...",
                flush=True,
            )
            time.sleep(spec.retry_sleep_seconds)


def run_ephemeral(extra_args: Sequence[str]) -> None:
    spec = EphemeralPodSpec.from_env()
    pod = create_pod(spec)
    try:
        api_verified = verify_requested_gpu_from_api(
            spec, pod.id, pod.requested_gpu_type_ids
        )
        config = wait_for_ssh(spec, pod.id)
        if not api_verified:
            verify_requested_gpu_from_ssh(config, pod.requested_gpu_type_ids)
        run_all(config, extra_args)
    finally:
        if spec.keep_pod:
            print(f"Keeping Runpod pod for debugging: {pod.id}", flush=True)
        else:
            terminate_pod(spec.api_key, pod.id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-config")
    subparsers.add_parser("check-ephemeral-config")
    subparsers.add_parser("check-h100-config")
    subparsers.add_parser("persistent-h100")
    subparsers.add_parser("sync")
    subparsers.add_parser("setup")
    subparsers.add_parser("gpu-check")
    subparsers.add_parser("download-models")

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    prepare_datasets_parser = subparsers.add_parser("prepare-datasets")
    prepare_datasets_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

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
            case "check-h100-config":
                check_persistent_h100_config(PersistentH100PodSpec.from_env())
            case "persistent-h100":
                run_persistent_h100()
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
            case "prepare-datasets":
                prepare_datasets(RunpodConfig.from_env(), args.extra_args)
            case "all":
                run_all(RunpodConfig.from_env(), args.extra_args)
            case "exec":
                exec_remote(RunpodConfig.from_env(), args.remote_command)
            case _:
                raise ValueError(f"Unknown command: {args.command}")
    except (
        RuntimeError,
        subprocess.CalledProcessError,
        urllib.error.URLError,
        ValueError,
    ) as exc:
        print(f"runpod: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
