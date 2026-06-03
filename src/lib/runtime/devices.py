"""Device and dtype selection for model execution."""

from __future__ import annotations

import gc
import sys

import torch


def release_memory(device: str) -> None:
    """Collect garbage and empty the accelerator cache after model use."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def pick_device(require_cuda: bool) -> str:
    """Choose the runtime device for model execution."""
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError(
            "CUDA is required for this task. Use a GPU runtime or pass "
            "--allow-local for an explicitly local CPU/MPS run."
        )
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device_arg: str, *, allow_local: bool) -> str:
    """Resolve ``--device``/``--allow-local`` to a device string.

    Default requires CUDA; ``--allow-local`` permits CPU/MPS. Exits with code 2
    (matching the entry-script contract) if CUDA is required but an explicit
    non-CUDA device was requested.
    """
    require_cuda = not allow_local
    device = (
        pick_device(require_cuda=require_cuda)
        if device_arg == "auto"
        else device_arg
    )
    if require_cuda and device != "cuda":
        print(
            "CUDA is required by default. Use --allow-local to run on "
            f"{device!r} intentionally.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return device


def torch_dtype_for_device(device: str) -> torch.dtype:
    """Lower precision on accelerators, float32 on CPU for compatibility."""
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def default_dtype_name(device: str) -> str:
    """CLI dtype name matching :func:`torch_dtype_for_device`."""
    return "float16" if device in {"cuda", "mps"} else "float32"


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    """Convert a CLI dtype name into a torch dtype."""
    match dtype:
        case "float32":
            return torch.float32
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")
