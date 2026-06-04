"""Inspect benign completions under direction addition (entry point)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make src/ importable so `from lib...` resolves without an editable
# install (e.g. on a fresh Colab runtime, where only PYTHONPATH or an
# install would otherwise expose the package).
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path
from typing import cast

import torch

from lib.interventions.inspection import DEFAULT_ALPHAS, inspect_for_model
from lib.runtime import resolve_device

DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Single HF model id (defaults to the 1B Instruct model).",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument(
        "--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR)
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--position-index", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS
    )
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--eval-tokens", type=int, default=48)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(
        cast(str, args.device), allow_local=cast(bool, args.allow_local)
    )
    torch.set_grad_enabled(False)
    return inspect_for_model(
        cast(str, args.model),
        device=device,
        activations_dir=Path(cast(str, args.activations_dir)),
        artifacts_dir=Path(cast(str, args.artifacts_dir)),
        dataset_dir=Path(cast(str, args.dataset_dir)),
        position_index=cast("int | None", args.position_index),
        layer=cast("int | None", args.layer),
        alphas=cast(list[float], args.alphas),
        n_prompts=cast(int, args.n_prompts),
        eval_tokens=cast(int, args.eval_tokens),
        gen_batch_size=cast(int, args.gen_batch_size),
    )


if __name__ == "__main__":
    raise SystemExit(main())
