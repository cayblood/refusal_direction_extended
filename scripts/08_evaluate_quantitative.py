"""Quantitative ablation + addition 2x2 on the test split (entry point)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch

from lib.interventions.quantitative import DEFAULT_ALPHAS, evaluate_for_model
from lib.runtime import resolve_device

DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="HF model id; repeatable",
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
    parser.add_argument(
        "--position-index",
        type=int,
        default=None,
        help="Override the position index (default: best ablation candidate).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override the source layer (default: best ablation candidate).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Addition strengths in units of the raw diff-in-means norm.",
    )
    parser.add_argument(
        "--addition-threshold",
        type=float,
        default=0.5,
        help="Benign refusal rate the chosen alpha must reach.",
    )
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(
        cast(str, args.device), allow_local=cast(bool, args.allow_local)
    )
    models = cast(list[str], args.models) if args.models else DEFAULT_MODELS
    torch.set_grad_enabled(False)

    exit_code = 0
    for model_name in models:
        exit_code = max(
            exit_code,
            evaluate_for_model(
                model_name,
                device=device,
                activations_dir=Path(cast(str, args.activations_dir)),
                artifacts_dir=Path(cast(str, args.artifacts_dir)),
                dataset_dir=Path(cast(str, args.dataset_dir)),
                position_index=cast("int | None", args.position_index),
                layer=cast("int | None", args.layer),
                alphas=cast(list[float], args.alphas),
                addition_threshold=cast(float, args.addition_threshold),
                eval_prompts=cast(int, args.eval_prompts),
                eval_tokens=cast(int, args.eval_tokens),
                example_count=cast(int, args.example_count),
                gen_batch_size=cast(int, args.gen_batch_size),
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
