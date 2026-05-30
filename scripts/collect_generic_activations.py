"""Collect residual activations for the neutral generic instruction set.

Mirrors ``collect_activations.py`` but for a single, unlabelled generic class,
writing ``resid_post_generic.pt`` per model. These paired activations are the
independent fitting distribution for the cross-scale transfer control
(``evaluate_transfer_independent.py``): a different prompt distribution from
the harmful/benign prompts that define the refusal direction.

Reuses the model loader and the last-K activation capture from
``collect_activations`` so the activations are produced identically to the main
pipeline (same hook, same end-relative positions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import torch
from collect_activations import (
    collect_last_k_activations,
    load_model,
    load_records,
    position_offsets,
)
from evaluate_ablation import pick_device

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")
DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_NUM_POSITIONS = 5


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def collect_for_model(
    model_name: str,
    *,
    device: str,
    dataset_dir: Path,
    activations_dir: Path,
    label: str,
    num_positions: int,
    batch_size: int,
    limit: int | None,
) -> int:
    records = load_records(dataset_dir, label, limit)
    prompts = [cast(str, row["formatted_prompt"]) for row in records]
    print(f"\n# MODEL: {model_name}", flush=True)
    print(f"Loading on {device}; {len(prompts)} generic prompts.", flush=True)
    model = load_model(model_name, device)

    acts = collect_last_k_activations(
        model,
        prompts,
        device=device,
        batch_size=batch_size,
        num_positions=num_positions,
    )
    offsets = position_offsets(num_positions)
    out_path = (
        activations_dir / model_slug(model_name) / "resid_post_generic.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "hook": "resid_post",
            "label": label,
            "position_offsets": offsets,
            "num_positions": num_positions,
            "n_layers": acts.shape[2],
            "d_model": acts.shape[3],
            "generic": acts,
            "generic_pair_ids": [int(row["pair_id"]) for row in records],
        },
        out_path,
    )
    print(f"Saved {tuple(acts.shape)} -> {out_path}", flush=True)
    return 0


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
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR)
    )
    parser.add_argument("--label", default="generic")
    parser.add_argument(
        "--num-positions", type=int, default=DEFAULT_NUM_POSITIONS
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_cuda = not cast(bool, args.allow_local)
    device = (
        pick_device(require_cuda=require_cuda)
        if args.device == "auto"
        else cast(str, args.device)
    )
    if require_cuda and device != "cuda":
        print(
            "CUDA is required by default. Use --allow-local to run on "
            f"{device!r} intentionally.",
            file=sys.stderr,
        )
        return 2

    models = cast(list[str], args.models) if args.models else DEFAULT_MODELS
    torch.set_grad_enabled(False)
    exit_code = 0
    for model_name in models:
        exit_code = max(
            exit_code,
            collect_for_model(
                model_name,
                device=device,
                dataset_dir=Path(cast(str, args.dataset_dir)),
                activations_dir=Path(cast(str, args.activations_dir)),
                label=cast(str, args.label),
                num_positions=cast(int, args.num_positions),
                batch_size=cast(int, args.batch_size),
                limit=cast("int | None", args.limit),
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
