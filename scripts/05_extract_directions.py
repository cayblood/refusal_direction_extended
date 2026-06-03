"""Extract difference-in-means candidate refusal directions (entry point).

For every captured ``(position, layer)`` pair, compute the unit-norm
difference-in-means direction between harmful and benign prompts on a *train*
split; held-out val/test pairs are reserved for the ablation sweep and the
quantitative evaluation. The split (by ``pair_id``) is saved alongside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

import torch

from lib.activations import (
    difference_in_means,
    load_pt,
    raw_diff_norms,
    split_indices,
    split_pair_ids,
)
from lib.runtime import model_slug

DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
DEFAULT_TRAIN_SIZE = 128
DEFAULT_VAL_SIZE = 64


def extract_for_model(
    model_name: str,
    *,
    activations_dir: Path,
    artifacts_dir: Path,
    train_size: int,
    val_size: int,
    seed: int,
) -> int:
    slug = model_slug(model_name)
    activations_path = activations_dir / slug / "resid_post.pt"
    if not activations_path.exists():
        print(
            f"Missing {activations_path}. Run collect-activations first.",
            file=sys.stderr,
        )
        return 1

    blob = load_pt(activations_path, "cpu")
    harmful = cast(torch.Tensor, blob["harmful"])
    benign = cast(torch.Tensor, blob["benign"])
    offsets = cast(list[int], blob["position_offsets"])
    n_prompts, num_positions, n_layers, d_model = harmful.shape
    print(f"\n# MODEL: {model_name}", flush=True)
    print(
        f"Loaded activations {tuple(harmful.shape)} (positions {offsets}).",
        flush=True,
    )

    if train_size + val_size >= n_prompts:
        raise RuntimeError(
            f"train_size + val_size ({train_size + val_size}) must be < "
            f"n_prompts ({n_prompts})"
        )
    split = split_indices(n_prompts, train_size, val_size, seed)
    harmful_pair_ids = cast(list[int], blob["harmful_pair_ids"])
    benign_pair_ids = cast(list[int], blob["benign_pair_ids"])
    pair_id_splits = split_pair_ids(split, harmful_pair_ids, benign_pair_ids)

    directions = difference_in_means(harmful, benign, split["train"])
    norms = raw_diff_norms(harmful, benign, split["train"])

    directions_path = activations_dir / slug / "directions.pt"
    torch.save(
        {
            "model": model_name,
            "directions": directions,  # [K, n_layers, d_model], unit norm
            "raw_diff_norm": norms,  # [K, n_layers]
            "position_offsets": offsets,
            "num_positions": num_positions,
            "n_layers": n_layers,
            "d_model": d_model,
            "train_size": train_size,
            "val_size": val_size,
            "seed": seed,
            "split_indices": split,
            "split_pair_ids": pair_id_splits,
        },
        directions_path,
    )
    print(f"Saved directions {tuple(directions.shape)}: {directions_path}")

    # Identify the (position, layer) candidate with the largest raw separation.
    flat_index = int(torch.argmax(norms))
    best_pos = flat_index // n_layers
    best_layer = flat_index % n_layers
    summary = {
        "model": model_name,
        "num_positions": num_positions,
        "position_offsets": offsets,
        "n_layers": n_layers,
        "d_model": d_model,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "max_raw_diff": {
            "position_offset": offsets[best_pos],
            "position_index": best_pos,
            "layer": best_layer,
            "value": float(norms[best_pos, best_layer]),
        },
        "raw_diff_norm_by_position_layer": norms.tolist(),
    }
    summary_path = artifacts_dir / slug / "directions_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved summary: {summary_path}")
    print(
        "Largest raw separation at "
        f"position_offset={offsets[best_pos]}, layer={best_layer} "
        f"(norm={float(norms[best_pos, best_layer]):.3f})",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="HF model id; may be repeated. Defaults to both Llama 3.2 sizes.",
    )
    parser.add_argument(
        "--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR)
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = cast(list[str], args.models) if args.models else DEFAULT_MODELS
    torch.set_grad_enabled(False)

    exit_code = 0
    for model_name in models:
        exit_code = max(
            exit_code,
            extract_for_model(
                model_name,
                activations_dir=Path(cast(str, args.activations_dir)),
                artifacts_dir=Path(cast(str, args.artifacts_dir)),
                train_size=cast(int, args.train_size),
                val_size=cast(int, args.val_size),
                seed=cast(int, args.seed),
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
