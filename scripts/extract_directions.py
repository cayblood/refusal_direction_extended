"""Extract candidate refusal directions from cached activations.

Reads the residual-stream activations produced by ``collect_activations.py`` and
computes, for every captured ``(position, layer)`` pair, the difference-in-means
direction between harmful and benign prompts:

    r[pos, layer] = mean(harmful[:, pos, layer]) - mean(benign[:, pos, layer])

each normalized to unit length. Following Arditi et al., the means are taken
over a *train* split only; held-out *val* and *test* pairs are reserved for the
ablation sweep and final quantitative evaluation. The split
(by dataset ``pair_id``) is stored alongside the directions so later steps
evaluate on prompts that never informed the directions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch

DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
DEFAULT_TRAIN_SIZE = 128
DEFAULT_VAL_SIZE = 64


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def split_indices(
    n: int, train_size: int, val_size: int, seed: int
) -> dict[str, list[int]]:
    """Deterministically split [0, n) into train/val/test index lists."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    train = permutation[:train_size]
    val = permutation[train_size : train_size + val_size]
    test = permutation[train_size + val_size :]
    return {"train": train, "val": val, "test": test}


def difference_in_means(
    harmful: torch.Tensor, benign: torch.Tensor, train: Sequence[int]
) -> torch.Tensor:
    """Unit-norm difference-in-means per (position, layer): [K, n_layers, d].

    ``harmful``/``benign`` are ``[n_prompts, K, n_layers, d_model]``.
    """
    index = torch.tensor(train, dtype=torch.long)
    mean_harmful = harmful[index].mean(dim=0)  # [K, n_layers, d_model]
    mean_benign = benign[index].mean(dim=0)
    diff = mean_harmful - mean_benign
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return diff / norm


def raw_diff_norms(
    harmful: torch.Tensor, benign: torch.Tensor, train: Sequence[int]
) -> torch.Tensor:
    """Unnormalized difference norm per (position, layer): [K, n_layers]."""
    index = torch.tensor(train, dtype=torch.long)
    diff = harmful[index].mean(dim=0) - benign[index].mean(dim=0)
    return diff.norm(dim=-1)


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
            f"Missing {activations_path}. Run collect_activations.py first.",
            file=sys.stderr,
        )
        return 1

    blob = torch.load(activations_path, map_location="cpu")
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
    split_pair_ids = {
        name: {
            "harmful": [harmful_pair_ids[i] for i in idx],
            "benign": [benign_pair_ids[i] for i in idx],
        }
        for name, idx in split.items()
    }

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
            "split_pair_ids": split_pair_ids,
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
