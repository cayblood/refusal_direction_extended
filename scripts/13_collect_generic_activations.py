"""Collect residual activations for the generic instruction set (entry point).

Mirrors collect-activations but for a single unlabelled generic class, writing
``resid_post_generic.pt`` per model — the independent fitting distribution for
the cross-scale transfer control.
"""

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

from lib.activations import collect_last_k_activations, position_offsets
from lib.data import load_records
from lib.runtime import load_model, model_slug, release_memory, resolve_device

DEFAULT_DATASET_DIR = Path("data/refusal_datasets")
DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_NUM_POSITIONS = 5
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


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

    del model
    release_memory(device)
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
    device = resolve_device(
        cast(str, args.device), allow_local=cast(bool, args.allow_local)
    )
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
