"""Cross-scale linear transfer of the refusal direction (entry point)."""

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

from lib.runtime import resolve_device
from lib.transfer import run_transfer

DEFAULT_SOURCE = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TARGET = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE)
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
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
        "--ridge-rel",
        type=float,
        default=1e-3,
        help="Ridge penalty relative to mean feature energy (0 = OLS).",
    )
    parser.add_argument(
        "--fit-split",
        default="val",
        choices=["train", "val", "test"],
        help="Split used to fit the map; must differ from the 'train' split "
        "that defines the directions to avoid circularity.",
    )
    parser.add_argument(
        "--recon-split",
        default="test",
        choices=["train", "val", "test"],
        help="Split used to measure held-out reconstruction error.",
    )
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(
        cast(str, args.device), allow_local=cast(bool, args.allow_local)
    )
    torch.set_grad_enabled(False)
    return run_transfer(
        source_model=cast(str, args.source_model),
        target_model=cast(str, args.target_model),
        device=device,
        activations_dir=Path(cast(str, args.activations_dir)),
        artifacts_dir=Path(cast(str, args.artifacts_dir)),
        dataset_dir=Path(cast(str, args.dataset_dir)),
        ridge_rel=cast(float, args.ridge_rel),
        fit_split=cast(str, args.fit_split),
        recon_split=cast(str, args.recon_split),
        eval_prompts=cast(int, args.eval_prompts),
        eval_tokens=cast(int, args.eval_tokens),
        gen_batch_size=cast(int, args.gen_batch_size),
        random_seed=cast(int, args.random_seed),
    )


if __name__ == "__main__":
    raise SystemExit(main())
