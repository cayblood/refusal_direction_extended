"""Plot the addition sweep and 2x2 table for each model (entry point, CPU)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make src/ importable so `from lib...` resolves without an editable
# install (e.g. on a fresh Colab runtime, where only PYTHONPATH or an
# install would otherwise expose the package).
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from pathlib import Path
from typing import cast

from lib.plots import plot_quantitative
from lib.runtime import model_slug

DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
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
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = cast(list[str], args.models) if args.models else DEFAULT_MODELS
    artifacts_dir = Path(cast(str, args.artifacts_dir))

    for model_name in models:
        slug = model_slug(model_name)
        data_path = artifacts_dir / slug / "quantitative_2x2.json"
        if not data_path.exists():
            print(f"Skipping {slug}: {data_path} not found")
            continue
        data = json.loads(data_path.read_text())
        output_path = artifacts_dir / slug / "quantitative_2x2.png"
        plot_quantitative(data, output_path)
        print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
