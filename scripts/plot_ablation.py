"""Plot the ablation sweep: refusal-bypass rate by layer and position.

Reads the ``ablation_sweep.json`` artifacts written by ``evaluate_ablation.py``
and draws, for each model, one line per captured token position showing how the
refusal-bypass rate varies with the layer the ablated direction was taken from.
Runs on CPU; no model or GPU required.

The peak of these curves is the most *causally effective* layer, which is
typically earlier than the layer of maximum representational separation shown in
the divergence plot.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def plot_sweep(sweep: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_offset: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in sweep["candidates"]:
        by_offset[row["position_offset"]].append(
            (row["layer"], row["bypass_rate"])
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset in sorted(by_offset):
        points = sorted(by_offset[offset])
        layers = [layer for layer, _ in points]
        bypass = [value for _, value in points]
        ax.plot(layers, bypass, marker="o", label=f"position {offset:+d}")

    baseline_bypass = 1.0 - sweep["baseline_refusal_rate"]
    ax.axhline(
        baseline_bypass,
        color="grey",
        linestyle="--",
        label=f"baseline bypass ({baseline_bypass:.2f})",
    )
    ax.set_xlabel("layer the ablated direction was extracted from")
    ax.set_ylabel("harmful refusal-bypass rate")
    ax.set_title(f"Directional ablation sweep\n{sweep['model']}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


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
        sweep_path = artifacts_dir / slug / "ablation_sweep.json"
        if not sweep_path.exists():
            print(f"Skipping {slug}: {sweep_path} not found")
            continue
        sweep = json.loads(sweep_path.read_text())
        output_path = artifacts_dir / slug / "ablation_sweep.png"
        plot_sweep(sweep, output_path)
        print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
