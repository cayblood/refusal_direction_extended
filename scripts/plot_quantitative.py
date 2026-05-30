"""Plot the quantitative results: addition sweep + the 2x2 table.

Reads ``quantitative_2x2.json`` written by ``evaluate_quantitative.py`` and, for
each model, draws two panels:

* the benign refusal rate as a function of the addition strength ``alpha``
  (sufficiency: how hard you must push before benign prompts get refused), and
* the headline 2x2 grouped bars (baseline vs. intervention, harmful vs. benign).

Runs on CPU; no model or GPU required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def plot_quantitative(data: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = data["addition_sweep"]
    alphas = [row["alpha"] for row in sweep]
    benign = [row["benign_refusal_rate"] for row in sweep]
    chosen_alpha = data["chosen_alpha"]
    table = data["table"]

    fig, (ax_sweep, ax_bars) = plt.subplots(1, 2, figsize=(12, 5))

    ax_sweep.plot(alphas, benign, marker="o", color="tab:red")
    ax_sweep.axvline(
        chosen_alpha,
        color="grey",
        linestyle="--",
        label=f"chosen alpha={chosen_alpha:g}",
    )
    ax_sweep.axhline(
        data["addition_threshold"], color="tab:blue", linestyle=":", alpha=0.6
    )
    ax_sweep.set_xlabel("addition strength alpha (units of raw diff norm)")
    ax_sweep.set_ylabel("benign refusal rate")
    ax_sweep.set_title("Direction addition (sufficiency)")
    ax_sweep.set_ylim(-0.02, 1.02)
    ax_sweep.grid(True, alpha=0.3)
    ax_sweep.legend(fontsize=8)

    groups = ["harmful", "benign"]
    baseline = [
        table["baseline"]["harmful_refusal_rate"],
        table["baseline"]["benign_refusal_rate"],
    ]
    intervention = [
        table["intervention"]["harmful_refusal_rate_ablated"],
        table["intervention"]["benign_refusal_rate_added"],
    ]
    x = range(len(groups))
    width = 0.38
    ax_bars.bar(
        [i - width / 2 for i in x],
        baseline,
        width,
        label="baseline",
        color="tab:gray",
    )
    ax_bars.bar(
        [i + width / 2 for i in x],
        intervention,
        width,
        label="ablation / addition",
        color="tab:purple",
    )
    ax_bars.set_xticks(list(x))
    ax_bars.set_xticklabels(groups)
    ax_bars.set_ylabel("refusal rate")
    ax_bars.set_title("Refusal 2x2")
    ax_bars.set_ylim(0, 1.02)
    ax_bars.grid(True, axis="y", alpha=0.3)
    ax_bars.legend(fontsize=8)

    fig.suptitle(
        f"Refusal direction: necessity and sufficiency\n{data['model']}"
    )
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
