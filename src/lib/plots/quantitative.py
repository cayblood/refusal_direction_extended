"""Plot the addition sweep and the headline 2x2 table."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
