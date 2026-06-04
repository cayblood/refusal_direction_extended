"""Plot the ablation sweep: refusal-bypass rate by layer and position."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def plot_sweep(sweep: dict[str, Any], output_path: Path) -> None:
    import os

    # Force a headless backend before importing matplotlib (Colab exports an
    # invalid MPLBACKEND that crashes matplotlib at import in a subprocess).
    os.environ["MPLBACKEND"] = "Agg"
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
