"""Plot per-layer harmful/benign activation divergence."""

from __future__ import annotations

from pathlib import Path


def plot_divergence(
    summary: dict[str, list[float]], model_name: str, output_path: Path
) -> None:
    """Plot cosine similarity and relative difference norm across layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cosine = summary["cosine_similarity"]
    relative = summary["relative_diff_norm"]
    layers = list(range(len(cosine)))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    top.plot(layers, cosine, marker="o", color="#c0392b")
    top.set_ylabel("cosine(mean_harmful, mean_benign)")
    top.set_title(f"Harmful vs. benign activation divergence\n{model_name}")
    top.grid(True, alpha=0.3)

    bottom.plot(layers, relative, marker="o", color="#2c3e50")
    bottom.set_ylabel("||Δ mean|| / mean ||activation||")
    bottom.set_xlabel("layer (resid_post)")
    bottom.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
