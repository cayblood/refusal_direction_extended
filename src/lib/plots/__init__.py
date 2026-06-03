"""Matplotlib figures for the pipeline (CPU-only)."""

from lib.plots.ablation import plot_sweep
from lib.plots.divergence import plot_divergence
from lib.plots.quantitative import plot_quantitative

__all__ = [
    "plot_divergence",
    "plot_quantitative",
    "plot_sweep",
]
