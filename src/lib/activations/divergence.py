"""Per-layer harmful/benign divergence diagnostics."""

from __future__ import annotations

from typing import Any

import torch


def layerwise_cosine(
    harmful: torch.Tensor, benign: torch.Tensor
) -> dict[str, list[float]]:
    """Per-layer divergence stats between mean harmful and benign activations.

    A low cosine similarity (and a large normalized difference) at a layer
    means harmful and benign prompts are linearly separable there — the
    signature of a refusal-mediating layer.
    """
    mean_harmful = harmful.mean(dim=0)  # [n_layers, d_model]
    mean_benign = benign.mean(dim=0)
    diff = mean_harmful - mean_benign

    cosine = torch.nn.functional.cosine_similarity(
        mean_harmful, mean_benign, dim=1
    )
    # Difference norm relative to the typical activation norm at that layer.
    mean_norm = 0.5 * (mean_harmful.norm(dim=1) + mean_benign.norm(dim=1))
    relative_diff = diff.norm(dim=1) / mean_norm.clamp_min(1e-6)

    return {
        "cosine_similarity": cosine.tolist(),
        "diff_norm": diff.norm(dim=1).tolist(),
        "relative_diff_norm": relative_diff.tolist(),
    }


def summarize_best_layer(summary: dict[str, list[float]]) -> dict[str, Any]:
    """Identify the layer where harmful/benign separation peaks."""
    cosine = summary["cosine_similarity"]
    relative = summary["relative_diff_norm"]
    min_cosine_layer = min(range(len(cosine)), key=lambda i: cosine[i])
    max_diff_layer = max(range(len(relative)), key=lambda i: relative[i])
    return {
        "min_cosine_layer": min_cosine_layer,
        "min_cosine_value": cosine[min_cosine_layer],
        "max_relative_diff_layer": max_diff_layer,
        "max_relative_diff_value": relative[max_diff_layer],
    }
