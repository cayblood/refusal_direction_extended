"""Directional ablation: project a direction out of the residual stream."""

from __future__ import annotations

from typing import Any

import torch


def make_ablation_hook(direction: torch.Tensor):
    """Project ``direction`` (unit norm) out of whatever residual tensor flows
    through the hook: ``x <- x - (x . d) d``."""

    def hook(tensor: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
        d = direction.to(tensor.dtype)
        projection = (tensor @ d).unsqueeze(-1) * d
        return tensor - projection

    return hook


def ablation_hooks(
    direction: torch.Tensor, n_layers: int
) -> list[tuple[str, Any]]:
    """Hook every residual-stream write point at every layer."""
    points = ("hook_resid_pre", "hook_resid_mid", "hook_resid_post")
    hook = make_ablation_hook(direction)
    return [
        (f"blocks.{layer}.{point}", hook)
        for layer in range(n_layers)
        for point in points
    ]
