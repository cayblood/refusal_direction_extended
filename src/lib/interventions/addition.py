"""Direction addition: inject a scaled direction into the residual stream."""

from __future__ import annotations

from typing import Any

import torch


def make_addition_hook(vector: torch.Tensor):
    """Add a fixed steering ``vector`` to every position of the residual tensor.

    Broadcasts over batch and sequence (including tokens generated under a KV
    cache), so the steering signal persists for the whole completion.
    """

    def hook(tensor: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
        return tensor + vector.to(tensor.dtype)

    return hook


def addition_hooks(vector: torch.Tensor, layer: int) -> list[tuple[str, Any]]:
    """Add ``vector`` at the residual stream of a single source ``layer``."""
    return [(f"blocks.{layer}.hook_resid_post", make_addition_hook(vector))]
