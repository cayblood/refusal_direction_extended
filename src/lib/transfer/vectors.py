"""Anchor-matrix extraction and control vectors for transfer."""

from __future__ import annotations

from typing import Any

import torch


def paired_anchor_matrix(
    activations: dict[str, Any], pos_index: int, layer: int
) -> torch.Tensor:
    """Stack harmful+benign activations at one (position, layer): [2N, d]."""
    harmful = activations["harmful"][:, pos_index, layer, :].float()
    benign = activations["benign"][:, pos_index, layer, :].float()
    return torch.cat([harmful, benign], dim=0)


def generic_anchor_matrix(
    generic: dict[str, Any], pos_index: int, layer: int
) -> torch.Tensor:
    """Generic activations at one (position, layer): [n_generic, d]."""
    return generic["generic"][:, pos_index, layer, :].float()


def class_split_rows(
    split_indices: dict[str, list[int]], n_per_class: int
) -> dict[str, torch.Tensor]:
    """Map per-class split indices onto rows of a stacked harmful+benign matrix.

    Row ``i`` is harmful prompt ``i``; row ``n_per_class + i`` is benign prompt
    ``i`` (same prompt order across models, verified upstream).
    """
    rows = {}
    for name, idx in split_indices.items():
        harmful_rows = idx
        benign_rows = [n_per_class + i for i in idx]
        rows[name] = torch.tensor(harmful_rows + benign_rows, dtype=torch.long)
    return rows


def random_unit_direction(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(dim, generator=generator)
    return vector / vector.norm().clamp_min(1e-8)
