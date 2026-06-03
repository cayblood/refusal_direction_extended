"""Difference-in-means candidate refusal directions."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def difference_in_means(
    harmful: torch.Tensor, benign: torch.Tensor, train: Sequence[int]
) -> torch.Tensor:
    """Unit-norm difference-in-means per (position, layer): [K, n_layers, d].

    ``harmful``/``benign`` are ``[n_prompts, K, n_layers, d_model]``.
    """
    index = torch.tensor(train, dtype=torch.long)
    mean_harmful = harmful[index].mean(dim=0)  # [K, n_layers, d_model]
    mean_benign = benign[index].mean(dim=0)
    diff = mean_harmful - mean_benign
    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return diff / norm


def raw_diff_norms(
    harmful: torch.Tensor, benign: torch.Tensor, train: Sequence[int]
) -> torch.Tensor:
    """Unnormalized difference norm per (position, layer): [K, n_layers]."""
    index = torch.tensor(train, dtype=torch.long)
    diff = harmful[index].mean(dim=0) - benign[index].mean(dim=0)
    return diff.norm(dim=-1)
