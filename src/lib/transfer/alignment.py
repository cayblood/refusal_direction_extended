"""Linear alignment between two activation spaces (ridge-regularized)."""

from __future__ import annotations

import torch


def fit_linear_map(
    source: torch.Tensor, target: torch.Tensor, ridge_rel: float
) -> torch.Tensor:
    """Ridge-fit ``W_T`` s.t. ``source @ W_T ~= target``: shape [d_src, d_tgt].

    ``ridge_rel`` scales the penalty by the mean feature energy, so it adapts to
    activation magnitude. ``ridge_rel=0`` is ordinary least squares.
    """
    d_src = source.shape[1]
    gram = source.T @ source
    lam = ridge_rel * (torch.trace(gram) / d_src)
    a = gram + lam * torch.eye(d_src, dtype=source.dtype)
    b = source.T @ target
    return torch.linalg.solve(a, b)


def relative_reconstruction_error(
    source: torch.Tensor, target: torch.Tensor, w_t: torch.Tensor
) -> float:
    predicted = source @ w_t
    return float((predicted - target).norm() / target.norm().clamp_min(1e-8))
