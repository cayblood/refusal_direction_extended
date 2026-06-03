"""Deterministic train/val/test splits over prompt indices."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def split_indices(
    n: int, train_size: int, val_size: int, seed: int
) -> dict[str, list[int]]:
    """Deterministically split ``[0, n)`` into train/val/test index lists."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).tolist()
    train = permutation[:train_size]
    val = permutation[train_size : train_size + val_size]
    test = permutation[train_size + val_size :]
    return {"train": train, "val": val, "test": test}


def split_pair_ids(
    split: dict[str, list[int]],
    harmful_pair_ids: Sequence[int],
    benign_pair_ids: Sequence[int],
) -> dict[str, dict[str, list[int]]]:
    """Map index splits onto dataset ``pair_id`` lists for each class."""
    return {
        name: {
            "harmful": [harmful_pair_ids[i] for i in idx],
            "benign": [benign_pair_ids[i] for i in idx],
        }
        for name, idx in split.items()
    }
