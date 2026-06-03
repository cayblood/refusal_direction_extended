"""Persistence for activation tensors and direction blobs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


def save_activations(
    path: Path,
    *,
    model_name: str,
    harmful: torch.Tensor,
    benign: torch.Tensor,
    offsets: Sequence[int],
    harmful_records: Sequence[dict[str, Any]],
    benign_records: Sequence[dict[str, Any]],
    position_diag: dict[str, Any],
) -> None:
    """Persist activations and provenance for the direction-extraction step.

    Tensors have shape ``[n_prompts, num_positions, n_layers, d_model]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "hook": "resid_post",
            "position": "last_instruction_tokens",
            "position_offsets": list(offsets),
            "num_positions": harmful.shape[1],
            "n_layers": harmful.shape[2],
            "d_model": harmful.shape[3],
            "harmful": harmful,
            "benign": benign,
            "harmful_pair_ids": [r["pair_id"] for r in harmful_records],
            "benign_pair_ids": [r["pair_id"] for r in benign_records],
            "position_diagnostic": position_diag,
        },
        path,
    )


def load_pt(path: Path, device: str = "cpu") -> dict[str, Any]:
    """Load a tensor blob saved by the pipeline (tensors + simple data only)."""
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. Run earlier pipeline stages first."
        )
    return torch.load(path, map_location=device, weights_only=True)
