"""Selecting which (position, layer) candidate direction to use."""

from __future__ import annotations

import json
from pathlib import Path


def candidate_layers(n_layers: int, layer_step: int) -> list[int]:
    return list(range(0, n_layers, layer_step))


def best_anchor(artifact_subdir: Path) -> tuple[int, int, int]:
    """(position_index, layer, position_offset) of the best ablation pick."""
    best = json.loads((artifact_subdir / "ablation_best.json").read_text())[
        "best"
    ]
    return (
        int(best["position_index"]),
        int(best["layer"]),
        int(best["position_offset"]),
    )


def choose_candidate(
    artifact_subdir: Path,
    position_index: int | None,
    layer: int | None,
) -> tuple[int, int, int]:
    """Resolve the (position_index, layer, position_offset) to evaluate.

    Defaults to the best ablation candidate recorded in ``ablation_best.json``;
    explicit ``position_index``/``layer`` override it.
    """
    best_path = artifact_subdir / "ablation_best.json"
    if not best_path.exists():
        raise RuntimeError(
            f"Missing {best_path}. Run evaluate_ablation first, or pass "
            "--position-index and --layer explicitly."
        )
    best = json.loads(best_path.read_text())["best"]
    pos_index = (
        position_index if position_index is not None else best["position_index"]
    )
    chosen_layer = layer if layer is not None else best["layer"]
    return int(pos_index), int(chosen_layer), int(best["position_offset"])
