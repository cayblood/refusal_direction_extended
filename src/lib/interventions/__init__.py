"""Causal residual-stream edits (ablation, addition) and their evaluations."""

from lib.interventions.ablation import ablation_hooks, make_ablation_hook
from lib.interventions.addition import addition_hooks, make_addition_hook
from lib.interventions.candidates import (
    best_anchor,
    candidate_layers,
    choose_candidate,
)

__all__ = [
    "ablation_hooks",
    "addition_hooks",
    "best_anchor",
    "candidate_layers",
    "choose_candidate",
    "make_ablation_hook",
    "make_addition_hook",
]
