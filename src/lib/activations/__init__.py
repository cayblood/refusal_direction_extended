"""Activation capture, divergence diagnostics, and direction extraction."""

from lib.activations.capture import (
    collect_last_k_activations,
    position_offsets,
    resid_post_hook_name,
    verify_last_token,
)
from lib.activations.directions import difference_in_means, raw_diff_norms
from lib.activations.divergence import layerwise_cosine, summarize_best_layer
from lib.activations.splits import split_indices, split_pair_ids
from lib.activations.storage import load_pt, save_activations

__all__ = [
    "collect_last_k_activations",
    "difference_in_means",
    "layerwise_cosine",
    "load_pt",
    "position_offsets",
    "raw_diff_norms",
    "resid_post_hook_name",
    "save_activations",
    "split_indices",
    "split_pair_ids",
    "summarize_best_layer",
    "verify_last_token",
]
