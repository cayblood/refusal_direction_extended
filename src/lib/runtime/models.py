"""Model loading, the same way across the pipeline."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from lib.runtime.devices import torch_dtype_for_device


def model_slug(model_name: str) -> str:
    """Filesystem-safe short name, e.g. 'Llama-3.2-1B-Instruct'."""
    return model_name.split("/")[-1]


def load_model(
    model_name: str,
    device: str,
    *,
    dtype: torch.dtype | None = None,
    processed: bool = False,
) -> HookedTransformer:
    """Load a model the way the rest of the pipeline does.

    Defaults to ``from_pretrained_no_processing`` with half precision on
    accelerators. Pass ``processed=True`` for the full ``from_pretrained``
    path (used by the baseline sanity check at float32).
    """
    resolved_dtype = dtype or torch_dtype_for_device(device)
    loader = (
        HookedTransformer.from_pretrained
        if processed
        else HookedTransformer.from_pretrained_no_processing
    )
    return loader(
        model_name,
        device=device,
        dtype=resolved_dtype,
        default_prepend_bos=False,
    )
