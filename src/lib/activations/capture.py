"""Capture residual-stream activations with forward hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase


def resid_post_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def position_offsets(num_positions: int) -> list[int]:
    """End-relative offsets captured, ordered oldest-first, last token last.

    For ``num_positions=5`` this is ``[-4, -3, -2, -1, 0]`` where ``0`` is the
    final (last instruction) token — the position used in the divergence plot.
    """
    return list(range(-(num_positions - 1), 1))


def collect_last_k_activations(
    model: HookedTransformer,
    prompts: Sequence[str],
    *,
    device: str,
    batch_size: int,
    num_positions: int,
) -> torch.Tensor:
    """Capture resid_post over the last K positions at every layer.

    Returns ``[n_prompts, num_positions, n_layers, d_model]``. The position
    axis is ordered oldest-first (see :func:`position_offsets`), so index ``-1``
    is the final instruction token. Activations are captured with forward hooks,
    gathered at each right-padded sequence's true positions, and moved to CPU
    float32 for stable downstream mean/cosine math.
    """
    tokenizer = cast(PreTrainedTokenizerBase, model.tokenizer)
    n_layers = model.cfg.n_layers
    offsets = torch.tensor(
        position_offsets(num_positions), device=device
    )  # [K]
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(tensor: torch.Tensor, hook: Any) -> None:  # noqa: ARG001
            captured[layer] = tensor.detach()

        return hook

    fwd_hooks = [
        (resid_post_hook_name(layer), make_hook(layer))
        for layer in range(n_layers)
    ]

    collected: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        tokens = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        # True last real-token index for each right-padded sequence.
        last_index = attention_mask.sum(dim=1) - 1
        # Per-sequence positions to gather: [batch, K], clamped into range.
        gather_index = last_index.unsqueeze(1) + offsets.unsqueeze(0)
        gather_index = gather_index.clamp_min(0)

        captured.clear()
        with model.hooks(fwd_hooks=fwd_hooks):
            model(tokens, attention_mask=attention_mask, return_type=None)

        batch_arange = torch.arange(tokens.shape[0], device=device).unsqueeze(1)
        # Each layer: gather [batch, K, d_model], then stack over layers.
        per_layer = [
            captured[layer][batch_arange, gather_index]
            for layer in range(n_layers)
        ]
        batch_acts = torch.stack(per_layer, dim=2)  # [batch, K, n_layers, d]
        collected.append(batch_acts.to("cpu", dtype=torch.float32))

    return torch.cat(collected, dim=0)


def verify_last_token(
    model: HookedTransformer, prompt: str, tail_tokens: int = 6
) -> dict[str, Any]:
    """Decode the final tokens so the chosen position can be eyeballed."""
    tokenizer = cast(PreTrainedTokenizerBase, model.tokenizer)
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tail = ids[-tail_tokens:]
    return {
        "n_tokens": len(ids),
        "last_token_id": ids[-1],
        "last_token_repr": repr(tokenizer.decode([ids[-1]])),
        "tail_decoded": repr(tokenizer.decode(tail)),
    }
