"""Print benign completions under direction addition at several strengths.

A qualitative companion to the quantitative 2x2: the sweep reports only refusal
*rates*, while this shows the raw completions, so the over-steering regime is
legible — moderate strength induces refusal on benign prompts, large strength
degrades the output into incoherence (which is why the keyword rate collapses
back toward zero at high alpha).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

from lib.activations.storage import load_pt
from lib.data.records import records_by_pair_id, select_prompts
from lib.interventions.addition import addition_hooks
from lib.interventions.candidates import choose_candidate
from lib.runtime.devices import release_memory
from lib.runtime.generation import generate_batch, setup_tokenizer
from lib.runtime.models import load_model, model_slug
from lib.runtime.scoring import is_refusal

DEFAULT_ALPHAS = [0.0, 1.0, 4.0, 8.0]


def inspect_for_model(
    model_name: str,
    *,
    device: str,
    activations_dir: Path,
    artifacts_dir: Path,
    dataset_dir: Path,
    position_index: int | None,
    layer: int | None,
    alphas: list[float],
    n_prompts: int,
    eval_tokens: int,
    gen_batch_size: int,
) -> int:
    slug = model_slug(model_name)
    directions_path = activations_dir / slug / "directions.pt"
    if not directions_path.exists():
        print(
            f"Missing {directions_path}. Run extract-directions first.",
            file=sys.stderr,
        )
        return 1
    artifact_subdir = artifacts_dir / slug

    print(f"Loading {model_name} on {device}...", flush=True)
    try:
        model = load_model(model_name, device)
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1
    tokenizer = setup_tokenizer(model)

    blob = load_pt(directions_path, device)
    directions = cast(torch.Tensor, blob["directions"])
    raw_diff_norm = cast(torch.Tensor, blob["raw_diff_norm"])
    offsets = cast(list[int], blob["position_offsets"])
    split_pair_ids = cast(
        dict[str, dict[str, list[int]]], blob["split_pair_ids"]
    )

    # Explicit site avoids needing ablation_best.json; otherwise fall back to
    # the best ablation candidate recorded there.
    if position_index is not None and layer is not None:
        pos_index, chosen_layer = position_index, layer
        pos_offset = int(offsets[pos_index])
    else:
        pos_index, chosen_layer, pos_offset = choose_candidate(
            artifact_subdir, position_index, layer
        )

    benign_ids = split_pair_ids["test"]["benign"]
    benign_records = records_by_pair_id(dataset_dir, "benign")
    prompts = select_prompts(benign_records, benign_ids, n_prompts)

    unit_direction = directions[pos_index, chosen_layer]
    natural_scale = float(raw_diff_norm[pos_index, chosen_layer])
    print(f"\n# MODEL: {model_name}", flush=True)
    print(
        f"Direction site: offset {pos_offset:+d} layer {chosen_layer}; "
        f"natural scale {natural_scale:.3f}. {len(prompts)} benign prompts.",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for alpha in alphas:
        if alpha == 0.0:
            hooks = None
        else:
            vector = alpha * natural_scale * unit_direction
            hooks = addition_hooks(vector, chosen_layer)
        completions = generate_batch(
            model, tokenizer, prompts, eval_tokens, hooks, gen_batch_size
        )
        refusals = sum(is_refusal(t) for t in completions)
        print(
            f"\n===== alpha = {alpha:g}  "
            f"(benign refusal {refusals}/{len(completions)}) =====",
            flush=True,
        )
        for prompt, text in zip(prompts, completions, strict=True):
            ask = prompt.split("user<|end_header_id|>")[-1]
            ask = ask.replace("<|eot_id|>", "").strip()[:70]
            shown = " ".join(text.split())[:160]
            print(f"  • {ask!r}\n      -> {shown!r}", flush=True)
        records.append(
            {
                "alpha": alpha,
                "benign_refusal_rate": round(refusals / len(completions), 4),
                "completions": completions,
            }
        )

    artifact_subdir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_subdir / "addition_examples.json"
    out_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "position_offset": pos_offset,
                "layer": chosen_layer,
                "natural_scale": round(natural_scale, 4),
                "prompts": prompts,
                "by_alpha": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved {out_path}", flush=True)

    del model
    release_memory(device)
    return 0
