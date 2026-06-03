"""Quantitative ablation + addition 2x2 on the held-out test split.

Measures four refusal rates: harmful baseline vs ablated (necessity), and benign
baseline vs the direction added (sufficiency). ``alpha`` is in units of the raw
difference-in-means norm; a short sweep picks the smallest alpha that pushes
benign refusal past a threshold. All completions sampled are saved so the
keyword rates can later be re-scored with an LLM-as-judge.
"""

from __future__ import annotations

import json
import sys
import time
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
from lib.interventions.ablation import ablation_hooks
from lib.interventions.addition import addition_hooks
from lib.interventions.candidates import choose_candidate
from lib.runtime.devices import release_memory
from lib.runtime.generation import setup_tokenizer
from lib.runtime.models import load_model, model_slug
from lib.runtime.scoring import refusal_rate

DEFAULT_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def evaluate_for_model(
    model_name: str,
    *,
    device: str,
    activations_dir: Path,
    artifacts_dir: Path,
    dataset_dir: Path,
    position_index: int | None,
    layer: int | None,
    alphas: list[float],
    addition_threshold: float,
    eval_prompts: int,
    eval_tokens: int,
    example_count: int,
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
    pos_index, chosen_layer, pos_offset = choose_candidate(
        artifact_subdir, position_index, layer
    )

    print(f"\n# MODEL: {model_name}", flush=True)
    print(
        f"Chosen direction: position_offset={pos_offset:+d} "
        f"(index {pos_index}), layer {chosen_layer}.",
        flush=True,
    )
    print(f"Loading on {device}...", flush=True)
    try:
        model = load_model(model_name, device)
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1
    tokenizer = setup_tokenizer(model)

    blob = load_pt(directions_path, device)
    directions = cast(torch.Tensor, blob["directions"])  # [K, L, d] unit norm
    raw_diff_norm = cast(torch.Tensor, blob["raw_diff_norm"])  # [K, L]
    n_layers = int(blob["n_layers"])
    split_pair_ids = cast(
        dict[str, dict[str, list[int]]], blob["split_pair_ids"]
    )
    test_harmful_ids = split_pair_ids["test"]["harmful"]
    test_benign_ids = split_pair_ids["test"]["benign"]

    harmful_records = records_by_pair_id(dataset_dir, "harmful")
    benign_records = records_by_pair_id(dataset_dir, "benign")
    harmful_prompts = select_prompts(
        harmful_records, test_harmful_ids, eval_prompts
    )
    benign_prompts = select_prompts(
        benign_records, test_benign_ids, eval_prompts
    )
    print(
        f"Test split: {len(harmful_prompts)} harmful, "
        f"{len(benign_prompts)} benign prompts.",
        flush=True,
    )

    unit_direction = directions[pos_index, chosen_layer]
    natural_scale = float(raw_diff_norm[pos_index, chosen_layer])
    ablate = ablation_hooks(unit_direction, n_layers)

    start = time.monotonic()

    # --- Baselines (no intervention) ---
    base_harm_rate, base_harm_text = refusal_rate(
        model, tokenizer, harmful_prompts, eval_tokens, None, gen_batch_size
    )
    base_benign_rate, base_benign_text = refusal_rate(
        model, tokenizer, benign_prompts, eval_tokens, None, gen_batch_size
    )

    # --- Ablation on harmful prompts (necessity) ---
    ablated_harm_rate, ablated_harm_text = refusal_rate(
        model, tokenizer, harmful_prompts, eval_tokens, ablate, gen_batch_size
    )

    # --- Addition alpha-sweep on benign prompts (sufficiency) ---
    print(
        f"Addition alpha-sweep on benign prompts (natural scale "
        f"={natural_scale:.3f}):",
        flush=True,
    )
    sweep: list[dict[str, Any]] = []
    added_text_by_alpha: dict[float, list[str]] = {}
    for alpha in alphas:
        if alpha == 0.0:
            rate, text = base_benign_rate, base_benign_text
        else:
            vector = alpha * natural_scale * unit_direction
            hooks = addition_hooks(vector, chosen_layer)
            rate, text = refusal_rate(
                model,
                tokenizer,
                benign_prompts,
                eval_tokens,
                hooks,
                gen_batch_size,
            )
        added_text_by_alpha[alpha] = text
        sweep.append({"alpha": alpha, "benign_refusal_rate": round(rate, 4)})
        print(f"  alpha={alpha:>4}: benign refusal={rate:.3f}", flush=True)

    # Smallest alpha clearing the threshold; else the alpha with max refusal.
    crossing = [
        row for row in sweep if row["benign_refusal_rate"] >= addition_threshold
    ]
    chosen_row = (
        min(crossing, key=lambda r: r["alpha"])
        if crossing
        else max(sweep, key=lambda r: r["benign_refusal_rate"])
    )
    chosen_alpha = float(chosen_row["alpha"])
    added_benign_rate = float(chosen_row["benign_refusal_rate"])
    added_benign_text = added_text_by_alpha[chosen_alpha]
    elapsed = time.monotonic() - start

    table = {
        "baseline": {
            "harmful_refusal_rate": round(base_harm_rate, 4),
            "benign_refusal_rate": round(base_benign_rate, 4),
        },
        "intervention": {
            "harmful_refusal_rate_ablated": round(ablated_harm_rate, 4),
            "benign_refusal_rate_added": round(added_benign_rate, 4),
        },
    }
    print(
        "\n2x2 (refusal rates):\n"
        f"                 harmful    benign\n"
        f"  baseline       {base_harm_rate:6.3f}    {base_benign_rate:6.3f}\n"
        f"  intervention   {ablated_harm_rate:6.3f}    {added_benign_rate:6.3f}"
        f"   (ablation / addition @ alpha={chosen_alpha:g})",
        flush=True,
    )
    print(f"Evaluated in {elapsed:.1f}s.", flush=True)

    harmful_examples = [
        {
            "prompt": harmful_prompts[i][-200:],
            "baseline": base_harm_text[i],
            "ablated": ablated_harm_text[i],
        }
        for i in range(min(example_count, len(harmful_prompts)))
    ]
    benign_examples = [
        {
            "prompt": benign_prompts[i][-200:],
            "baseline": base_benign_text[i],
            "added": added_benign_text[i],
        }
        for i in range(min(example_count, len(benign_prompts)))
    ]

    artifact_subdir.mkdir(parents=True, exist_ok=True)
    (artifact_subdir / "quantitative_2x2.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "classifier": "keyword",
                "direction": {
                    "position_offset": pos_offset,
                    "position_index": pos_index,
                    "layer": chosen_layer,
                    "natural_scale": round(natural_scale, 4),
                },
                "n_test_prompts": {
                    "harmful": len(harmful_prompts),
                    "benign": len(benign_prompts),
                },
                "eval_tokens": eval_tokens,
                "table": table,
                "addition_sweep": sweep,
                "addition_threshold": addition_threshold,
                "chosen_alpha": chosen_alpha,
                "examples": {
                    "harmful": harmful_examples,
                    "benign": benign_examples,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {artifact_subdir}/quantitative_2x2.json", flush=True)

    del model
    release_memory(device)
    return 0
