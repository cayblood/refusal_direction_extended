"""Validate refusal directions by ablation: sweep + deep evaluation.

For each candidate ``(position, layer)`` difference-in-means direction, project
it out of the residual stream at *every* layer (Arditi et al.'s directional
ablation) and measure whether the model stops refusing held-out harmful
prompts. Candidates are ranked by bypass rate; the best is evaluated more fully,
with example completions saved. Uses the val split only; the test split is
reserved for the quantitative evaluation.
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
from lib.interventions.candidates import candidate_layers
from lib.runtime.devices import release_memory
from lib.runtime.generation import setup_tokenizer
from lib.runtime.models import load_model, model_slug
from lib.runtime.scoring import refusal_rate


def evaluate_for_model(
    model_name: str,
    *,
    device: str,
    activations_dir: Path,
    artifacts_dir: Path,
    dataset_dir: Path,
    layer_step: int,
    sweep_prompts: int,
    sweep_tokens: int,
    eval_prompts: int,
    benign_eval_prompts: int,
    eval_tokens: int,
    top_k: int,
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

    print(f"\n# MODEL: {model_name}", flush=True)
    print(f"Loading on {device}...", flush=True)
    try:
        model = load_model(model_name, device)
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1
    tokenizer = setup_tokenizer(model)

    blob = load_pt(directions_path, device)
    directions = cast(torch.Tensor, blob["directions"])  # [K, L, d]
    offsets = cast(list[int], blob["position_offsets"])
    n_layers = int(blob["n_layers"])
    split_pair_ids = cast(
        dict[str, dict[str, list[int]]], blob["split_pair_ids"]
    )
    val_harmful_ids = split_pair_ids["val"]["harmful"]
    val_benign_ids = split_pair_ids["val"]["benign"]

    harmful_records = records_by_pair_id(dataset_dir, "harmful")
    benign_records = records_by_pair_id(dataset_dir, "benign")
    sweep_set = select_prompts(harmful_records, val_harmful_ids, sweep_prompts)
    print(
        f"Val harmful={len(val_harmful_ids)}, sweep on {len(sweep_set)} "
        f"prompts; positions={offsets}, layers step {layer_step}.",
        flush=True,
    )

    baseline_rate, _ = refusal_rate(
        model, tokenizer, sweep_set, sweep_tokens, None, gen_batch_size
    )
    print(
        f"Baseline refusal rate (no ablation): {baseline_rate:.2f}", flush=True
    )

    layers = candidate_layers(n_layers, layer_step)
    results: list[dict[str, Any]] = []
    start = time.monotonic()
    for pos_index, offset in enumerate(offsets):
        for layer in layers:
            direction = directions[pos_index, layer]
            hooks = ablation_hooks(direction, n_layers)
            rate, _ = refusal_rate(
                model, tokenizer, sweep_set, sweep_tokens, hooks, gen_batch_size
            )
            bypass = 1.0 - rate
            results.append(
                {
                    "position_offset": offset,
                    "position_index": pos_index,
                    "layer": layer,
                    "ablated_refusal_rate": round(rate, 4),
                    "bypass_rate": round(bypass, 4),
                }
            )
    elapsed = time.monotonic() - start
    results.sort(
        key=lambda r: (-r["bypass_rate"], r["layer"], -r["position_offset"])
    )
    print(
        f"Swept {len(results)} candidates in {elapsed:.1f}s. Top 5:",
        flush=True,
    )
    for row in results[:5]:
        print(
            f"  offset={row['position_offset']:+d} layer={row['layer']:2d} "
            f"bypass={row['bypass_rate']:.2f}",
            flush=True,
        )

    artifact_subdir = artifacts_dir / slug
    artifact_subdir.mkdir(parents=True, exist_ok=True)
    (artifact_subdir / "ablation_sweep.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "baseline_refusal_rate": round(baseline_rate, 4),
                "sweep_prompts": len(sweep_set),
                "sweep_tokens": sweep_tokens,
                "layer_step": layer_step,
                "candidates": results,
            },
            indent=2,
        )
        + "\n"
    )

    # Deep-evaluate the top candidates on more prompts + benign coherence.
    eval_harmful = select_prompts(
        harmful_records, val_harmful_ids, eval_prompts
    )
    eval_benign = select_prompts(
        benign_records, val_benign_ids, benign_eval_prompts
    )
    base_harm_rate, base_harm_text = refusal_rate(
        model, tokenizer, eval_harmful, eval_tokens, None, gen_batch_size
    )
    base_benign_rate, _ = refusal_rate(
        model, tokenizer, eval_benign, eval_tokens, None, gen_batch_size
    )
    print(
        f"\nDeep eval baseline: harmful refusal={base_harm_rate:.2f}, "
        f"benign refusal={base_benign_rate:.2f}",
        flush=True,
    )

    deep_results: list[dict[str, Any]] = []
    for row in results[:top_k]:
        direction = directions[row["position_index"], row["layer"]]
        hooks = ablation_hooks(direction, n_layers)
        harm_rate, harm_text = refusal_rate(
            model, tokenizer, eval_harmful, eval_tokens, hooks, gen_batch_size
        )
        benign_rate, _ = refusal_rate(
            model, tokenizer, eval_benign, eval_tokens, hooks, gen_batch_size
        )
        examples = [
            {
                "prompt": eval_harmful[i][-200:],
                "baseline": base_harm_text[i],
                "ablated": harm_text[i],
            }
            for i in range(min(example_count, len(eval_harmful)))
        ]
        deep_results.append(
            {
                "position_offset": row["position_offset"],
                "position_index": row["position_index"],
                "layer": row["layer"],
                "harmful_refusal_rate": round(harm_rate, 4),
                "harmful_bypass_rate": round(1.0 - harm_rate, 4),
                "benign_refusal_rate": round(benign_rate, 4),
                "examples": examples,
            }
        )
        print(
            f"  candidate offset={row['position_offset']:+d} "
            f"layer={row['layer']:2d}: harmful refusal "
            f"{base_harm_rate:.2f} -> {harm_rate:.2f}, benign refusal "
            f"{base_benign_rate:.2f} -> {benign_rate:.2f}",
            flush=True,
        )

    best = deep_results[0] if deep_results else None
    (artifact_subdir / "ablation_best.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "baseline": {
                    "harmful_refusal_rate": round(base_harm_rate, 4),
                    "benign_refusal_rate": round(base_benign_rate, 4),
                    "eval_harmful_prompts": len(eval_harmful),
                    "eval_benign_prompts": len(eval_benign),
                    "eval_tokens": eval_tokens,
                },
                "best": best,
                "top_candidates": deep_results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Saved ablation artifacts to {artifact_subdir}/ablation_*.json",
        flush=True,
    )

    del model
    release_memory(device)
    return 0
