"""Cross-scale refusal-direction transfer via a fitted linear map.

``run_transfer`` fits the map on a split of the harmful/benign activations (same
distribution as the directions). ``run_independent_transfer`` refits it on an
independent generic distribution as a control: if the apparent transfer
survives, it is genuine; if it collapses, it was an artifact of the fitting
distribution. Both push the source refusal direction through the map and test it
on the target model by ablation, against native and random-direction baselines.
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
from lib.interventions.ablation import ablation_hooks
from lib.interventions.candidates import best_anchor
from lib.runtime.devices import release_memory
from lib.runtime.generation import setup_tokenizer
from lib.runtime.models import load_model, model_slug
from lib.runtime.scoring import refusal_rate
from lib.transfer.alignment import (
    fit_linear_map,
    relative_reconstruction_error,
)
from lib.transfer.vectors import (
    class_split_rows,
    generic_anchor_matrix,
    paired_anchor_matrix,
    random_unit_direction,
)


def _ablation_variants(
    *,
    target_model: str,
    device: str,
    tgt_dir: dict[str, Any],
    dataset_dir: Path,
    r_native: torch.Tensor,
    r_transfer: torch.Tensor,
    eval_prompts: int,
    eval_tokens: int,
    gen_batch_size: int,
    random_seed: int,
) -> tuple[dict[str, float], int] | None:
    """Load the target model and ablate native/transferred/random directions.

    Returns (refusal rates per variant, n_eval_prompts), or None on load error.
    """
    print(f"Loading {target_model} on {device}...", flush=True)
    try:
        model = load_model(target_model, device)
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return None
    tokenizer = setup_tokenizer(model)
    n_layers = int(tgt_dir["n_layers"])
    d_model = int(tgt_dir["d_model"])

    harmful_records = records_by_pair_id(dataset_dir, "harmful")
    test_harmful_ids = cast(
        dict[str, dict[str, list[int]]], tgt_dir["split_pair_ids"]
    )["test"]["harmful"]
    harmful_prompts = select_prompts(
        harmful_records, test_harmful_ids, eval_prompts
    )
    print(f"Ablation eval on {len(harmful_prompts)} held-out harmful prompts.")

    r_random = random_unit_direction(d_model, random_seed)
    variants = {
        "baseline": None,
        "native_3b": r_native,
        "transferred_1b": r_transfer,
        "random_control": r_random,
    }
    results: dict[str, float] = {}
    for name, direction in variants.items():
        hooks = (
            None
            if direction is None
            else ablation_hooks(direction.to(device), n_layers)
        )
        rate, _ = refusal_rate(
            model,
            tokenizer,
            harmful_prompts,
            eval_tokens,
            hooks,
            gen_batch_size,
        )
        results[name] = round(rate, 4)
        print(f"  {name:16s} harmful refusal={rate:.3f}", flush=True)

    del model
    release_memory(device)
    return results, len(harmful_prompts)


def run_transfer(
    *,
    source_model: str,
    target_model: str,
    device: str,
    activations_dir: Path,
    artifacts_dir: Path,
    dataset_dir: Path,
    ridge_rel: float,
    fit_split: str,
    recon_split: str,
    eval_prompts: int,
    eval_tokens: int,
    gen_batch_size: int,
    random_seed: int,
) -> int:
    src_slug, tgt_slug = model_slug(source_model), model_slug(target_model)
    src_dir = load_pt(activations_dir / src_slug / "directions.pt")
    tgt_dir = load_pt(activations_dir / tgt_slug / "directions.pt")
    src_acts = load_pt(activations_dir / src_slug / "resid_post.pt")
    tgt_acts = load_pt(activations_dir / tgt_slug / "resid_post.pt")

    src_pos, src_layer, src_offset = best_anchor(artifacts_dir / src_slug)
    tgt_pos, tgt_layer, tgt_offset = best_anchor(artifacts_dir / tgt_slug)
    n_per_class = src_acts["harmful"].shape[0]

    print(f"\n# TRANSFER: {source_model} -> {target_model}", flush=True)
    print(
        f"Source refusal site: offset {src_offset:+d}, layer {src_layer} "
        f"({src_layer}/{int(src_dir['n_layers'])} = "
        f"{src_layer / int(src_dir['n_layers']):.2f} depth).",
        flush=True,
    )
    print(
        f"Target refusal site: offset {tgt_offset:+d}, layer {tgt_layer} "
        f"({tgt_layer}/{int(tgt_dir['n_layers'])} = "
        f"{tgt_layer / int(tgt_dir['n_layers']):.2f} depth).",
        flush=True,
    )

    x_src = paired_anchor_matrix(src_acts, src_pos, src_layer)
    x_tgt = paired_anchor_matrix(tgt_acts, tgt_pos, tgt_layer)
    rows = class_split_rows(
        cast(dict[str, list[int]], tgt_dir["split_indices"]), n_per_class
    )
    # Fitting on 'train' is circular with the direction definition; fit on a
    # disjoint split so transfer is a genuine out-of-sample test.
    if fit_split == "train":
        print(
            "WARNING: fitting the map on 'train' is circular with the "
            "direction definition; cos will be trivially ~1.",
            file=sys.stderr,
        )
    fit_rows, recon_rows = rows[fit_split], rows[recon_split]
    print(
        f"Fitting map on '{fit_split}' ({fit_rows.numel()} rows); "
        f"reconstruction on '{recon_split}' ({recon_rows.numel()} rows).",
        flush=True,
    )

    mu_src = x_src[fit_rows].mean(0)
    mu_tgt = x_tgt[fit_rows].mean(0)
    xc_src, xc_tgt = x_src - mu_src, x_tgt - mu_tgt

    w_t = fit_linear_map(xc_src[fit_rows], xc_tgt[fit_rows], ridge_rel)
    recon_err = relative_reconstruction_error(
        xc_src[recon_rows], xc_tgt[recon_rows], w_t
    )
    print(
        f"Held-out relative reconstruction error: {recon_err:.3f}", flush=True
    )

    r_src = src_dir["directions"][src_pos, src_layer].float()
    r_tgt_native = tgt_dir["directions"][tgt_pos, tgt_layer].float()
    r_transfer = r_src @ w_t
    r_transfer = r_transfer / r_transfer.norm().clamp_min(1e-8)
    cos_align = float(torch.dot(r_transfer, r_tgt_native))
    print(
        f"cos(transferred, native target refusal direction) = {cos_align:+.3f}",
        flush=True,
    )

    outcome = _ablation_variants(
        target_model=target_model,
        device=device,
        tgt_dir=tgt_dir,
        dataset_dir=dataset_dir,
        r_native=r_tgt_native,
        r_transfer=r_transfer,
        eval_prompts=eval_prompts,
        eval_tokens=eval_tokens,
        gen_batch_size=gen_batch_size,
        random_seed=random_seed,
    )
    if outcome is None:
        return 1
    ablation_results, n_eval = outcome

    summary = {
        "source_model": source_model,
        "target_model": target_model,
        "source_site": {
            "position_offset": src_offset,
            "layer": src_layer,
            "relative_depth": round(src_layer / int(src_dir["n_layers"]), 4),
        },
        "target_site": {
            "position_offset": tgt_offset,
            "layer": tgt_layer,
            "relative_depth": round(tgt_layer / int(tgt_dir["n_layers"]), 4),
        },
        "ridge_rel": ridge_rel,
        "fit_split": fit_split,
        "recon_split": recon_split,
        "n_fit_rows": int(fit_rows.numel()),
        "held_out_reconstruction_error": round(recon_err, 4),
        "cos_transfer_vs_native": round(cos_align, 4),
        "ablation_harmful_refusal_rate": ablation_results,
        "n_eval_harmful_prompts": n_eval,
        "eval_tokens": eval_tokens,
    }
    out_dir = artifacts_dir / tgt_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {out_dir}/transfer_summary.json", flush=True)
    return 0


def run_independent_transfer(
    *,
    source_model: str,
    target_model: str,
    device: str,
    activations_dir: Path,
    artifacts_dir: Path,
    dataset_dir: Path,
    ridge_rel: float,
    fit_frac: float,
    eval_prompts: int,
    eval_tokens: int,
    gen_batch_size: int,
    random_seed: int,
) -> int:
    src_slug, tgt_slug = model_slug(source_model), model_slug(target_model)
    src_dir = load_pt(activations_dir / src_slug / "directions.pt")
    tgt_dir = load_pt(activations_dir / tgt_slug / "directions.pt")
    src_gen = load_pt(activations_dir / src_slug / "resid_post_generic.pt")
    tgt_gen = load_pt(activations_dir / tgt_slug / "resid_post_generic.pt")
    if src_gen["generic_pair_ids"] != tgt_gen["generic_pair_ids"]:
        raise RuntimeError(
            "Generic activations are not prompt-aligned across models; "
            "re-run collect-generic-activations for both."
        )

    src_pos, src_layer, src_offset = best_anchor(artifacts_dir / src_slug)
    tgt_pos, tgt_layer, tgt_offset = best_anchor(artifacts_dir / tgt_slug)

    print(f"\n# INDEPENDENT TRANSFER: {source_model} -> {target_model}")
    print(
        f"Map fit on '{src_gen.get('label', 'generic')}' "
        f"({src_gen['generic'].shape[0]} prompts), a distribution disjoint "
        "from the harmful/benign prompts that define the direction.",
        flush=True,
    )

    x_src = generic_anchor_matrix(src_gen, src_pos, src_layer)
    x_tgt = generic_anchor_matrix(tgt_gen, tgt_pos, tgt_layer)
    n_generic = x_src.shape[0]
    n_fit = int(fit_frac * n_generic)
    if not 0 < n_fit < n_generic:
        raise RuntimeError(
            f"fit_frac={fit_frac} gives {n_fit} fit rows of {n_generic}."
        )
    fit_rows = torch.arange(n_fit)
    recon_rows = torch.arange(n_fit, n_generic)
    print(
        f"Generic split: {n_fit} fit / {recon_rows.numel()} reconstruction "
        "rows.",
        flush=True,
    )

    mu_src = x_src[fit_rows].mean(0)
    mu_tgt = x_tgt[fit_rows].mean(0)
    xc_src, xc_tgt = x_src - mu_src, x_tgt - mu_tgt

    w_t = fit_linear_map(xc_src[fit_rows], xc_tgt[fit_rows], ridge_rel)
    recon_err = relative_reconstruction_error(
        xc_src[recon_rows], xc_tgt[recon_rows], w_t
    )
    print(f"Held-out generic reconstruction error: {recon_err:.3f}", flush=True)

    r_src = src_dir["directions"][src_pos, src_layer].float()
    r_tgt_native = tgt_dir["directions"][tgt_pos, tgt_layer].float()
    r_transfer = r_src @ w_t
    r_transfer = r_transfer / r_transfer.norm().clamp_min(1e-8)
    cos_align = float(torch.dot(r_transfer, r_tgt_native))
    print(
        f"cos(transferred, native target refusal direction) = {cos_align:+.3f}",
        flush=True,
    )

    outcome = _ablation_variants(
        target_model=target_model,
        device=device,
        tgt_dir=tgt_dir,
        dataset_dir=dataset_dir,
        r_native=r_tgt_native,
        r_transfer=r_transfer,
        eval_prompts=eval_prompts,
        eval_tokens=eval_tokens,
        gen_batch_size=gen_batch_size,
        random_seed=random_seed,
    )
    if outcome is None:
        return 1
    ablation_results, n_eval = outcome

    summary = {
        "source_model": source_model,
        "target_model": target_model,
        "fit_distribution": "generic_independent",
        "fit_dataset_label": src_gen.get("label", "generic"),
        "source_site": {"position_offset": src_offset, "layer": src_layer},
        "target_site": {"position_offset": tgt_offset, "layer": tgt_layer},
        "ridge_rel": ridge_rel,
        "n_fit_rows": int(fit_rows.numel()),
        "n_recon_rows": int(recon_rows.numel()),
        "held_out_reconstruction_error": round(recon_err, 4),
        "cos_transfer_vs_native": round(cos_align, 4),
        "ablation_harmful_refusal_rate": ablation_results,
        "n_eval_harmful_prompts": n_eval,
        "eval_tokens": eval_tokens,
    }
    out_dir = artifacts_dir / tgt_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_independent_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {out_dir}/transfer_independent_summary.json", flush=True)
    return 0
