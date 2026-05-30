"""Cross-scale transfer fit on an INDEPENDENT distribution (control).

BONUS step — additive, modifies nothing. Strengthens the transfer claim from
``evaluate_transfer.py``. There, the alignment map is fit on the held-out *val*
split of the same harmful/benign prompts that define the refusal direction —
disjoint prompts, but the same distribution. A skeptic can still argue the map
saw refusal-relevant activations. This control fits the map on a fully
independent, format-matched neutral-instruction set (``resid_post_generic.pt``
from ``collect_generic_activations.py``) that contains no harmful/benign prompts
at all, then asks the same question: does the 1B refusal direction, pushed
through that map, still act as 3B's refusal direction?

Directions remain the train-split difference-in-means (unchanged); only the
map's fitting data changes. The generic set is split into a fit and a held-out
reconstruction portion. Reuses the map-fitting, scoring, and ablation helpers
from ``evaluate_transfer`` / ``evaluate_ablation``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
from evaluate_ablation import (
    DEFAULT_ACTIVATIONS_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_DIR,
    ablation_hooks,
    load_model,
    model_slug,
    pick_device,
    records_by_pair_id,
    refusal_rate,
    select_prompts,
    setup_tokenizer,
)
from evaluate_transfer import (
    best_anchor,
    fit_linear_map,
    load_pt,
    random_unit_direction,
    relative_reconstruction_error,
)
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

DEFAULT_SOURCE = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TARGET = "meta-llama/Llama-3.2-3B-Instruct"


def generic_anchor_matrix(
    generic: dict[str, Any], pos_index: int, layer: int
) -> torch.Tensor:
    """Generic activations at one (position, layer): [n_generic, d]."""
    return generic["generic"][:, pos_index, layer, :].float()


def run(
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
            "re-run collect_generic_activations.py for both."
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
        f"cos(transferred, native 3B refusal direction) = {cos_align:+.3f}",
        flush=True,
    )

    print(f"Loading {target_model} on {device}...", flush=True)
    try:
        model = load_model(target_model, device)
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1
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
        "native_3b": r_tgt_native,
        "transferred_1b": r_transfer,
        "random_control": r_random,
    }
    ablation_results: dict[str, float] = {}
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
        ablation_results[name] = round(rate, 4)
        print(f"  {name:16s} harmful refusal={rate:.3f}", flush=True)

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
        "n_eval_harmful_prompts": len(harmful_prompts),
        "eval_tokens": eval_tokens,
    }
    out_dir = artifacts_dir / tgt_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_independent_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {out_dir}/transfer_independent_summary.json", flush=True)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE)
    parser.add_argument("--target-model", default=DEFAULT_TARGET)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument(
        "--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR)
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--ridge-rel", type=float, default=1e-3)
    parser.add_argument(
        "--fit-frac",
        type=float,
        default=0.8,
        help="Fraction of generic prompts used to fit the map; the rest "
        "measure held-out reconstruction error.",
    )
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_cuda = not cast(bool, args.allow_local)
    device = (
        pick_device(require_cuda=require_cuda)
        if args.device == "auto"
        else cast(str, args.device)
    )
    if require_cuda and device != "cuda":
        print(
            "CUDA is required by default. Use --allow-local to run on "
            f"{device!r} intentionally.",
            file=sys.stderr,
        )
        return 2

    torch.set_grad_enabled(False)
    return run(
        source_model=cast(str, args.source_model),
        target_model=cast(str, args.target_model),
        device=device,
        activations_dir=Path(cast(str, args.activations_dir)),
        artifacts_dir=Path(cast(str, args.artifacts_dir)),
        dataset_dir=Path(cast(str, args.dataset_dir)),
        ridge_rel=cast(float, args.ridge_rel),
        fit_frac=cast(float, args.fit_frac),
        eval_prompts=cast(int, args.eval_prompts),
        eval_tokens=cast(int, args.eval_tokens),
        gen_batch_size=cast(int, args.gen_batch_size),
        random_seed=cast(int, args.random_seed),
    )


if __name__ == "__main__":
    raise SystemExit(main())
