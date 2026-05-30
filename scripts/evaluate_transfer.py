"""Cross-scale refusal-direction transfer via a linear map.

The within-family extension asks: is the refusal geometry *structurally
conserved* across model scale, not merely present in both? Llama-3.2-1B and -3B
have different residual widths (2048 vs 3072), so a refusal direction cannot
transfer literally. Instead we fit a linear alignment between the two activation
spaces and push the 1B refusal direction through it.

Method (Procrustes-style, ridge-regularized):

1. Pair the cached residual activations by prompt (collect_activations stores
   both models' activations over the same prompt order) at each model's own
   *refusal site* — the (position, layer) of its best ablation candidate.
2. Fit a linear map ``W`` (source -> target) on the *train* split, using BOTH
   harmful and benign activations so the map learns a general space alignment
   rather than the refusal axis specifically (otherwise the transfer test would
   be circular). Activations are mean-centered, so a direction (a difference of
   means) maps through ``W`` with the intercept cancelling.
3. Transfer the 1B refusal direction: ``r_transfer = normalize(W @ r_1B)``.
4. Score it three ways:
   * ``cos(r_transfer, r_3B_native)`` — geometric alignment of the transferred
     direction with 3B's own refusal direction (the headline number);
   * held-out reconstruction error of the map (does the alignment generalize?);
   * causal ablation on held-out harmful prompts: baseline vs native-3B vs
     transferred vs a random-direction control.

Runs the CPU fit and the GPU ablation in one pass; expects both models'
``directions.pt`` (with ``ablation_best.json``) and ``resid_post.pt`` present.
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
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

DEFAULT_SOURCE = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TARGET = "meta-llama/Llama-3.2-3B-Instruct"


def load_pt(path: Path, device: str = "cpu") -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. Run earlier pipeline stages first."
        )
    return torch.load(path, map_location=device, weights_only=True)


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


def paired_anchor_matrix(
    activations: dict[str, Any], pos_index: int, layer: int
) -> torch.Tensor:
    """Stack harmful+benign activations at one (position, layer): [2N, d]."""
    harmful = activations["harmful"][:, pos_index, layer, :].float()
    benign = activations["benign"][:, pos_index, layer, :].float()
    return torch.cat([harmful, benign], dim=0)


def class_split_rows(split_indices: dict[str, list[int]], n_per_class: int):
    """Map per-class split indices onto rows of the stacked harmful+benign rows.

    Row ``i`` is harmful prompt ``i``; row ``n_per_class + i`` is benign prompt
    ``i`` (same prompt order across models, verified upstream).
    """
    rows = {}
    for name, idx in split_indices.items():
        harmful_rows = idx
        benign_rows = [n_per_class + i for i in idx]
        rows[name] = torch.tensor(harmful_rows + benign_rows, dtype=torch.long)
    return rows


def fit_linear_map(
    source: torch.Tensor, target: torch.Tensor, ridge_rel: float
) -> torch.Tensor:
    """Ridge-fit ``W_T`` s.t. ``source @ W_T ~= target``: shape [d_src, d_tgt].

    ``ridge_rel`` scales the penalty by the mean feature energy, so it adapts to
    activation magnitude. ``ridge_rel=0`` is ordinary least squares.
    """
    d_src = source.shape[1]
    gram = source.T @ source
    lam = ridge_rel * (torch.trace(gram) / d_src)
    a = gram + lam * torch.eye(d_src, dtype=source.dtype)
    b = source.T @ target
    return torch.linalg.solve(a, b)


def relative_reconstruction_error(
    source: torch.Tensor, target: torch.Tensor, w_t: torch.Tensor
) -> float:
    predicted = source @ w_t
    return float((predicted - target).norm() / target.norm().clamp_min(1e-8))


def random_unit_direction(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(dim, generator=generator)
    return vector / vector.norm().clamp_min(1e-8)


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
    # The refusal directions are difference-in-means over the TRAIN split, so
    # fitting the map on train would be circular: an underdetermined map
    # interpolates its training rows and trivially reproduces the target
    # direction (cos ~ 1). Fit on a disjoint split so transfer is a genuine
    # out-of-sample test, and report reconstruction on yet another split.
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
        f"cos(transferred, native 3B refusal direction) = {cos_align:+.3f}",
        flush=True,
    )

    # --- Causal check on the target model (GPU) ---
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
        "n_eval_harmful_prompts": len(harmful_prompts),
        "eval_tokens": eval_tokens,
    }
    out_dir = artifacts_dir / tgt_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Saved {out_dir}/transfer_summary.json", flush=True)

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
    parser.add_argument(
        "--ridge-rel",
        type=float,
        default=1e-3,
        help="Ridge penalty relative to mean feature energy (0 = OLS).",
    )
    parser.add_argument(
        "--fit-split",
        default="val",
        choices=["train", "val", "test"],
        help="Split used to fit the map; must differ from the 'train' split "
        "that defines the directions to avoid circularity.",
    )
    parser.add_argument(
        "--recon-split",
        default="test",
        choices=["train", "val", "test"],
        help="Split used to measure held-out reconstruction error.",
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
    return run_transfer(
        source_model=cast(str, args.source_model),
        target_model=cast(str, args.target_model),
        device=device,
        activations_dir=Path(cast(str, args.activations_dir)),
        artifacts_dir=Path(cast(str, args.artifacts_dir)),
        dataset_dir=Path(cast(str, args.dataset_dir)),
        ridge_rel=cast(float, args.ridge_rel),
        fit_split=cast(str, args.fit_split),
        recon_split=cast(str, args.recon_split),
        eval_prompts=cast(int, args.eval_prompts),
        eval_tokens=cast(int, args.eval_tokens),
        gen_batch_size=cast(int, args.gen_batch_size),
        random_seed=cast(int, args.random_seed),
    )


if __name__ == "__main__":
    raise SystemExit(main())
