"""Quantitative refusal evaluation: ablation + addition 2x2.

This is the headline result. Using the held-out *test* split (never seen by
direction extraction or the ablation sweep), it measures four refusal
rates per model and assembles the canonical 2x2 table from Arditi et al.:

                       harmful prompts        benign prompts
    baseline           high refusal           ~0 refusal
    intervention       ablation -> low         addition -> high

* Ablation (necessity): project the chosen refusal direction out of the residual
  stream at every layer and confirm refusal on harmful prompts collapses.
* Addition (sufficiency): add ``alpha * r`` at the layer the direction came from
  and confirm benign prompts start getting refused. ``alpha`` is expressed in
  units of the raw difference-in-means norm, so ``alpha=1`` injects one natural
  unit of harmful-vs-benign separation. A short sweep finds the smallest alpha
  that pushes benign refusal past a threshold (avoiding over-steering into
  gibberish, which a keyword classifier would silently undercount).

The chosen ``(position, layer)`` defaults to the best ablation candidate from
``ablation_best.json``. All completions are saved so the keyword refusal rates
can later be re-scored with an LLM-as-judge (``judge_completions.py``) without
re-running the GPU.

The generation harness and keyword refusal classifier are imported from
``evaluate_ablation`` so both stages score refusals identically.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from evaluate_ablation import (
    DEFAULT_ACTIVATIONS_DIR,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_MODELS,
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

DEFAULT_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def make_addition_hook(vector: torch.Tensor):
    """Add a fixed steering ``vector`` to every position of the residual tensor.

    Broadcasts over batch and sequence (including tokens generated under a KV
    cache), so the steering signal persists for the whole completion.
    """

    def hook(tensor: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
        return tensor + vector.to(tensor.dtype)

    return hook


def addition_hooks(vector: torch.Tensor, layer: int) -> list[tuple[str, Any]]:
    """Add ``vector`` at the residual stream of a single source ``layer``."""
    return [(f"blocks.{layer}.hook_resid_post", make_addition_hook(vector))]


def choose_candidate(
    artifact_subdir: Path,
    position_index: int | None,
    layer: int | None,
) -> tuple[int, int, int]:
    """Resolve the (position_index, layer, position_offset) to evaluate.

    Defaults to the best ablation candidate recorded in ``ablation_best.json``;
    explicit ``--position-index``/``--layer`` override it.
    """
    best_path = artifact_subdir / "ablation_best.json"
    if not best_path.exists():
        raise RuntimeError(
            f"Missing {best_path}. Run evaluate_ablation.py first, or pass "
            "--position-index and --layer explicitly."
        )
    best = json.loads(best_path.read_text())["best"]
    pos_index = (
        position_index if position_index is not None else best["position_index"]
    )
    chosen_layer = layer if layer is not None else best["layer"]
    return int(pos_index), int(chosen_layer), int(best["position_offset"])


def evaluate_for_model(
    model_name: str,
    *,
    device: str,
    activations_dir: Path,
    artifacts_dir: Path,
    dataset_dir: Path,
    position_index: int | None,
    layer: int | None,
    alphas: Sequence[float],
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
            f"Missing {directions_path}. Run extract_directions.py first.",
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

    blob = torch.load(directions_path, map_location=device, weights_only=True)
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
    print(
        f"Saved {artifact_subdir}/quantitative_2x2.json",
        flush=True,
    )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="HF model id; repeatable",
    )
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
        "--position-index",
        type=int,
        default=None,
        help="Override the position index (default: best ablation candidate).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override the source layer (default: best ablation candidate).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Addition strengths in units of the raw diff-in-means norm.",
    )
    parser.add_argument(
        "--addition-threshold",
        type=float,
        default=0.5,
        help="Benign refusal rate the chosen alpha must reach.",
    )
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument("--gen-batch-size", type=int, default=64)
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

    models = cast(list[str], args.models) if args.models else DEFAULT_MODELS
    torch.set_grad_enabled(False)

    exit_code = 0
    for model_name in models:
        exit_code = max(
            exit_code,
            evaluate_for_model(
                model_name,
                device=device,
                activations_dir=Path(cast(str, args.activations_dir)),
                artifacts_dir=Path(cast(str, args.artifacts_dir)),
                dataset_dir=Path(cast(str, args.dataset_dir)),
                position_index=cast("int | None", args.position_index),
                layer=cast("int | None", args.layer),
                alphas=cast(list[float], args.alphas),
                addition_threshold=cast(float, args.addition_threshold),
                eval_prompts=cast(int, args.eval_prompts),
                eval_tokens=cast(int, args.eval_tokens),
                example_count=cast(int, args.example_count),
                gen_batch_size=cast(int, args.gen_batch_size),
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
