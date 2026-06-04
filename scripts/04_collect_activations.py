"""Collect residual-stream activations for each model (entry point).

Runs every prepared harmful/benign prompt through the model, captures
``resid_post`` over the last K post-instruction token positions at every layer,
saves the tensor for direction extraction, and writes a per-layer divergence
summary + plot.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make src/ importable so `from lib...` resolves without an editable
# install (e.g. on a fresh Colab runtime, where only PYTHONPATH or an
# install would otherwise expose the package).
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import torch
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)

from lib.activations import (
    collect_last_k_activations,
    layerwise_cosine,
    position_offsets,
    save_activations,
    summarize_best_layer,
    verify_last_token,
)
from lib.data import load_records
from lib.plots import plot_divergence
from lib.runtime import load_model, model_slug, release_memory, resolve_device

DEFAULT_DATASET_DIR = Path("data/refusal_datasets")
DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_NUM_POSITIONS = 5
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def collect_for_model(
    model_name: str,
    *,
    device: str,
    batch_size: int,
    num_positions: int,
    dataset_dir: Path,
    activations_dir: Path,
    artifacts_dir: Path,
    limit: int | None,
) -> int:
    slug = model_slug(model_name)
    print(f"\n# MODEL: {model_name}", flush=True)
    print(f"Loading on {device}...", flush=True)

    try:
        model = load_model(model_name, device)
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(
            "\nCould not load the model from Hugging Face. For gated Llama "
            "models, ensure access and retry after login:\n"
            "  mise hf-login\n  mise download-models\n\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    harmful_records = load_records(dataset_dir, "harmful", limit)
    benign_records = load_records(dataset_dir, "benign", limit)
    print(
        f"Loaded {len(harmful_records)} harmful / {len(benign_records)} "
        f"benign prompts ({model.cfg.n_layers} layers, "
        f"d_model={model.cfg.d_model}).",
        flush=True,
    )

    # Verify the position being read is the post-instruction token.
    position_diag = verify_last_token(
        model, cast(str, harmful_records[0]["formatted_prompt"])
    )
    print("Last-token position check (first harmful prompt):", flush=True)
    print(json.dumps(position_diag, indent=2), flush=True)

    offsets = position_offsets(num_positions)
    start = time.monotonic()
    harmful_acts = collect_last_k_activations(
        model,
        [cast(str, r["formatted_prompt"]) for r in harmful_records],
        device=device,
        batch_size=batch_size,
        num_positions=num_positions,
    )
    benign_acts = collect_last_k_activations(
        model,
        [cast(str, r["formatted_prompt"]) for r in benign_records],
        device=device,
        batch_size=batch_size,
        num_positions=num_positions,
    )
    elapsed = time.monotonic() - start
    print(
        f"Captured activations in {elapsed:.1f}s "
        f"(harmful {tuple(harmful_acts.shape)}, "
        f"benign {tuple(benign_acts.shape)}; positions {offsets}).",
        flush=True,
    )

    # The sanity plot uses the final instruction token (offset 0).
    summary = layerwise_cosine(harmful_acts[:, -1], benign_acts[:, -1])
    best = summarize_best_layer(summary)
    print("Divergence summary (final token position):", flush=True)
    print(json.dumps(best, indent=2), flush=True)

    activations_path = activations_dir / slug / "resid_post.pt"
    save_activations(
        activations_path,
        model_name=model_name,
        harmful=harmful_acts,
        benign=benign_acts,
        offsets=offsets,
        harmful_records=harmful_records,
        benign_records=benign_records,
        position_diag=position_diag,
    )
    print(f"Saved activations: {activations_path}", flush=True)

    artifact_subdir = artifacts_dir / slug
    summary_path = artifact_subdir / "divergence_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "n_prompts_per_class": harmful_acts.shape[0],
                "num_positions": harmful_acts.shape[1],
                "position_offsets": offsets,
                "summary_position_offset": offsets[-1],
                "n_layers": harmful_acts.shape[2],
                "d_model": harmful_acts.shape[3],
                "position_diagnostic": position_diag,
                "best_layer": best,
                "per_layer": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved summary: {summary_path}", flush=True)

    plot_path = artifact_subdir / "divergence.png"
    plot_divergence(summary, model_name, plot_path)
    print(f"Saved plot: {plot_path}", flush=True)

    del model
    release_memory(device)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="HF model id; may be repeated. Defaults to both Llama 3.2 sizes.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-positions",
        type=int,
        default=DEFAULT_NUM_POSITIONS,
        help=(
            "Number of trailing post-instruction token positions to capture "
            "per prompt (Arditi extracts candidate directions over these)."
        ),
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--activations-dir", default=str(DEFAULT_ACTIVATIONS_DIR)
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N prompts per class (for smoke tests).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(
        cast(str, args.device), allow_local=cast(bool, args.allow_local)
    )
    models = cast(list[str], args.models) if args.models else DEFAULT_MODELS
    torch.set_grad_enabled(False)

    exit_code = 0
    for model_name in models:
        exit_code = max(
            exit_code,
            collect_for_model(
                model_name,
                device=device,
                batch_size=cast(int, args.batch_size),
                num_positions=cast(int, args.num_positions),
                dataset_dir=Path(cast(str, args.dataset_dir)),
                activations_dir=Path(cast(str, args.activations_dir)),
                artifacts_dir=Path(cast(str, args.artifacts_dir)),
                limit=cast("int | None", args.limit),
            ),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
