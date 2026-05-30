"""Collect residual-stream activations for refusal-direction extraction.

For each model it runs every prepared harmful/benign prompt through the network
and captures the residual stream (`resid_post`) at the *last instruction-token
position* — the token right before the assistant reply begins — at every layer.
The result is a tensor of shape ``[n_prompts, n_layers, d_model]`` saved to
disk so direction extraction does not need to recompute it.

It also writes a sanity summary: the cosine similarity between the mean harmful
and mean benign activation at each layer (which should dip in the middle
layers, where refusal is mediated) and a matching plot.

Methodology notes:

* The dataset ``formatted_prompt`` already contains the chat template and a
  leading ``<|begin_of_text|>``. We therefore tokenize with
  ``add_special_tokens=False`` and never prepend another BOS, matching how
  ``baseline.py`` and ``smoke_datasets.py`` load the model.
* Prompts in a batch are *right*-padded and each sequence's activation is
  gathered at its own true last index. Right padding keeps rotary position
  indices correct for the real tokens; left padding would shift them.
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
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
)
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")
DEFAULT_ACTIVATIONS_DIR = Path("data/activations")
DEFAULT_ARTIFACTS_DIR = Path("artifacts/activations")
DEFAULT_NUM_POSITIONS = 5
LABELS = ("harmful", "benign")


def model_slug(model_name: str) -> str:
    """Filesystem-safe short name, e.g. 'Llama-3.2-1B-Instruct'."""
    return model_name.split("/")[-1]


def pick_device(require_cuda: bool) -> str:
    """Choose the runtime device for model execution."""
    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError(
            "CUDA is required for this task. Use a GPU runtime or pass "
            "--allow-local for an explicitly local CPU/MPS run."
        )
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype_for_device(device: str) -> torch.dtype:
    """Lower precision on accelerators, float32 on CPU for compatibility."""
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def load_records(
    dataset_dir: Path, label: str, limit: int | None
) -> list[dict[str, Any]]:
    """Load prepared prompts for one class, optionally truncated."""
    path = dataset_dir / f"{label}.jsonl"
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. Run scripts/prepare_datasets.py first."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_model(model_name: str, device: str) -> HookedTransformer:
    """Load a model the same way the rest of the pipeline does."""
    return HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        dtype=torch_dtype_for_device(device),
        default_prepend_bos=False,
    )


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


def layerwise_cosine(
    harmful: torch.Tensor, benign: torch.Tensor
) -> dict[str, list[float]]:
    """Per-layer divergence stats between mean harmful and benign activations.

    A low cosine similarity (and a large normalized difference) at a layer
    means harmful and benign prompts are linearly separable there — the
    signature of a refusal-mediating layer.
    """
    mean_harmful = harmful.mean(dim=0)  # [n_layers, d_model]
    mean_benign = benign.mean(dim=0)
    diff = mean_harmful - mean_benign

    cosine = torch.nn.functional.cosine_similarity(
        mean_harmful, mean_benign, dim=1
    )
    # Difference norm relative to the typical activation norm at that layer.
    mean_norm = 0.5 * (mean_harmful.norm(dim=1) + mean_benign.norm(dim=1))
    relative_diff = diff.norm(dim=1) / mean_norm.clamp_min(1e-6)

    return {
        "cosine_similarity": cosine.tolist(),
        "diff_norm": diff.norm(dim=1).tolist(),
        "relative_diff_norm": relative_diff.tolist(),
    }


def plot_divergence(
    summary: dict[str, list[float]], model_name: str, output_path: Path
) -> None:
    """Plot cosine similarity and relative difference norm across layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cosine = summary["cosine_similarity"]
    relative = summary["relative_diff_norm"]
    layers = list(range(len(cosine)))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    top.plot(layers, cosine, marker="o", color="#c0392b")
    top.set_ylabel("cosine(mean_harmful, mean_benign)")
    top.set_title(f"Harmful vs. benign activation divergence\n{model_name}")
    top.grid(True, alpha=0.3)

    bottom.plot(layers, relative, marker="o", color="#2c3e50")
    bottom.set_ylabel("||Δ mean|| / mean ||activation||")
    bottom.set_xlabel("layer (resid_post)")
    bottom.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def save_activations(
    path: Path,
    *,
    model_name: str,
    harmful: torch.Tensor,
    benign: torch.Tensor,
    offsets: Sequence[int],
    harmful_records: Sequence[dict[str, Any]],
    benign_records: Sequence[dict[str, Any]],
    position_diag: dict[str, Any],
) -> None:
    """Persist activations and provenance for the direction-extraction step.

    Tensors have shape ``[n_prompts, num_positions, n_layers, d_model]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "hook": "resid_post",
            "position": "last_instruction_tokens",
            "position_offsets": list(offsets),
            "num_positions": harmful.shape[1],
            "n_layers": harmful.shape[2],
            "d_model": harmful.shape[3],
            "harmful": harmful,
            "benign": benign,
            "harmful_pair_ids": [r["pair_id"] for r in harmful_records],
            "benign_pair_ids": [r["pair_id"] for r in benign_records],
            "position_diagnostic": position_diag,
        },
        path,
    )


def summarize_best_layer(summary: dict[str, list[float]]) -> dict[str, Any]:
    """Identify the layer where harmful/benign separation peaks."""
    cosine = summary["cosine_similarity"]
    relative = summary["relative_diff_norm"]
    min_cosine_layer = min(range(len(cosine)), key=lambda i: cosine[i])
    max_diff_layer = max(range(len(relative)), key=lambda i: relative[i])
    return {
        "min_cosine_layer": min_cosine_layer,
        "min_cosine_value": cosine[min_cosine_layer],
        "max_relative_diff_layer": max_diff_layer,
        "max_relative_diff_value": relative[max_diff_layer],
    }


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
        help="HF model id; may be repeated. Defaults to both Llama 3.2 sizes.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--allow-local",
        action="store_true",
        help="Allow CPU/MPS execution when CUDA is unavailable.",
    )
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
