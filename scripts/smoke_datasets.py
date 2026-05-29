"""Run a short generation smoke test on prepared refusal datasets."""

from __future__ import annotations

import argparse
import gc
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
from transformer_lens import HookedTransformer

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")


def load_records(
    dataset_dir: Path, examples_per_class: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in ["harmful", "benign"]:
        path = dataset_dir / f"{label}.jsonl"
        if not path.exists():
            raise RuntimeError(
                f"Missing {path}. Run scripts/prepare_datasets.py first."
            )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        records.extend(rows[:examples_per_class])
    return records


def run_smoke_test(args: argparse.Namespace) -> int:
    dataset_dir = Path(cast(str, args.dataset_dir))
    records = load_records(dataset_dir, cast(int, args.examples_per_class))
    model_name = cast(str, args.model)
    device = cast(str, args.device)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )

    print(
        f"Loading {model_name} on {device} for dataset smoke test...",
        flush=True,
    )
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=torch.float16 if device in {"cuda", "mps"} else torch.float32,
            default_prepend_bos=False,
        )
    except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError) as exc:
        print(
            "Could not load the model from Hugging Face. Ensure your token "
            "has access to the gated Llama repositories and that the model "
            "download/cache cell succeeded.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 1

    torch.set_grad_enabled(False)
    for record in records:
        print("\n" + "=" * 80, flush=True)
        label = cast(str, record["label"]).upper()
        pair_id = record["pair_id"]
        raw_tokens = record["raw_token_count"]
        print(f"{label} pair_id={pair_id} raw_tokens={raw_tokens}", flush=True)
        print(f"Instruction: {record['instruction']}", flush=True)
        start = time.monotonic()
        generated = model.generate(
            cast(str, record["formatted_prompt"]),
            max_new_tokens=cast(int, args.max_new_tokens),
            do_sample=False,
            stop_at_eos=True,
            prepend_bos=False,
            verbose=False,
        )
        prompt_length = len(cast(str, record["formatted_prompt"]))
        completion = cast(str, generated)[prompt_length:].strip()
        print(f"Completed in {time.monotonic() - start:.1f}s", flush=True)
        print("Completion:", flush=True)
        print(completion, flush=True)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "mps", "cpu"]
    )
    parser.add_argument("--examples-per-class", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    try:
        return run_smoke_test(parse_args())
    except RuntimeError as exc:
        print(f"smoke-datasets: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
