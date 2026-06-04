"""Run a short generation smoke test on prepared refusal datasets (entry)."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make src/ importable so `from lib...` resolves without an editable
# install (e.g. on a fresh Colab runtime, where only PYTHONPATH or an
# install would otherwise expose the package).
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
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

from lib.data import load_records
from lib.runtime import load_model, release_memory

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DATASET_DIR = Path("data/refusal_datasets")


def collect_records(
    dataset_dir: Path, examples_per_class: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label in ["harmful", "benign"]:
        records.extend(load_records(dataset_dir, label, examples_per_class))
    return records


def run_smoke_test(args: argparse.Namespace) -> int:
    dataset_dir = Path(cast(str, args.dataset_dir))
    records = collect_records(dataset_dir, cast(int, args.examples_per_class))
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
        model = load_model(model_name, device)
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
    release_memory(device)
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
