"""Prepare an independent, format-matched neutral instruction set.

Builds the generic distribution used as the control for the cross-scale transfer
experiment: instructions disjoint from the harmful (AdvBench) and benign
(Alpaca) prompts, formatted with the same chat template, written as
``generic.jsonl`` alongside the refusal datasets.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make src/ importable so `from lib...` resolves without an editable
# install (e.g. on a fresh Colab runtime, where only PYTHONPATH or an
# install would otherwise expose the package).
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lib.data import (
    DEFAULT_GENERIC_DATASET,
    build_examples,
    load_generic_instructions,
    sample_instructions,
    write_jsonl,
    write_metadata,
)

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT_DIR = Path("data/refusal_datasets")
DEFAULT_LABEL = "generic"


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(cast(str, args.output_dir))
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(cast(str, args.model)),
    )
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"Tokenizer for {args.model!r} has no chat_template; use an "
            "Instruct/chat tokenizer."
        )

    instructions = load_generic_instructions(
        dataset_name=cast(str, args.dataset),
        split=cast(str, args.split),
        column=cast("str | None", args.column),
        context_free=not cast(bool, args.allow_context),
    )
    sample = sample_instructions(
        instructions, cast(int, args.sample_size), cast(int, args.seed)
    )
    examples = build_examples(
        source=cast(str, args.dataset), instructions=sample, tokenizer=tokenizer
    )
    label = cast(str, args.label)
    records: list[dict[str, Any]] = [
        {
            "pair_id": index,
            "label": label,
            "source": example.source,
            "source_index": example.source_index,
            "instruction": example.instruction,
            "raw_token_count": example.raw_token_count,
            "formatted_prompt": example.formatted_prompt,
            "formatted_token_count": example.formatted_token_count,
        }
        for index, example in enumerate(examples)
    ]

    write_jsonl(output_dir / f"{label}.jsonl", records)
    write_metadata(
        output_dir / f"{label}_metadata.json",
        {
            "model": args.model,
            "label": label,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "dataset": args.dataset,
            "split": args.split,
            "column": args.column,
            "context_free": not args.allow_context,
            "purpose": (
                "independent format-matched fitting distribution for the "
                "cross-scale refusal-direction transfer control"
            ),
        },
    )
    print(f"Wrote {len(records)} generic instructions to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--dataset", default=DEFAULT_GENERIC_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--column", default=None)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-context",
        action="store_true",
        help="Keep instructions that carry a separate context field.",
    )
    return parser.parse_args()


def main() -> int:
    prepare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
