"""Prepare a neutral, format-matched instruction set (transfer control).

The cross-scale transfer map must be fit on prompts DISJOINT from the harmful
(AdvBench) and benign (Alpaca) prompts that define the refusal direction;
otherwise "does refusal transfer?" is circular. This builds an independent,
format-matched neutral-instruction set (default: databricks-dolly-15k,
context-free instructions) formatted with the exact same Llama chat template as
the refusal datasets, written as ``generic.jsonl`` alongside them.

It is "format-matched" on purpose: the prompts are instruction-style and pass
through the same chat template, so the refusal *site* (the last instruction
token before the assistant reply) is comparable to where the refusal direction
was extracted — while still coming from a different prompt distribution.

Reuses the formatting helpers from ``prepare_datasets`` so the generic prompts
are tokenized and templated identically to harmful/benign.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from prepare_datasets import (
    build_examples,
    choose_text_column,
    compact_unique,
    write_jsonl,
    write_metadata,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT_DIR = Path("data/refusal_datasets")
DEFAULT_GENERIC_DATASET = "databricks/databricks-dolly-15k"
DEFAULT_LABEL = "generic"


def load_generic_instructions(
    dataset_name: str,
    split: str,
    column: str | None,
    context_free: bool,
) -> list[str]:
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))
    if context_free and "context" in dataset.column_names:
        dataset = dataset.filter(
            lambda row: not str(row.get("context") or "").strip()
        )
    text_column = choose_text_column(dataset.column_names, column)
    return compact_unique(dataset[text_column])


def sample_instructions(
    instructions: list[str], sample_size: int, seed: int
) -> list[str]:
    if len(instructions) < sample_size:
        raise RuntimeError(
            f"Need {sample_size} generic instructions, found "
            f"{len(instructions)}"
        )
    return random.Random(seed).sample(instructions, sample_size)


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
        cast(str, args.dataset),
        cast(str, args.split),
        cast("str | None", args.column),
        context_free=not cast(bool, args.allow_context),
    )
    sample = sample_instructions(
        instructions, cast(int, args.sample_size), cast(int, args.seed)
    )
    examples = build_examples(
        source=cast(str, args.dataset), instructions=sample, tokenizer=tokenizer
    )
    records: list[dict[str, Any]] = [
        {
            "pair_id": index,
            "label": cast(str, args.label),
            "source": example.source,
            "source_index": example.source_index,
            "instruction": example.instruction,
            "raw_token_count": example.raw_token_count,
            "formatted_prompt": example.formatted_prompt,
            "formatted_token_count": example.formatted_token_count,
        }
        for index, example in enumerate(examples)
    ]

    label = cast(str, args.label)
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
