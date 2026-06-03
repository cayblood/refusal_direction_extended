"""Prepare harmful/benign instruction datasets (entry point).

Samples harmful instructions from AdvBench, length-matches benign instructions
from Alpaca, and writes JSONL files with raw + Llama-chat-formatted prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lib.data import (
    DEFAULT_ALPACA_DATASET,
    DEFAULT_HARMFUL_DATASET,
    build_examples,
    example_record,
    length_match_pairs,
    load_benign_instructions,
    load_harmful_instructions,
    sample_harmful,
    summarize_deltas,
    write_jsonl,
    write_metadata,
)

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT_DIR = Path("data/refusal_datasets")


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

    harmful_instructions = load_harmful_instructions(
        harmful_dataset=cast(str, args.harmful_dataset),
        harmful_split=cast(str, args.harmful_split),
        harmful_column=cast("str | None", args.harmful_column),
        advbench_csv_url=cast("str | None", args.advbench_csv_url),
    )
    benign_instructions = load_benign_instructions(
        benign_dataset=cast(str, args.benign_dataset),
        benign_split=cast(str, args.benign_split),
        benign_column=cast("str | None", args.benign_column),
    )
    harmful_sample = sample_harmful(
        harmful_instructions, cast(int, args.sample_size), cast(int, args.seed)
    )

    harmful_examples = build_examples(
        source=cast(str, args.harmful_dataset),
        instructions=harmful_sample,
        tokenizer=tokenizer,
    )
    benign_examples = build_examples(
        source=cast(str, args.benign_dataset),
        instructions=benign_instructions,
        tokenizer=tokenizer,
    )
    pairs = length_match_pairs(harmful_examples, benign_examples)

    harmful_records = [
        example_record(
            pair.harmful, "harmful", pair_id, pair.token_length_delta
        )
        for pair_id, pair in enumerate(pairs)
    ]
    benign_records = [
        example_record(pair.benign, "benign", pair_id, pair.token_length_delta)
        for pair_id, pair in enumerate(pairs)
    ]
    combined_records: list[dict[str, Any]] = [
        record
        for pair in zip(harmful_records, benign_records, strict=True)
        for record in pair
    ]

    write_jsonl(output_dir / "harmful.jsonl", harmful_records)
    write_jsonl(output_dir / "benign.jsonl", benign_records)
    write_jsonl(output_dir / "combined.jsonl", combined_records)
    write_metadata(
        output_dir / "metadata.json",
        {
            "model": args.model,
            "chat_template": tokenizer.chat_template,
            "sample_size_per_class": args.sample_size,
            "seed": args.seed,
            "harmful_dataset": args.harmful_dataset,
            "harmful_split": args.harmful_split,
            "harmful_column": args.harmful_column,
            "benign_dataset": args.benign_dataset,
            "benign_split": args.benign_split,
            "benign_column": args.benign_column,
            "output_files": {
                "harmful": "harmful.jsonl",
                "benign": "benign.jsonl",
                "combined": "combined.jsonl",
            },
            "matching": summarize_deltas(pairs),
        },
    )

    print(f"Wrote {len(harmful_records)} harmful examples")
    print(f"Wrote {len(benign_records)} length-matched benign examples")
    print(f"Output directory: {output_dir}")
    print(json.dumps(summarize_deltas(pairs), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--harmful-dataset", default=DEFAULT_HARMFUL_DATASET)
    parser.add_argument("--harmful-split", default="train")
    parser.add_argument("--harmful-column")
    parser.add_argument(
        "--advbench-csv-url",
        default=None,
        help=(
            "Load harmful instructions from an AdvBench CSV URL instead of "
            "the Hugging Face dataset. If HF loading fails, the canonical "
            "CSV is used automatically."
        ),
    )
    parser.add_argument("--benign-dataset", default=DEFAULT_ALPACA_DATASET)
    parser.add_argument("--benign-split", default="train")
    parser.add_argument("--benign-column", default="instruction")
    return parser.parse_args()


def main() -> int:
    prepare(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
