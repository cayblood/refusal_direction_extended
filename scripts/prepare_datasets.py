"""Prepare harmful/benign instruction datasets for refusal-direction work.

The first reproduction step needs harmful and benign prompts formatted with the
exact target model chat template. This script samples harmful instructions from
AdvBench, length-matches benign instructions from Alpaca, and writes JSONL files
containing both raw instructions and Llama-chat-formatted prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT_DIR = Path("data/refusal_datasets")
DEFAULT_HARMFUL_DATASET = "walledai/AdvBench"
DEFAULT_ALPACA_DATASET = "tatsu-lab/alpaca"
DEFAULT_ADV_BENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
TEXT_COLUMNS = (
    "goal",
    "instruction",
    "prompt",
    "query",
    "question",
    "text",
)


@dataclass(frozen=True)
class Example:
    source: str
    source_index: int
    instruction: str
    raw_token_count: int
    formatted_prompt: str
    formatted_token_count: int


@dataclass(frozen=True)
class MatchedPair:
    harmful: Example
    benign: Example
    token_length_delta: int


def normalize_instruction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


def choose_text_column(columns: Iterable[str], preferred: str | None) -> str:
    column_set = set(columns)
    if preferred:
        if preferred not in column_set:
            raise ValueError(
                f"Requested text column {preferred!r} was not found. "
                f"Available columns: {sorted(column_set)!r}"
            )
        return preferred
    for column in TEXT_COLUMNS:
        if column in column_set:
            return column
    raise ValueError(
        "Could not infer an instruction column. Available columns: "
        f"{sorted(column_set)!r}"
    )


def load_harmful_from_hf(
    dataset_name: str,
    split: str,
    column: str | None,
) -> list[str]:
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))
    text_column = choose_text_column(dataset.column_names, column)
    return compact_unique(dataset[text_column])


def load_harmful_from_csv(url: str, column: str | None) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as response:
        decoded = response.read().decode("utf-8")
    rows = list(csv.DictReader(decoded.splitlines()))
    if not rows:
        raise RuntimeError(f"AdvBench CSV at {url} had no rows")
    text_column = choose_text_column(rows[0].keys(), column)
    return compact_unique(row[text_column] for row in rows)


def load_harmful_instructions(args: argparse.Namespace) -> list[str]:
    if args.advbench_csv_url:
        return load_harmful_from_csv(
            cast(str, args.advbench_csv_url),
            cast(str | None, args.harmful_column),
        )

    try:
        return load_harmful_from_hf(
            cast(str, args.harmful_dataset),
            cast(str, args.harmful_split),
            cast(str | None, args.harmful_column),
        )
    except Exception as exc:  # noqa: BLE001 - fallback keeps setup practical
        print(
            "Could not load AdvBench from Hugging Face; falling back to "
            f"canonical CSV. Original error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return load_harmful_from_csv(
            DEFAULT_ADV_BENCH_CSV_URL, cast(str | None, args.harmful_column)
        )


def load_benign_instructions(args: argparse.Namespace) -> list[str]:
    dataset = cast(
        Dataset,
        load_dataset(
            cast(str, args.benign_dataset),
            split=cast(str, args.benign_split),
        ),
    )
    text_column = choose_text_column(
        dataset.column_names, cast(str | None, args.benign_column)
    )
    return compact_unique(dataset[text_column])


def compact_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    instructions: list[str] = []
    for value in values:
        instruction = normalize_instruction(value)
        if instruction is None or instruction in seen:
            continue
        seen.add(instruction)
        instructions.append(instruction)
    return instructions


def format_chat_prompt(
    tokenizer: PreTrainedTokenizerBase, instruction: str
) -> str:
    if tokenizer.chat_template is None:
        raise RuntimeError(
            "Tokenizer has no chat_template. Use a Llama 3.2 Instruct "
            "tokenizer for reproduction datasets."
        )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return cast(str, prompt)


def token_count(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_examples(
    *,
    source: str,
    instructions: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
) -> list[Example]:
    examples: list[Example] = []
    for source_index, instruction in enumerate(instructions):
        formatted_prompt = format_chat_prompt(tokenizer, instruction)
        examples.append(
            Example(
                source=source,
                source_index=source_index,
                instruction=instruction,
                raw_token_count=token_count(tokenizer, instruction),
                formatted_prompt=formatted_prompt,
                formatted_token_count=token_count(tokenizer, formatted_prompt),
            )
        )
    return examples


def sample_harmful(
    instructions: Sequence[str], sample_size: int, seed: int
) -> list[str]:
    if len(instructions) < sample_size:
        raise RuntimeError(
            f"Need {sample_size} harmful instructions, found "
            f"{len(instructions)}"
        )
    rng = random.Random(seed)
    return rng.sample(list(instructions), sample_size)


def pop_nearest_by_length(
    sorted_examples: list[tuple[int, int, Example]], target_length: int
) -> Example:
    if not sorted_examples:
        raise RuntimeError("No benign examples remain for length matching")

    insertion_index = bisect_left(
        sorted_examples, (target_length, -1, sorted_examples[0][2])
    )
    candidate_indexes = []
    if insertion_index < len(sorted_examples):
        candidate_indexes.append(insertion_index)
    if insertion_index > 0:
        candidate_indexes.append(insertion_index - 1)
    best_index = min(
        candidate_indexes,
        key=lambda index: (
            abs(sorted_examples[index][0] - target_length),
            sorted_examples[index][1],
        ),
    )
    return sorted_examples.pop(best_index)[2]


def length_match_pairs(
    harmful_examples: Sequence[Example], benign_examples: Sequence[Example]
) -> list[MatchedPair]:
    sorted_benign = sorted(
        (
            (example.raw_token_count, example.source_index, example)
            for example in benign_examples
        ),
        key=lambda item: (item[0], item[1]),
    )
    pairs: list[MatchedPair] = []
    for harmful in harmful_examples:
        benign = pop_nearest_by_length(sorted_benign, harmful.raw_token_count)
        pairs.append(
            MatchedPair(
                harmful=harmful,
                benign=benign,
                token_length_delta=benign.raw_token_count
                - harmful.raw_token_count,
            )
        )
    return pairs


def example_record(
    example: Example,
    label: str,
    pair_id: int,
    token_length_delta: int,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "label": label,
        "source": example.source,
        "source_index": example.source_index,
        "instruction": example.instruction,
        "raw_token_count": example.raw_token_count,
        "formatted_prompt": example.formatted_prompt,
        "formatted_token_count": example.formatted_token_count,
        "match_raw_token_delta": token_length_delta,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def summarize_deltas(pairs: Sequence[MatchedPair]) -> dict[str, float | int]:
    deltas = [abs(pair.token_length_delta) for pair in pairs]
    signed = [pair.token_length_delta for pair in pairs]
    if not deltas:
        return {
            "count": 0,
            "mean_abs_raw_token_delta": 0.0,
            "max_abs_raw_token_delta": 0,
            "mean_signed_raw_token_delta": 0.0,
        }
    return {
        "count": len(deltas),
        "mean_abs_raw_token_delta": round(sum(deltas) / len(deltas), 3),
        "max_abs_raw_token_delta": max(deltas),
        "mean_signed_raw_token_delta": round(sum(signed) / len(signed), 3),
    }


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

    harmful_instructions = load_harmful_instructions(args)
    benign_instructions = load_benign_instructions(args)
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
    combined_records = [
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
