"""Load and normalize instruction datasets (harmful, benign, generic)."""

from __future__ import annotations

import csv
import urllib.request
from collections.abc import Iterable
from typing import Any, cast

from datasets import Dataset, load_dataset

DEFAULT_HARMFUL_DATASET = "walledai/AdvBench"
DEFAULT_ALPACA_DATASET = "tatsu-lab/alpaca"
DEFAULT_GENERIC_DATASET = "databricks/databricks-dolly-15k"
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


def load_harmful_from_hf(
    dataset_name: str, split: str, column: str | None
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


def load_harmful_instructions(
    *,
    harmful_dataset: str,
    harmful_split: str,
    harmful_column: str | None,
    advbench_csv_url: str | None,
) -> list[str]:
    if advbench_csv_url:
        return load_harmful_from_csv(advbench_csv_url, harmful_column)

    try:
        return load_harmful_from_hf(
            harmful_dataset, harmful_split, harmful_column
        )
    except Exception as exc:  # noqa: BLE001 - fallback keeps setup practical
        print(
            "Could not load AdvBench from Hugging Face; falling back to "
            f"canonical CSV. Original error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return load_harmful_from_csv(DEFAULT_ADV_BENCH_CSV_URL, harmful_column)


def load_benign_instructions(
    *, benign_dataset: str, benign_split: str, benign_column: str | None
) -> list[str]:
    dataset = cast(
        Dataset, load_dataset(benign_dataset, split=benign_split)
    )
    text_column = choose_text_column(dataset.column_names, benign_column)
    return compact_unique(dataset[text_column])


def load_generic_instructions(
    *, dataset_name: str, split: str, column: str | None, context_free: bool
) -> list[str]:
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))
    if context_free and "context" in dataset.column_names:
        dataset = dataset.filter(
            lambda row: not str(row.get("context") or "").strip()
        )
    text_column = choose_text_column(dataset.column_names, column)
    return compact_unique(dataset[text_column])
