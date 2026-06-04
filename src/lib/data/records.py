"""JSONL dataset record I/O and prompt selection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from lib.data.formatting import Example


def example_record(
    example: Example, label: str, pair_id: int, token_length_delta: int
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


def load_records(
    dataset_dir: Path, label: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load prepared prompts for one class, optionally truncated."""
    path = dataset_dir / f"{label}.jsonl"
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. Run scripts/01_prepare_datasets.py first."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def records_by_pair_id(
    dataset_dir: Path, label: str
) -> dict[int, dict[str, Any]]:
    path = dataset_dir / f"{label}.jsonl"
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. Run scripts/01_prepare_datasets.py first."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {int(row["pair_id"]): row for row in rows}


def select_prompts(
    records: dict[int, dict[str, Any]], pair_ids: Sequence[int], count: int
) -> list[str]:
    chosen = [pid for pid in pair_ids if pid in records][:count]
    return [cast(str, records[pid]["formatted_prompt"]) for pid in chosen]
