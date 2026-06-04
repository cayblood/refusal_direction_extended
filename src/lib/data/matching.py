"""Sampling and length-matching of harmful/benign instruction pairs."""

from __future__ import annotations

import random
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from lib.data.formatting import Example


@dataclass(frozen=True)
class MatchedPair:
    harmful: Example
    benign: Example
    token_length_delta: int


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


def sample_instructions(
    instructions: Sequence[str], sample_size: int, seed: int
) -> list[str]:
    if len(instructions) < sample_size:
        raise RuntimeError(
            f"Need {sample_size} instructions, found {len(instructions)}"
        )
    return random.Random(seed).sample(list(instructions), sample_size)


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
