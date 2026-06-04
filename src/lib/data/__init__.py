"""Dataset preparation, formatting, and record I/O."""

from lib.data.formatting import (
    Example,
    build_examples,
    format_chat_prompt,
    token_count,
)
from lib.data.matching import (
    MatchedPair,
    length_match_pairs,
    sample_harmful,
    sample_instructions,
    summarize_deltas,
)
from lib.data.records import (
    example_record,
    load_records,
    records_by_pair_id,
    select_prompts,
    write_jsonl,
    write_metadata,
)
from lib.data.sources import (
    DEFAULT_ADV_BENCH_CSV_URL,
    DEFAULT_ALPACA_DATASET,
    DEFAULT_GENERIC_DATASET,
    DEFAULT_HARMFUL_DATASET,
    choose_text_column,
    compact_unique,
    load_benign_instructions,
    load_generic_instructions,
    load_harmful_instructions,
)

__all__ = [
    "DEFAULT_ADV_BENCH_CSV_URL",
    "DEFAULT_ALPACA_DATASET",
    "DEFAULT_GENERIC_DATASET",
    "DEFAULT_HARMFUL_DATASET",
    "Example",
    "MatchedPair",
    "build_examples",
    "choose_text_column",
    "compact_unique",
    "example_record",
    "format_chat_prompt",
    "length_match_pairs",
    "load_benign_instructions",
    "load_generic_instructions",
    "load_harmful_instructions",
    "load_records",
    "records_by_pair_id",
    "sample_harmful",
    "sample_instructions",
    "select_prompts",
    "summarize_deltas",
    "token_count",
]
