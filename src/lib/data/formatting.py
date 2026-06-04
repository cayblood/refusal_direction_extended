"""Chat-template formatting and example construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Example:
    source: str
    source_index: int
    instruction: str
    raw_token_count: int
    formatted_prompt: str
    formatted_token_count: int


def format_chat_prompt(
    tokenizer: PreTrainedTokenizerBase, instruction: str
) -> str:
    """Apply the model's native chat template exactly once."""
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
