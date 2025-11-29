from __future__ import annotations

from typing import Any, Dict, Optional

from .utils import apply_chat_prompt

DEFAULT_PATH = "meg-tong/sycophancy-eval"


def format_example(
    example: Dict[str, Any],
    tokenizer,
    format_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    format_kwargs = format_kwargs or {}
    question_key = format_kwargs.get("question_key", "question")
    chosen_key = format_kwargs.get("preferred_key", "answer_not_matching_behavior")
    rejected_key = format_kwargs.get("rejected_key", "answer_matching_behavior")
    system_prompt = format_kwargs.get("system_prompt")

    prompt_value = example.get(question_key)
    if prompt_value is None:
        raise KeyError(f"Sycophancy example missing '{question_key}' column.")

    chosen_value = example.get(chosen_key)
    rejected_value = example.get(rejected_key)
    if chosen_value is None or rejected_value is None:
        raise KeyError(
            f"Sycophancy example missing preferred ('{chosen_key}') or rejected ('{rejected_key}') response columns. "
            "Ensure you're using the paired preference split from meg-tong/sycophancy-eval."
        )

    prompt_text = apply_chat_prompt(
        tokenizer,
        system_prompt,
        [{"role": "user", "content": prompt_value}],
    )
    return {
        "prompt": prompt_text,
        "chosen": chosen_value,
        "rejected": rejected_value,
    }

