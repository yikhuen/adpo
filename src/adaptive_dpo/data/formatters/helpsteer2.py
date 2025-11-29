from __future__ import annotations

from typing import Any, Dict, Optional

from .utils import apply_chat_prompt

DEFAULT_PATH = "nvidia/HelpSteer2"


def format_example(
    example: Dict[str, Any],
    tokenizer,
    format_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    format_kwargs = format_kwargs or {}
    prompt_key = format_kwargs.get("prompt_key", "prompt")
    chosen_key = format_kwargs.get("chosen_key", "chosen")
    rejected_key = format_kwargs.get("rejected_key", "rejected")
    fallback_response_key = format_kwargs.get("response_key", "response")

    if prompt_key not in example:
        raise KeyError(f"HelpSteer2 example missing '{prompt_key}' column.")

    chosen_value = example.get(chosen_key)
    rejected_value = example.get(rejected_key)

    if chosen_value is None and fallback_response_key in example:
        chosen_value = example[fallback_response_key]
    if rejected_value is None and fallback_response_key in example:
        rejected_value = example[fallback_response_key]

    if chosen_value is None or rejected_value is None:
        raise KeyError(
            f"HelpSteer2 example missing '{chosen_key}' and/or '{rejected_key}' columns or "
            f"fallback response '{fallback_response_key}'. Provide a preference-paired split or "
            "preprocess pairs before evaluation."
        )

    prompt_text = apply_chat_prompt(
        tokenizer,
        None,
        [{"role": "user", "content": example[prompt_key]}],
    )

    return {
        "prompt": prompt_text,
        "chosen": chosen_value,
        "rejected": rejected_value,
    }

