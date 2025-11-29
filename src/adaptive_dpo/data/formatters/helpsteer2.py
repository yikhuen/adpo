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

    if prompt_key not in example:
        raise KeyError(f"HelpSteer2 example missing '{prompt_key}' column.")
    if chosen_key not in example or rejected_key not in example:
        raise KeyError(
            f"HelpSteer2 example missing '{chosen_key}' and/or '{rejected_key}' columns. "
            "Ensure you are using the preference-paired split (e.g., "
            "nvidia/HelpSteer2) or preprocess pairs before evaluation."
        )

    prompt_text = apply_chat_prompt(
        tokenizer,
        None,
        [{"role": "user", "content": example[prompt_key]}],
    )

    return {
        "prompt": prompt_text,
        "chosen": example[chosen_key],
        "rejected": example[rejected_key],
    }

