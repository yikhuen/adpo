from __future__ import annotations

from typing import Any, Dict, List, Optional

from .utils import apply_chat_prompt

DEFAULT_PATH = "HuggingFaceH4/ultrafeedback_binarized"


def _extract_fields(example: Dict[str, Any]) -> Dict[str, str]:
    chosen_msgs = example["chosen"]
    rejected_msgs = example["rejected"]

    system_text = ""
    if chosen_msgs and chosen_msgs[0].get("role") == "system":
        system_text = chosen_msgs[0].get("content", "")

    user_text = ""
    for message in chosen_msgs:
        if message.get("role") == "user":
            user_text = message.get("content", "")
            break

    def _first_assistant(messages: List[Dict[str, str]]) -> str:
        for message in messages:
            if message.get("role") == "assistant":
                return message.get("content", "")
        return ""

    return {
        "system": system_text,
        "user": user_text,
        "chosen": _first_assistant(chosen_msgs),
        "rejected": _first_assistant(rejected_msgs),
    }


def format_example(
    example: Dict[str, Any],
    tokenizer,
    format_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    parts = _extract_fields(example)
    messages = [{"role": "user", "content": parts["user"]}]
    prompt_text = apply_chat_prompt(tokenizer, parts["system"], messages)
    return {
        "prompt": prompt_text,
        "chosen": parts["chosen"],
        "rejected": parts["rejected"],
    }

