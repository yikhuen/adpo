from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .utils import apply_chat_prompt

DEFAULT_PATH = "Anthropic/hh-rlhf"
TURN_PATTERN = re.compile(r"(human|assistant):", re.IGNORECASE)


def _parse_conversation(text: str) -> List[Dict[str, str]]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        raise ValueError("Anthropic HH example is empty.")

    matches = list(TURN_PATTERN.finditer(normalized))
    if len(matches) < 2:
        raise ValueError("Anthropic HH example must contain at least two labeled turns.")

    messages: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        role_token = match.group(1).lower()
        role = "user" if role_token == "human" else "assistant"
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        content = normalized[start:end].strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("Anthropic HH example produced no chat messages.")
    return messages


def format_example(
    example: Dict[str, Any],
    tokenizer,
    format_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    format_kwargs = format_kwargs or {}
    system_prompt = format_kwargs.get("system_prompt")

    chosen_msgs = _parse_conversation(example["chosen"])
    rejected_msgs = _parse_conversation(example["rejected"])
    if chosen_msgs[-1]["role"] != "assistant":
        raise ValueError("Anthropic HH example does not end with an assistant reply (chosen).")
    if rejected_msgs[-1]["role"] != "assistant":
        raise ValueError("Anthropic HH example does not end with an assistant reply (rejected).")

    context_messages = chosen_msgs[:-1]
    if not context_messages:
        raise ValueError("Anthropic HH example missing user context before assistant reply.")

    prompt_text = apply_chat_prompt(tokenizer, system_prompt, context_messages)
    return {
        "prompt": prompt_text,
        "chosen": chosen_msgs[-1]["content"],
        "rejected": rejected_msgs[-1]["content"],
    }

