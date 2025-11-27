from __future__ import annotations

from typing import Dict, List, Optional


def apply_chat_prompt(tokenizer, system: Optional[str], messages: List[Dict[str, str]]) -> str:
    prompt_messages: List[Dict[str, str]] = []
    if system:
        prompt_messages.append({"role": "system", "content": system})
    prompt_messages.extend(messages)
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

