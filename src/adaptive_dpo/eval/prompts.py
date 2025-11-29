from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for entry in value:
            parts.append(_stringify_content(entry))
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return _stringify_content(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        segments: List[str] = []
        for entry in value:
            if entry is None:
                continue
            if isinstance(entry, dict):
                role = entry.get("type") or entry.get("role")
                content = _stringify_content(entry.get("content"))
                if role:
                    segments.append(f"{role.capitalize()}: {content}".strip())
                else:
                    segments.append(content)
            else:
                segments.append(_stringify_content(entry))
        return "\n\n".join(segment for segment in segments if segment).strip()
    if isinstance(value, dict):
        return _stringify_content(value)
    return str(value)


def load_prompts(cfg: Dict[str, Any], override_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Compatibility: Ensure 'prompt' key exists and is a plain string
    for item in data:
        if "prompt" not in item and "question" in item:
            item["prompt"] = item["question"]
        item["prompt"] = _normalize_prompt_text(item.get("prompt", ""))

    limit = override_limit or cfg.get("limit")
    if limit:
        limit = min(limit, len(data))
        if cfg.get("shuffle", False):
            random.seed(cfg.get("seed", 42))
            random.shuffle(data)
        data = data[:limit]
    return data


def strip_prompt(prompt_text: str, full_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return full_text.strip()

