from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_prompts(cfg: Dict[str, Any], override_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    # Compatibility: Ensure 'prompt' key exists if 'question' is present
    for item in data:
        if "prompt" not in item and "question" in item:
            item["prompt"] = item["question"]
            
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

