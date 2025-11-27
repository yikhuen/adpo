#!/usr/bin/env python
"""
Quick sanity check for the OpenRouter judge path (Gemini 2.0 Flash via OpenRouter).
"""

from __future__ import annotations

import json
import os
import sys
from textwrap import dedent

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"
PROMPT = "Reply with the single word 'pong'."


def main() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.stderr.write("OPENROUTER_API_KEY is not set. Export it before running this script.\n")
        return 1

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a connectivity test bot. Reply succinctly."},
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - network failure
        sys.stderr.write(f"OpenRouter request failed: {exc}\n")
        return 2

    content = ""
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        sys.stderr.write(f"Unexpected OpenRouter response: {json.dumps(data, indent=2)}\n")
        return 3

    success = content.lower() == "pong"

    sys.stdout.write(
        dedent(
            f"""
            == OpenRouter Sanity Check ==
            Model: {MODEL}
            Prompt: {PROMPT}
            Response: {content!r}
            Result: {'SUCCESS' if success else 'UNEXPECTED RESPONSE'}
            """
        ).strip()
        + "\n"
    )

    return 0 if success else 4


if __name__ == "__main__":
    raise SystemExit(main())

