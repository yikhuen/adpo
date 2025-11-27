#!/usr/bin/env python
"""Quick sanity check for Gemini API connectivity."""

from __future__ import annotations

import os
import sys
from textwrap import dedent

import google.generativeai as genai


PROMPT = "Reply with the single word 'pong'."


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.stderr.write(
            "GEMINI_API_KEY is not set. Export it before running this script.\n"
        )
        return 1

    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(PROMPT)
    except Exception as exc:  # pragma: no cover - network issues
        sys.stderr.write(f"Gemini request failed: {exc}\n")
        return 2

    text = (response.text or "").strip()
    success = text.lower() == "pong"

    sys.stdout.write(
        dedent(
            f"""
            == Gemini Sanity Check ==
            Model: gemini-2.0-flash-001
            Prompt: {PROMPT}
            Response: {text!r}
            Result: {'SUCCESS' if success else 'UNEXPECTED RESPONSE'}
            """
        ).strip()
        + "\n"
    )

    return 0 if success else 3


if __name__ == "__main__":
    raise SystemExit(main())

