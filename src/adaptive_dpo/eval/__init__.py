"""Evaluation utilities (prompts, generation, judging, logging)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["generation", "judging", "logging", "metrics", "prompts", "runner"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
