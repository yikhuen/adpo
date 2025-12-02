from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IPOMethodConfig:
    """Configuration stub for IPO-style preference training."""

    label: str = "ipo"


