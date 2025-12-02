from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimPOConfig:
    """Default hyperparameters for SimPO loss."""

    gamma: float = 0.5


