from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KTOAlgorithmConfig:
    """Hyperparameters controlling the KTO trainer loss weights."""

    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


