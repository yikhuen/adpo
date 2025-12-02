from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .epsilon_dpo import EpsilonDPOConfig


class EpsilonDPOController:
    requires_entropy: bool = False

    def __init__(self, cfg: EpsilonDPOConfig):
        self.cfg = cfg
        self.beta = cfg.beta_init
        self.step_count = 0
        self.last_metrics: Dict[str, float] = {}

    def update(self, kl_batch: float, **_: Any) -> float:
        # Placeholder for Epsilon-DPO logic (Lee et al., 2025)
        # Usually involves checking monotonicity under perturbation.
        # Without full implementation details/repo access, we keep a stub
        # that acts like a fixed beta or simple schedule for now,
        # unless we add the perturbation step in the trainer.
        
        self.step_count += 1
        self.last_metrics = {"beta_total": self.beta}
        return self.beta

    def state(self) -> Dict[str, Any]:
        return self.last_metrics

