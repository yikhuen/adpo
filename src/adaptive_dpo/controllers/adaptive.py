from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BetaControllerConfig:
    """Configuration for the PID-like Adaptive Beta Controller."""

    kl_target: float = 0.03
    eta: float = 1.0
    ema_alpha: float = 0.10
    beta_init: float = 0.10
    beta_min: float = 0.05
    beta_max: float = 2.0
    deadband_ratio: float = 0.15
    use_hybrid_sensor: bool = True
    use_ema: bool = True
    use_deadband: bool = True
    use_clipping: bool = True


class AdaptiveBetaController:
    """Hybrid EMA-based beta controller used in legacy phases."""

    def __init__(self, cfg: BetaControllerConfig):
        self.cfg = cfg
        self.beta = cfg.beta_init
        self.kl_ema = cfg.kl_target
        self.kl_last = cfg.kl_target

    def update(self, kl_batch: float, **_: Any) -> float:
        self.kl_last = float(kl_batch)
        self.kl_ema = (1.0 - self.cfg.ema_alpha) * self.kl_ema + self.cfg.ema_alpha * self.kl_last

        if self.cfg.use_hybrid_sensor:
            control_signal = max(self.kl_ema, self.kl_last)
        else:
            control_signal = self.kl_ema

        lower = self.cfg.kl_target * (1.0 - self.cfg.deadband_ratio)
        upper = self.cfg.kl_target * (1.0 + self.cfg.deadband_ratio)
        if self.cfg.use_deadband and lower <= control_signal <= upper:
            return self.beta

        error_ratio = (control_signal / max(self.cfg.kl_target, 1e-8)) - 1.0
        factor = math.exp(self.cfg.eta * error_ratio)
        self.beta *= factor

        if self.cfg.use_clipping:
            self.beta = max(self.cfg.beta_min, min(self.beta, self.cfg.beta_max))
        return self.beta

    def state(self) -> Dict[str, float]:
        beta_val = float(self.beta)
        return {
            "beta": beta_val,
            "beta_total": beta_val,
            "beta_base": beta_val,
            "entropy_scalar": 1.0,
            "kl_ema": float(self.kl_ema),
            "kl_batch": float(self.kl_last),
            "error_ratio": (self.kl_ema / self.cfg.kl_target) - 1.0,
        }

