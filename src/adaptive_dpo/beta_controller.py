import math
from dataclasses import dataclass


@dataclass
class BetaControllerConfig:
    kl_target: float = 0.03
    eta: float = 0.02
    ema_alpha: float = 0.10
    beta_init: float = 0.10
    beta_min: float = 0.02
    beta_max: float = 3.0
    deadband_ratio: float = 0.20  # ±20%
    use_ema: bool = True
    use_deadband: bool = True
    use_clipping: bool = True


class AdaptiveBetaController:
    def __init__(self, cfg: BetaControllerConfig):
        self.cfg = cfg
        self.beta = cfg.beta_init
        self.kl_ema = 0.0
        self.kl_last = 0.0

    def update(self, kl_batch: float) -> float:
        self.kl_last = float(kl_batch)
        # EMA smoothing
        if self.cfg.use_ema:
            self.kl_ema = (1.0 - self.cfg.ema_alpha) * self.kl_ema + self.cfg.ema_alpha * self.kl_last
        else:
            self.kl_ema = self.kl_last

        # Deadband around target
        if self.cfg.use_deadband:
            lower = self.cfg.kl_target * (1.0 - self.cfg.deadband_ratio)
            upper = self.cfg.kl_target * (1.0 + self.cfg.deadband_ratio)
            if lower <= self.kl_ema <= upper:
                return self.beta

        # Adaptive multiplicative update toward target KL
        factor = math.exp(self.cfg.eta * (self.kl_ema / max(self.cfg.kl_target, 1e-8) - 1.0))
        self.beta *= factor

        # Clipping
        if self.cfg.use_clipping:
            self.beta = max(self.cfg.beta_min, min(self.beta, self.cfg.beta_max))
        return self.beta

    def state(self):
        return {
            "beta": self.beta,
            "kl_ema": self.kl_ema,
            "kl_batch": self.kl_last,
            "kl_target": self.cfg.kl_target,
        }
