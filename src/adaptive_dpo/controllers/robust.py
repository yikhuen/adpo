from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class RobustHybridConfig:
    target_kl: float = 0.04
    beta_init: float = 0.10
    beta_min: float = 0.05
    beta_max: float = 2.0
    eta: float = 0.01
    ema_alpha: float = 0.1
    deadband: float = 0.10
    lambda_entropy: float = 4.0
    vocab_size: int = 32000
    entropy_warmup_steps: int = 10


class RobustHybridController:
    requires_entropy: bool = True

    def __init__(self, cfg: RobustHybridConfig):
        self.cfg = cfg
        self.beta_base = cfg.beta_init
        self.kl_ema = cfg.target_kl
        self.step_count = 0
        self.max_entropy = math.log(max(2, int(cfg.vocab_size)))
        self.last_metrics: Dict[str, Any] = {}

    def _calc_entropy_scalar(
        self,
        batch_logits: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> float:
        if batch_logits is None:
            return 1.0

        logits = batch_logits.detach().float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)

        if attention_mask is not None:
            mask = attention_mask.float()
            avg_entropy = (token_entropy * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            avg_entropy = token_entropy.mean()

        norm_entropy = (avg_entropy / self.max_entropy).clamp(min=0.0).item()
        return 1.0 + (self.cfg.lambda_entropy * norm_entropy)

    def update(
        self,
        kl_batch: float,
        batch_logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> float:
        self.step_count += 1

        self.kl_ema = (1.0 - self.cfg.ema_alpha) * self.kl_ema + self.cfg.ema_alpha * kl_batch
        sensor_kl = self.kl_ema

        lower = self.cfg.target_kl * (1.0 - self.cfg.deadband)
        upper = self.cfg.target_kl * (1.0 + self.cfg.deadband)

        if not (lower <= sensor_kl <= upper):
            error = (sensor_kl - self.cfg.target_kl) / max(self.cfg.target_kl, 1e-8)
            self.beta_base *= math.exp(self.cfg.eta * error)
            self.beta_base = max(self.cfg.beta_min, min(self.beta_base, self.cfg.beta_max))

        entropy_scalar = 1.0
        if self.step_count > self.cfg.entropy_warmup_steps:
            entropy_scalar = self._calc_entropy_scalar(batch_logits, attention_mask)

        final_beta = self.beta_base * entropy_scalar
        self.last_metrics = {
            "beta_total": float(final_beta),
            "beta_base": float(self.beta_base),
            "entropy_scalar": float(entropy_scalar),
            "kl_ema": float(self.kl_ema),
            "kl_batch": float(kl_batch),
        }
        return float(final_beta)

    def state(self) -> Dict[str, Any]:
        return self.last_metrics

