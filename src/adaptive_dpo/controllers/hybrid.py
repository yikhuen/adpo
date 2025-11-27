from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class HybridControllerConfig:
    """Configuration for the Hybrid Entropy Adaptive controller."""

    beta_init: float = 0.10
    beta_min: float = 0.001
    beta_max: float = 10.0
    target_kl: float = 0.10
    alpha_rate: float = 0.01
    lambda_entropy: float = 1.0
    vocab_size: int = 32000
    entropy_warmup_steps: int = 0
    deadband_ratio: float = 0.10


class HybridAdaptiveKLController:
    """EMA baseline + entropy spike controller."""

    requires_entropy: bool = True

    def __init__(self, cfg: HybridControllerConfig):
        self.cfg = cfg
        self.beta_base = float(cfg.beta_init)
        self.beta_min = float(cfg.beta_min)
        self.beta_max = float(cfg.beta_max)
        self.target_kl = float(cfg.target_kl)
        self.alpha_rate = float(cfg.alpha_rate)
        self.lambda_entropy = float(cfg.lambda_entropy)
        self.entropy_warmup_steps = int(cfg.entropy_warmup_steps or 0)
        self.max_entropy = math.log(max(2, int(cfg.vocab_size)))
        self.step = 0
        self.last_entropy_scalar = 1.0
        self.last_normalized_entropy = 0.0

    def _normalized_entropy(
        self,
        batch_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> float:
        logits = batch_logits.detach().float()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        if attention_mask is not None:
            mask = attention_mask.to(token_entropy.dtype)
            sum_entropy = (token_entropy * mask).sum()
            token_count = mask.sum().clamp_min(1.0)
        else:
            sum_entropy = token_entropy.sum()
            token_count = torch.tensor(token_entropy.numel(), device=token_entropy.device, dtype=token_entropy.dtype)
        avg_entropy = (sum_entropy / token_count).item()
        normalized = avg_entropy / self.max_entropy
        return float(max(0.0, min(1.0, normalized)))

    def update(
        self,
        kl_batch: float,
        batch_logits: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        **_: Any,
    ) -> float:
        self.step = int(global_step) if global_step is not None else self.step + 1

        entropy_scalar = 1.0
        normalized_entropy = 0.0
        use_entropy = (
            self.lambda_entropy > 0.0 and batch_logits is not None and (self.step >= self.entropy_warmup_steps)
        )
        if use_entropy:
            normalized_entropy = self._normalized_entropy(batch_logits, attention_mask)
            entropy_scalar = 1.0 + self.lambda_entropy * normalized_entropy

        target = max(self.target_kl, 1e-8)
        error_ratio = (kl_batch / target) - 1.0
        if abs(error_ratio) > self.cfg.deadband_ratio:
            factor = math.exp(self.alpha_rate * error_ratio)
            self.beta_base *= factor

        self.beta_base = max(self.beta_min, min(self.beta_max, self.beta_base))
        final_beta = self.beta_base * entropy_scalar

        self.last_entropy_scalar = entropy_scalar
        self.last_normalized_entropy = normalized_entropy

        return float(final_beta)

    def state(self) -> Dict[str, Any]:
        beta_total = float(self.beta_base * self.last_entropy_scalar)
        return {
            "beta": beta_total,
            "beta_total": beta_total,
            "beta_base": float(self.beta_base),
            "entropy_scalar": float(self.last_entropy_scalar),
            "normalized_entropy": float(self.last_normalized_entropy),
            "target_kl": float(self.target_kl),
        }

