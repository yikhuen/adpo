from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class BetaDPOConfig:
    """Configuration for Beta-DPO (Wu et al., 2024)."""

    beta_min: float = 0.1
    beta_max: float = 0.5
    scale_coeff: float = 0.1
    smooth_alpha: float = 0.05  # EMA factor for stability


class BetaDPOController:
    """
    Implements Beta-DPO logic where beta adapts to the data difficulty.
    
    Logic:
      - Observe reward margin M = r_chosen - r_rejected (unscaled by beta).
      - Large M (easy sample) -> Higher beta (trust preference more).
      - Small M (hard sample) -> Lower beta (trust reference more).
      
    Implementation:
      - Uses EMA of the margin to avoid rapid oscillations.
      - beta = clamp(scale_coeff * margin_ema, min, max).
    """
    requires_entropy: bool = False

    def __init__(self, cfg: BetaDPOConfig):
        self.cfg = cfg
        # Initialize beta to a safe middle value or min
        self.beta = cfg.beta_min
        self.margin_ema = 0.0
        self.step_count = 0
        self.last_metrics: Dict[str, float] = {}

    def update(
        self,
        kl_batch: float,
        metrics: Optional[Dict[str, float]] = None,
        **_: Any,
    ) -> float:
        self.step_count += 1
        
        # If no metrics (e.g. first step or evaluation mode), return current beta
        if not metrics:
            return self.beta

        # Extract margin. TRL logs 'rewards/margins' which is typically scaled by the current beta.
        # We need the "raw" margin (log probability ratio difference) to be beta-independent.
        # scaled_margin = beta * raw_margin
        # raw_margin = scaled_margin / beta
        scaled_margin = metrics.get("rewards/margins", 0.0)
        
        # Avoid division by zero
        current_beta = max(self.beta, 1e-6)
        raw_margin = scaled_margin / current_beta

        # Update margin EMA
        if self.step_count == 1:
            self.margin_ema = raw_margin
        else:
            alpha = self.cfg.smooth_alpha
            self.margin_ema = (1 - alpha) * self.margin_ema + alpha * raw_margin

        # Calculate target beta
        # Heuristic: beta ~ proportional to margin
        target_beta = self.cfg.scale_coeff * self.margin_ema
        
        # Clamp
        self.beta = max(self.cfg.beta_min, min(target_beta, self.cfg.beta_max))
        
        self.last_metrics = {
            "beta": self.beta,
            "beta_total": self.beta,
            "margin_raw": raw_margin,
            "margin_ema": self.margin_ema,
        }
        return self.beta

    def state(self) -> Dict[str, Any]:
        return self.last_metrics
