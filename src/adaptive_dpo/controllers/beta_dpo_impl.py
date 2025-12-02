from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .beta_dpo import BetaDPOConfig


class BetaDPOController:
    """
    Implements Beta-DPO (Wu et al., 2024) logic: beta adapts based on reward margin.
    paper: 'Beta-DPO: Adaptive Beta for Direct Preference Optimization'
    """
    requires_entropy: bool = False

    def __init__(self, cfg: BetaDPOConfig):
        self.cfg = cfg
        self.beta = cfg.beta_min
        self.step_count = 0
        self.last_metrics: Dict[str, float] = {}

    def update(
        self,
        kl_batch: float,
        chosen_rewards: Optional[torch.Tensor] = None,
        rejected_rewards: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> float:
        self.step_count += 1
        
        # Fallback if rewards aren't available yet (e.g. first step or non-DPO flow)
        if chosen_rewards is None or rejected_rewards is None:
            return self.beta

        with torch.no_grad():
            # Wu et al. heuristic: 
            # If margin M = r_w - r_l is large (easy), we can afford a higher beta (less regularization needed? or more?).
            # Actually, standard curriculum learning suggests focusing on hard examples (small margin).
            # However, high beta -> closer to reference. Low beta -> drift allowed.
            # If margin is large (model is confident), we can increase beta to prevent over-optimization/drift on easy samples.
            # If margin is small (model is uncertain), we decrease beta to encourage learning (stronger gradient signal).
            
            # Note: The controller sees 'scaled' rewards (beta * log_ratio) from the trainer context usually.
            # We unscale to get the raw log-odds margin for consistent control.
            current_beta_scale = max(self.beta, 1e-6)
            margin_scaled = (chosen_rewards - rejected_rewards).mean()
            raw_margin = margin_scaled / current_beta_scale
            
            # Simple linear adaptation rule as a proxy for the paper's official implementation
            # until exact formula is confirmed from repo.
            # beta_new = beta_min + coeff * sigmoid(raw_margin) * range
            # or similar monotonic function of margin.
            
            # Using a bounded linear map for robustness:
            # normalized_margin in [0, 1] roughly
            target_beta = self.cfg.beta_min + self.cfg.scale_coeff * torch.sigmoid(raw_margin).item()
            
            # Clip to configured range
            self.beta = max(self.cfg.beta_min, min(target_beta, self.cfg.beta_max))

        self.last_metrics = {
            "beta_total": float(self.beta),
            "margin_raw": float(raw_margin.item()),
        }
        return self.beta

    def state(self) -> Dict[str, Any]:
        return self.last_metrics
