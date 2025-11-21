# import math
# from dataclasses import dataclass


# @dataclass
# class BetaControllerConfig:
#     kl_target: float = 0.03
#     eta: float = 0.02
#     ema_alpha: float = 0.10
#     beta_init: float = 0.10
#     beta_min: float = 0.02
#     beta_max: float = 3.0
#     deadband_ratio: float = 0.20  # ±20%
#     use_ema: bool = True
#     use_deadband: bool = True
#     use_clipping: bool = True


# class AdaptiveBetaController:
#     def __init__(self, cfg: BetaControllerConfig):
#         self.cfg = cfg
#         self.beta = cfg.beta_init
#         self.kl_ema = 0.0
#         self.kl_last = 0.0

#     def update(self, kl_batch: float) -> float:
#         self.kl_last = float(kl_batch)
#         # EMA smoothing
#         if self.cfg.use_ema:
#             self.kl_ema = (1.0 - self.cfg.ema_alpha) * self.kl_ema + self.cfg.ema_alpha * self.kl_last
#         else:
#             self.kl_ema = self.kl_last

#         # Deadband around target
#         if self.cfg.use_deadband:
#             lower = self.cfg.kl_target * (1.0 - self.cfg.deadband_ratio)
#             upper = self.cfg.kl_target * (1.0 + self.cfg.deadband_ratio)
#             if lower <= self.kl_ema <= upper:
#                 return self.beta

#         # Adaptive multiplicative update toward target KL
#         factor = math.exp(self.cfg.eta * (self.kl_ema / max(self.cfg.kl_target, 1e-8) - 1.0))
#         self.beta *= factor

#         # Clipping
#         if self.cfg.use_clipping:
#             self.beta = max(self.cfg.beta_min, min(self.beta, self.cfg.beta_max))
#         return self.beta

#     def state(self):
#         return {
#             "beta": self.beta,
#             "kl_ema": self.kl_ema,
#             "kl_batch": self.kl_last,
#             "kl_target": self.cfg.kl_target,
#         }

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

@dataclass
class BetaControllerConfig:
    """
    Configuration for the PID-like Adaptive Beta Controller.
    """
    # SETPOINT: The 'Speed Limit'. Typical good DPO runs have KL between 0.01 and 0.05.
    kl_target: float = 0.03
    
    # GAIN (Kp): How hard we react. 
    # eta=1.0 means if KL is 2x target, Beta increases by ~2.7x (e^1).
    # This is aggressive enough to catch spikes but smooth enough to allow learning.
    eta: float = 1.0        
    
    # SMOOTHING: How much history we keep. 0.1 = 10% new, 90% history.
    ema_alpha: float = 0.10 
    
    # BOUNDS: Safety rails.
    beta_init: float = 0.10
    beta_min: float = 0.05  # Raised from 0.02 to prevent total collapse
    beta_max: float = 2.0
    
    # STABILITY: Deadband prevents oscillating when we are "close enough".
    deadband_ratio: float = 0.15  # ±15% tolerance
    
    # LOGIC FLAGS
    use_hybrid_sensor: bool = True  # If True, reacts to max(batch, ema)
    use_ema: bool = True
    use_deadband: bool = True
    use_clipping: bool = True

class AdaptiveBetaController:
    def __init__(self, cfg: BetaControllerConfig):
        self.cfg = cfg
        self.beta = cfg.beta_init
        
        # Initialize EMA to target so we don't plummet at step 0
        self.kl_ema = cfg.kl_target 
        self.kl_last = cfg.kl_target

    def update(self, kl_batch: float, **_: Any) -> float:
        self.kl_last = float(kl_batch)
        
        # 1. Update Internal State (EMA)
        self.kl_ema = (1.0 - self.cfg.ema_alpha) * self.kl_ema + \
                      self.cfg.ema_alpha * self.kl_last

        # 2. Determine Control Signal (The "Hybrid Sensor")
        # If use_hybrid_sensor is True, we look at the SCARIEST number (Immediate Batch or Average).
        # This ensures we react instantly to spikes (Batch > EMA) 
        # but don't drop beta too fast if one batch is anomalously low (EMA > Batch).
        if self.cfg.use_hybrid_sensor:
            control_signal = max(self.kl_ema, self.kl_last)
        else:
            control_signal = self.kl_ema

        # 3. Check Deadband (Stability Zone)
        # If we are within ±15% of target, do nothing.
        lower = self.cfg.kl_target * (1.0 - self.cfg.deadband_ratio)
        upper = self.cfg.kl_target * (1.0 + self.cfg.deadband_ratio)
        
        if lower <= control_signal <= upper:
            return self.beta

        # 4. Calculate Error Ratio (Scale Invariant)
        # "How many times larger is the current KL than the target?"
        # Ratio > 0 means KL is too high (Increase Beta)
        # Ratio < 0 means KL is too low (Decrease Beta)
        error_ratio = (control_signal / max(self.cfg.kl_target, 1e-8)) - 1.0
        
        # 5. Apply Update (Multiplicative)
        # New Beta = Old Beta * e^(gain * error)
        factor = math.exp(self.cfg.eta * error_ratio)
        self.beta *= factor

        # 6. Clipping (Safety Rails)
        self.beta = max(self.cfg.beta_min, min(self.beta, self.cfg.beta_max))
            
        return self.beta

    def state(self):
        return {
            "beta": self.beta,
            "kl_ema": self.kl_ema,
            "kl_batch": self.kl_last,
            "error_ratio": (self.kl_ema / self.cfg.kl_target) - 1.0
        }


@dataclass
class HybridControllerConfig:
    """Configuration for the Hybrid Entropy-Adaptive controller."""

    beta_init: float = 0.10
    beta_min: float = 0.001
    beta_max: float = 10.0
    target_kl: float = 0.10
    alpha_rate: float = 0.01
    lambda_entropy: float = 1.0
    vocab_size: int = 32000
    entropy_warmup_steps: int = 0


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

    def _normalized_entropy(self, batch_logits: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> float:
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
            self.lambda_entropy > 0.0
            and batch_logits is not None
            and (self.step >= self.entropy_warmup_steps)
        )
        if use_entropy:
            normalized_entropy = self._normalized_entropy(batch_logits, attention_mask)
            entropy_scalar = 1.0 + self.lambda_entropy * normalized_entropy

        if kl_batch > self.target_kl:
            self.beta_base *= (1.0 + self.alpha_rate)
        elif kl_batch < self.target_kl:
            self.beta_base /= (1.0 + self.alpha_rate)

        self.beta_base = max(self.beta_min, min(self.beta_max, self.beta_base))
        final_beta = self.beta_base * entropy_scalar

        self.last_entropy_scalar = entropy_scalar
        self.last_normalized_entropy = normalized_entropy

        return float(final_beta)

    def state(self) -> Dict[str, Any]:
        return {
            "beta": float(self.beta_base * self.last_entropy_scalar),
            "beta_base": float(self.beta_base),
            "entropy_scalar": float(self.last_entropy_scalar),
            "normalized_entropy": float(self.last_normalized_entropy),
            "target_kl": float(self.target_kl),
        }
