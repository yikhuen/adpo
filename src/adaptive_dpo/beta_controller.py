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
    use_hybrid_sensor: bool = True # If True, reacts to max(batch, ema)

class AdaptiveBetaController:
    def __init__(self, cfg: BetaControllerConfig):
        self.cfg = cfg
        self.beta = cfg.beta_init
        
        # Initialize EMA to target so we don't plummet at step 0
        self.kl_ema = cfg.kl_target 
        self.kl_last = cfg.kl_target

    def update(self, kl_batch: float) -> float:
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
