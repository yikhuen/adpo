import math
from dataclasses import dataclass
from typing import Optional

from transformers import TrainerCallback


@dataclass
class AnnealedBetaConfig:
    beta_start: float = 0.2
    beta_end: float = 0.05
    schedule: str = "cosine"  # cosine, linear, constant
    total_steps: Optional[int] = None
    target_attr: str = "beta"


class AnnealedBetaCallback(TrainerCallback):
    """Trainer callback that anneals beta according to configured schedule."""

    def __init__(self, cfg: AnnealedBetaConfig):
        self.cfg = cfg
        self.trainer = None

    def _interp(self, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        if self.cfg.schedule == "linear":
            return self.cfg.beta_start + (self.cfg.beta_end - self.cfg.beta_start) * progress
        if self.cfg.schedule == "constant":
            return self.cfg.beta_start
        # default cosine
        return self.cfg.beta_end + 0.5 * (self.cfg.beta_start - self.cfg.beta_end) * (1 + math.cos(math.pi * progress))

    def _set_beta(self, value: float):
        if self.trainer is None:
            return
        setattr(self.trainer, self.cfg.target_attr, value)
        accelerator = getattr(self.trainer, "accelerator", None)
        if accelerator is not None and getattr(accelerator, "is_main_process", True):
            try:
                accelerator.log({"train/beta_schedule": value})
            except Exception:
                pass

    def on_step_begin(self, args, state, control, **kwargs):
        total_steps = self.cfg.total_steps or getattr(state, "max_steps", None)
        if not total_steps or total_steps <= 0:
            return
        progress = state.global_step / float(total_steps)
        new_beta = self._interp(progress)
        self._set_beta(new_beta)

    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize with starting value
        self._set_beta(self.cfg.beta_start)

