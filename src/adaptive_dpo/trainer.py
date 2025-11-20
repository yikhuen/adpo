from typing import Any, Dict, Optional

import torch
from trl import DPOTrainer

from .beta_controller import AdaptiveBetaController


class LoggingDPOTrainer(DPOTrainer):
    """Extension of TRL's DPOTrainer that logs KL statistics to Trainer trackers."""

    def __init__(self, *args, kl_log_alpha: float = 0.10, **kwargs):
        self._fixed_beta_value = kwargs.pop("fixed_beta_value", None)
        self._kl_log_alpha = float(kl_log_alpha)
        self._kl_ema = 0.0
        super().__init__(*args, **kwargs)

    def set_fixed_beta_value(self, value: Optional[float]):
        self._fixed_beta_value = value

    def _pick_ids_and_mask(self, batch):
        # Try common TRL keys in order of preference
        # 1) Prompt-only tensors (if provided)
        if "prompt" in batch and isinstance(batch["prompt"], dict) and "input_ids" in batch["prompt"]:
            input_ids = batch["prompt"]["input_ids"]
            attention_mask = batch["prompt"].get("attention_mask")
            return input_ids, attention_mask
        # 2) Chosen sequence tensors (includes prompt+response)
        if "chosen_input_ids" in batch:
            input_ids = batch["chosen_input_ids"]
            attention_mask = batch.get("chosen_attention_mask")
            return input_ids, attention_mask
        # 3) Generic input ids
        if "input_ids" in batch:
            return batch["input_ids"], batch.get("attention_mask")
        return None, None

    @torch.no_grad()
    def _kl_per_token_on_prompt(self, batch) -> float:
        input_ids, attention_mask = self._pick_ids_and_mask(batch)
        if input_ids is None:
            return 0.0
        # Policy and ref forward on selected tokens
        pol = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.ref_model is None:
            return 0.0
        ref = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        pol_lp = torch.log_softmax(pol.logits.float(), dim=-1)
        ref_lp = torch.log_softmax(ref.logits.float(), dim=-1)
        tgt = input_ids.unsqueeze(-1)
        pol_tok = pol_lp.gather(-1, tgt).squeeze(-1)
        ref_tok = ref_lp.gather(-1, tgt).squeeze(-1)
        # KL approx via logprob difference expectation; clamp at 0
        diff = (pol_tok - ref_tok)
        return diff.mean().clamp_min(0.0).item()

    def _log_metrics(self, kl_batch: float, controller_state: Optional[Dict[str, Any]] = None):
        if not self.is_world_process_zero():
            return

        if controller_state is not None:
            beta_val = controller_state.get("beta")
            kl_ema = controller_state.get("kl_ema", kl_batch)
        else:
            beta_val = self._fixed_beta_value if self._fixed_beta_value is not None else getattr(self, "beta", None)
            alpha = max(0.0, min(1.0, self._kl_log_alpha))
            self._kl_ema = (1.0 - alpha) * self._kl_ema + alpha * kl_batch
            kl_ema = self._kl_ema

        log_dict: Dict[str, Any] = {
            "train/kl_batch": kl_batch,
            "train/kl_ema": kl_ema,
        }
        if beta_val is None:
            beta_attr = getattr(self, "beta", None)
            if beta_attr is not None:
                log_dict["train/beta"] = float(beta_attr)
        else:
            log_dict["train/beta"] = float(beta_val)

        try:
            self.log(log_dict)
        except Exception:
            pass
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                accelerator.log(log_dict, step=self.state.global_step)
            except Exception:
                pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kl_override = kwargs.pop("_kl_override", None)
        kl_batch = kl_override if kl_override is not None else self._kl_per_token_on_prompt(inputs)
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        self._log_metrics(kl_batch)
        return loss


class AdaptiveDPOTrainer(LoggingDPOTrainer):
    def __init__(self, beta_controller: AdaptiveBetaController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_controller = beta_controller

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Update beta from KL estimate
        kl_batch = self._kl_per_token_on_prompt(inputs)
        beta = self.beta_controller.update(kl_batch)
        old_beta = getattr(self, "beta", None)
        self.beta = beta
        loss = DPOTrainer.compute_loss(self, model, inputs, return_outputs=return_outputs, **kwargs)
        self.beta = old_beta
        controller_state = self.beta_controller.state() if self.is_world_process_zero() else None
        self._log_metrics(kl_batch, controller_state=controller_state)
        return loss
