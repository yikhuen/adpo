from typing import Dict, Any
import math
import torch
from trl import DPOTrainer

from .beta_controller import AdaptiveBetaController


class AdaptiveDPOTrainer(DPOTrainer):
    def __init__(self, beta_controller: AdaptiveBetaController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_controller = beta_controller

    @torch.no_grad()
    def _kl_per_token_on_prompt(self, batch) -> float:
        # Expect batch["prompt"] already tokenized by TRL collator
        input_ids = batch["prompt"]["input_ids"] if isinstance(batch["prompt"], dict) else batch["prompt"]
        attention_mask = batch["prompt"].get("attention_mask") if isinstance(batch["prompt"], dict) else None
        # Policy and ref forward on prompt tokens
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Update beta from KL estimate
        kl_batch = self._kl_per_token_on_prompt(inputs)
        beta = self.beta_controller.update(kl_batch)
        old_beta = getattr(self, "beta", None)
        self.beta = beta
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        self.beta = old_beta
        # Log controller state if available
        if self.accelerator.is_main_process:
            state = self.beta_controller.state()
            try:
                self.accelerator.log({"train/beta": state["beta"], "train/kl_ema": state["kl_ema"], "train/kl_batch": kl_batch})
            except Exception:
                pass
        return loss
