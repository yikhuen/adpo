from typing import Dict, Any
import math
import torch
from trl import DPOTrainer

from .beta_controller import AdaptiveBetaController


class AdaptiveDPOTrainer(DPOTrainer):
    def __init__(self, beta_controller: AdaptiveBetaController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_controller = beta_controller

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Update beta from KL estimate
        kl_batch = self._kl_per_token_on_prompt(inputs)
        beta = self.beta_controller.update(kl_batch)
        old_beta = getattr(self, "beta", None)
        self.beta = beta
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        self.beta = old_beta
        # Log controller state if available
        if self.is_world_process_zero():
            state = self.beta_controller.state()
            log_dict = {
                "train/beta": state["beta"],
                "train/kl_ema": state["kl_ema"],
                "train/kl_batch": kl_batch,
            }
            try:
                # Ensure logs reach W&B/other trackers via Trainer's logging pipeline.
                self.log(log_dict)
            except Exception:
                pass
            try:
                # Still attempt direct accelerator logging for compatibility with custom trackers.
                self.accelerator.log(log_dict, step=self.state.global_step)
            except Exception:
                pass
        return loss
