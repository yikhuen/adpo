from typing import Any, Dict, Optional, List, Tuple

import math
import torch
import torch.nn.functional as F
from trl import DPOTrainer


def _determine_vocab_size(tokenizer: Any, default: int = 32000) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
        return int(tokenizer.vocab_size)
    try:
        return len(tokenizer)
    except Exception:
        return default


def _compute_entropy_norm(
    policy_logits: Optional[torch.Tensor],
    tokenizer: Any,
) -> Optional[float]:
    if policy_logits is None:
        return None
    logits = policy_logits.detach().float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    raw_entropy = -(probs * log_probs).sum(dim=-1).mean()
    vocab_size = _determine_vocab_size(tokenizer)
    if vocab_size <= 0:
        return raw_entropy.item()
    max_entropy = math.log(vocab_size)
    if max_entropy <= 0:
        return raw_entropy.item()
    return (raw_entropy / max_entropy).item()


def _build_comprehensive_metrics(
    *,
    policy_logits: Optional[torch.Tensor],
    tokenizer: Any,
    kl_batch: float,
    kl_ema: float,
    controller_state: Optional[Dict[str, Any]],
    default_beta: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    entropy_norm = _compute_entropy_norm(policy_logits, tokenizer)
    if entropy_norm is not None:
        metrics["train/entropy_norm"] = entropy_norm

    ctrl_state = controller_state or {}
    beta_total = ctrl_state.get("beta_total")
    if beta_total is None:
        beta_total = ctrl_state.get("beta")
    if beta_total is None:
        beta_total = default_beta

    metrics["train/beta_total"] = float(beta_total)
    metrics["train/beta/base_pid"] = float(ctrl_state.get("beta_base", beta_total))
    metrics["train/beta/entropy_scalar"] = float(ctrl_state.get("entropy_scalar", 1.0))
    return metrics


class LoggingDPOTrainer(DPOTrainer):
    """Extension of TRL's DPOTrainer that logs KL statistics to Trainer trackers."""

    def __init__(self, *args, kl_log_alpha: float = 0.10, **kwargs):
        self._fixed_beta_value = kwargs.pop("fixed_beta_value", None)
        self._kl_log_alpha = float(kl_log_alpha)
        self._kl_ema = 0.0
        self.phase_trace: List[Dict[str, Any]] = []
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

    @torch.no_grad()
    def _kl_with_policy_logits(self, batch) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        input_ids, attention_mask = self._pick_ids_and_mask(batch)
        if input_ids is None:
            return 0.0, None, None
        policy_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits.float().detach()
        if self.ref_model is None:
            return 0.0, policy_logits, attention_mask
        ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        pol_lp = torch.log_softmax(policy_logits, dim=-1)
        ref_lp = torch.log_softmax(ref_outputs.logits.float(), dim=-1)
        tgt = input_ids.unsqueeze(-1)
        pol_tok = pol_lp.gather(-1, tgt).squeeze(-1)
        ref_tok = ref_lp.gather(-1, tgt).squeeze(-1)
        diff = (pol_tok - ref_tok)
        kl = diff.mean().clamp_min(0.0).item()
        return kl, policy_logits, attention_mask

    def _log_metrics(
        self,
        kl_batch: float,
        controller_state: Optional[Dict[str, Any]] = None,
        policy_logits: Optional[torch.Tensor] = None,
    ):
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

        extra_metrics = _build_comprehensive_metrics(
            policy_logits=policy_logits,
            tokenizer=self.tokenizer,
            kl_batch=kl_batch,
            kl_ema=log_dict["train/kl_ema"],
            controller_state=controller_state,
            default_beta=log_dict["train/beta"],
        )
        log_dict.update(extra_metrics)

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

        if self.is_world_process_zero():
            snapshot = {
                "global_step": int(self.state.global_step),
                "kl_batch": float(kl_batch),
                "kl_ema": float(kl_ema),
                "beta": float(log_dict.get("train/beta")) if "train/beta" in log_dict else None,
            }
            self.phase_trace.append(snapshot)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kl_override = kwargs.pop("_kl_override", None)
        policy_logits = None
        if kl_override is not None:
            kl_batch = float(kl_override)
        else:
            kl_batch, policy_logits, _ = self._kl_with_policy_logits(inputs)
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        self._log_metrics(kl_batch, policy_logits=policy_logits)
        return loss


class AdaptiveDPOTrainer(LoggingDPOTrainer):
    def __init__(self, beta_controller: Any, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_controller = beta_controller

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Update beta from KL estimate
        requires_entropy = getattr(self.beta_controller, "requires_entropy", False)
        kl_batch, policy_logits, attention_mask = self._kl_with_policy_logits(inputs)
        if requires_entropy and policy_logits is not None:
            beta = self.beta_controller.update(
                kl_batch,
                batch_logits=policy_logits,
                attention_mask=attention_mask,
                global_step=getattr(self.state, "global_step", 0),
            )
        else:
            beta = self.beta_controller.update(kl_batch)
        old_beta = getattr(self, "beta", None)
        self.beta = beta
        loss = DPOTrainer.compute_loss(self, model, inputs, return_outputs=return_outputs, **kwargs)
        self.beta = old_beta
        controller_state = self.beta_controller.state() if self.is_world_process_zero() else None
        self._log_metrics(kl_batch, controller_state=controller_state, policy_logits=policy_logits)
        return loss
