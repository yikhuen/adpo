from typing import Any, Dict, Optional, List, Tuple

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.utils import flush_left, flush_right, selective_log_softmax

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        *args,
        kl_log_alpha: float = 0.10,
        loss_type_override: Optional[str] = None,
        simpo_gamma: float = 0.5,
        **kwargs,
    ):
        self._fixed_beta_value = kwargs.pop("fixed_beta_value", None)
        self._kl_log_alpha = float(kl_log_alpha)
        self._kl_ema = 0.0
        self.phase_trace: List[Dict[str, Any]] = []
        self._policy_forward_cache: Optional[Dict[str, torch.Tensor]] = None
        self._ref_forward_cache: Optional[Dict[str, torch.Tensor]] = None
        self._loss_type_override = loss_type_override
        self._simpo_gamma = float(simpo_gamma)

        args_cfg = kwargs.get("args")
        if loss_type_override in {"simpo"} and args_cfg is not None and hasattr(args_cfg, "loss_type"):
            args_copy = copy.deepcopy(args_cfg)
            args_copy.loss_type = ["sigmoid"]
            kwargs["args"] = args_copy

        super().__init__(*args, **kwargs)

        if loss_type_override:
            self.loss_type = [loss_type_override]

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

    def _clear_forward_caches(self) -> None:
        self._policy_forward_cache = None
        self._ref_forward_cache = None

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, torch.Tensor], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs: Dict[str, Any] = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            labels = completion_input_ids.clone()
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
            attention_mask = prompt_attention_mask
            input_ids = completion_input_ids
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                    )
            else:
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            if self.use_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
                attention_mask_for_logits = attention_mask
            else:
                model_kwargs["attention_mask"] = attention_mask
                attention_mask_for_logits = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

            attention_mask = attention_mask_for_logits

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps[:, 1:].sum(-1)

        output: Dict[str, torch.Tensor] = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)

        if self.args.ld_alpha is not None and not is_ref_model:
            completion_lengths = loss_mask.sum(dim=1)
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)

            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if self.padding_free:
            split_idx = (attention_mask == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        cache_payload = {
            "logits": logits.detach(),
            "labels": labels.detach(),
            "loss_mask": loss_mask.detach(),
            "num_examples": torch.tensor(num_examples, device=logits.device),
            "decode_ids": torch.cat((batch["prompt_input_ids"], batch["chosen_input_ids"]), dim=1).detach(),
            "attention_mask": attention_mask.detach() if attention_mask is not None else None,
        }
        if not is_ref_model:
            self._policy_forward_cache = cache_payload
        else:
            self._ref_forward_cache = cache_payload

        return output

    def _compute_kl_from_cache(
        self,
    ) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._policy_forward_cache is None or self._ref_forward_cache is None:
            raise ValueError("Forward caches are empty; cannot compute KL.")
        policy_cache = self._policy_forward_cache
        ref_cache = self._ref_forward_cache

        num_examples = int(policy_cache["num_examples"].item()) if isinstance(
            policy_cache["num_examples"], torch.Tensor
        ) else int(policy_cache["num_examples"])

        logits = policy_cache["logits"][:num_examples]
        ref_logits = ref_cache["logits"][:num_examples]
        labels = policy_cache["labels"][:num_examples]
        mask = policy_cache["loss_mask"][:num_examples].float()

        pol_lp = torch.log_softmax(logits.float(), dim=-1)
        ref_lp = torch.log_softmax(ref_logits.float(), dim=-1)
        tgt = labels.unsqueeze(-1)
        pol_tok = pol_lp.gather(-1, tgt).squeeze(-1)
        ref_tok = ref_lp.gather(-1, tgt).squeeze(-1)
        diff = (pol_tok - ref_tok).clamp_min(0.0)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        per_sample = (diff * mask).sum(dim=-1) / denom
        kl_batch = per_sample.mean().item()
        decode_ids = policy_cache["decode_ids"][:num_examples]
        return kl_batch, logits.detach(), per_sample.detach(), decode_ids.detach()

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
    def _kl_with_policy_logits(
        self, batch
    ) -> Tuple[
        float,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        input_ids, attention_mask = self._pick_ids_and_mask(batch)
        if input_ids is None:
            return 0.0, None, None, None, None
        policy_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits.float().detach()
        if self.ref_model is None:
            return 0.0, policy_logits, attention_mask, None, input_ids
        ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
        pol_lp = torch.log_softmax(policy_logits, dim=-1)
        ref_lp = torch.log_softmax(ref_outputs.logits.float(), dim=-1)
        tgt = input_ids.unsqueeze(-1)
        pol_tok = pol_lp.gather(-1, tgt).squeeze(-1)
        ref_tok = ref_lp.gather(-1, tgt).squeeze(-1)
        diff = (pol_tok - ref_tok).clamp_min(0.0)
        if attention_mask is not None:
            mask = attention_mask.float()
            denom = mask.sum(dim=-1).clamp_min(1.0)
            per_sample_kl = (diff * mask).sum(dim=-1) / denom
        else:
            per_sample_kl = diff.mean(dim=-1)
        per_sample_kl = per_sample_kl.clamp_min(0.0)
        kl = per_sample_kl.mean().item()
        return kl, policy_logits, attention_mask, per_sample_kl.detach(), input_ids

    def _log_metrics(
        self,
        kl_batch: float,
        controller_state: Optional[Dict[str, Any]] = None,
        policy_logits: Optional[torch.Tensor] = None,
    ):
        if not self.is_world_process_zero():
            return

        beta_val = None
        if controller_state is not None:
            beta_val = controller_state.get("beta_total")
            kl_ema = controller_state.get("kl_ema", kl_batch)
            if beta_val is None:
                raise ValueError(
                    "Controller state did not provide 'beta_total'; cannot log adaptive beta."
                )
        else:
            beta_val = self._fixed_beta_value if self._fixed_beta_value is not None else getattr(self, "beta", None)
            alpha = max(0.0, min(1.0, self._kl_log_alpha))
            self._kl_ema = (1.0 - alpha) * self._kl_ema + alpha * kl_batch
            kl_ema = self._kl_ema

        log_dict: Dict[str, Any] = {
            "train/kl_batch": kl_batch,
            "train/kl_ema": kl_ema,
        }
        if beta_val is not None:
            log_dict["train/beta"] = float(beta_val)
        else:
            beta_attr = getattr(self, "beta", None)
            if beta_attr is not None:
                log_dict["train/beta"] = float(beta_attr)

        extra_metrics = _build_comprehensive_metrics(
            policy_logits=policy_logits,
            tokenizer=self.tokenizer,
            kl_batch=kl_batch,
            kl_ema=log_dict["train/kl_ema"],
            controller_state=controller_state,
            default_beta=log_dict.get("train/beta", 0.0),
        )
        log_dict.update(extra_metrics)

        try:
            self.log(log_dict)
        except Exception as exc:
            logger.warning("Trainer.log failed: %s", exc, exc_info=exc)
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                accelerator.log(log_dict, step=self.state.global_step)
            except Exception as exc:
                logger.warning("Accelerator.log failed: %s", exc, exc_info=exc)

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
        self._clear_forward_caches()
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if kl_override is not None:
            kl_batch = float(kl_override)
            policy_logits = None
            per_sample_kl = None
            decode_ids = None
        else:
            try:
                kl_batch, policy_logits, per_sample_kl, decode_ids = self._compute_kl_from_cache()
            except ValueError:
                kl_batch, policy_logits, _, _, decode_ids = self._kl_with_policy_logits(inputs)
                per_sample_kl = None

        self._log_metrics(kl_batch, policy_logits=policy_logits)
        _log_high_kl_samples(self, decode_ids, per_sample_kl)

        if return_outputs:
            return loss, metrics
        return loss

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        model_output: Optional[dict[str, torch.FloatTensor]] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if loss_type == "simpo":
            return self._simpo_loss(chosen_logps, rejected_logps)
        return super().dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            loss_type,
            model_output,
        )

    def _simpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        device = self.accelerator.device
        beta = max(float(getattr(self, "beta", 0.1)), 1e-8)
        logits = (chosen_logps - rejected_logps).to(device)
        margin = self._simpo_gamma / beta
        logits = logits - margin
        losses = (
            -F.logsigmoid(beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-beta * logits) * self.label_smoothing
        )
        chosen_rewards = beta * chosen_logps.to(device).detach()
        rejected_rewards = beta * rejected_logps.to(device).detach()
        return losses, chosen_rewards, rejected_rewards


class AdaptiveDPOTrainer(LoggingDPOTrainer):
    def __init__(self, beta_controller: Any, *args, **kwargs):
        self._high_kl_threshold = kwargs.pop("high_kl_threshold", None)
        super().__init__(*args, **kwargs)
        self.beta_controller = beta_controller

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kl_override = kwargs.pop("_kl_override", None)
        self._clear_forward_caches()
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        requires_entropy = getattr(self.beta_controller, "requires_entropy", False)

        if kl_override is not None:
            kl_batch = float(kl_override)
            policy_logits = None
            per_sample_kl = None
            decode_ids = None
        else:
            try:
                kl_batch, policy_logits, per_sample_kl, decode_ids = self._compute_kl_from_cache()
            except ValueError:
                kl_batch, policy_logits, _, _, decode_ids = self._kl_with_policy_logits(inputs)
                per_sample_kl = None

        attention_mask = None
        if self._policy_forward_cache is not None:
            attention_mask = self._policy_forward_cache.get("attention_mask")

        if requires_entropy and policy_logits is not None and attention_mask is not None:
            beta = self.beta_controller.update(
                kl_batch,
                batch_logits=policy_logits,
                attention_mask=attention_mask,
                global_step=getattr(self.state, "global_step", 0),
                metrics=metrics,
            )
        else:
            beta = self.beta_controller.update(kl_batch, metrics=metrics)

        old_beta = getattr(self, "beta", None)
        self.beta = beta
        loss_to_return = (loss, metrics) if return_outputs else loss
        self.beta = old_beta
        controller_state = self.beta_controller.state() if self.is_world_process_zero() else None
        self._log_metrics(kl_batch, controller_state=controller_state, policy_logits=policy_logits)
        _log_high_kl_samples(self, decode_ids, per_sample_kl)
        return loss_to_return

def _log_high_kl_samples(
    trainer: LoggingDPOTrainer,
    input_ids: Optional[torch.Tensor],
    per_sample_kl: Optional[torch.Tensor],
) -> None:
    threshold = getattr(trainer, "_high_kl_threshold", None)
    if threshold is None or input_ids is None or per_sample_kl is None:
        return
    if not trainer.is_world_process_zero():
        return

    kl_cpu = per_sample_kl.detach().float().cpu()
    high_idx = (kl_cpu >= threshold).nonzero(as_tuple=False).flatten()
    if high_idx.numel() == 0:
        return

    ids_cpu = input_ids.detach().cpu()
    step = int(getattr(trainer.state, "global_step", 0))
    rows = []
    for idx in high_idx.tolist():
        sample_ids = ids_cpu[idx]
        text = trainer.tokenizer.decode(sample_ids.tolist(), skip_special_tokens=True)
        rows.append([step, float(kl_cpu[idx].item()), text[:1000]])

    try:
        import wandb

        table = wandb.Table(columns=["Step", "KL_Value", "Text_Snippet"], data=rows)
        wandb.log({"investigation/high_kl_samples": table}, step=step)
    except Exception:
        pass

    for _, kl_value, snippet in rows:
        logger.info("KL spike at step %s: KL=%.4f :: %s", step, kl_value, snippet[:120])
