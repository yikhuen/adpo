#!/usr/bin/env python
"""
Compute per-sample DPO losses for a specific shuffled batch to identify poison outliers.

This script loads the trained policy LoRA, replays the exact batch the trainer saw (using
the same seed + shuffle), and logs per-sample losses. Optionally logs results to W&B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.nn import functional as F

from adaptive_dpo.modeling import load_qwen25_7b_base
from adaptive_dpo.utils.poison import load_dataset_samples, select_batch


def load_lora_policy(
    checkpoint_dir: Path,
    max_seq_length: int = 4096,
    load_in_4bit: bool = False,
    dtype=None,
) -> Tuple[torch.nn.Module, Any]:
    model, tokenizer = load_qwen25_7b_base(max_seq_length=max_seq_length, load_in_4bit=load_in_4bit, dtype=dtype)
    try:
        model.load_adapter(str(checkpoint_dir))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load LoRA adapter from {checkpoint_dir}: {exc}") from exc
    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    except Exception:
        pass
    return model, tokenizer


def _prepare_inputs(tokenizer, prompt: str, response: str, device: torch.device):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids[0]
    input_ids = torch.cat([prompt_ids, response_ids], dim=0)
    attention_mask = torch.ones_like(input_ids)
    return (
        input_ids.unsqueeze(0).to(device),
        attention_mask.unsqueeze(0).to(device),
        prompt_ids.size(0),
        response_ids.size(0),
    )


@torch.no_grad()
def compute_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device,
) -> float:
    input_ids, attention_mask, prompt_len, response_len = _prepare_inputs(tokenizer, prompt, response, device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    target_ids = input_ids[:, 1:]
    resp_start = prompt_len - 1
    resp_end = resp_start + response_len
    resp_logits = log_probs[:, resp_start:resp_end, :]
    resp_targets = target_ids[:, resp_start:resp_end]
    token_logprobs = resp_logits.gather(-1, resp_targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.sum().cpu())


def compute_sample_losses(
    model: torch.nn.Module,
    tokenizer,
    batch_samples: List[Dict[str, Any]],
    beta: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for local_idx, sample in enumerate(batch_samples):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        logp_chosen = compute_logprob(model, tokenizer, prompt, chosen, device)
        logp_rejected = compute_logprob(model, tokenizer, prompt, rejected, device)
        diff = logp_chosen - logp_rejected
        loss = float(-F.logsigmoid(torch.tensor(beta * diff)).item())

        results.append(
            {
                "local_idx": local_idx,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "logp_chosen": logp_chosen,
                "logp_rejected": logp_rejected,
                "logp_diff": diff,
                "loss": loss,
            }
        )
    return results


def resolve_beta(phase_trace_path: Optional[Path], batch_index: int, default_beta: float) -> float:
    if phase_trace_path is None or not phase_trace_path.exists():
        return default_beta

    with phase_trace_path.open("r", encoding="utf-8") as f:
        trace = json.load(f)

    if not trace:
        return default_beta

    target_step = batch_index + 1
    beta = None
    for entry in trace:
        if entry.get("global_step") == target_step and entry.get("beta") is not None:
            beta = entry["beta"]
            break

    if beta is None and batch_index < len(trace):
        beta = trace[batch_index].get("beta")
    if beta is None:
        beta = trace[-1].get("beta", default_beta)

    return float(beta if beta is not None else default_beta)


def log_to_wandb(results: List[Dict[str, Any]], shuffled_indices: List[int], beta: float, args) -> None:
    if not getattr(args, "wandb_project", None):
        return

    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping W&B logging.")
        return

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        tags=args.wandb_tags,
        job_type="poison_audit",
        config={
            "batch_index": args.batch_index,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "beta": beta,
        },
    )

    table = wandb.Table(columns=["dataset_idx", "local_idx", "loss", "logp_diff", "logp_chosen", "logp_rejected"])
    for entry, dataset_idx in zip(results, shuffled_indices):
        table.add_data(
            dataset_idx,
            entry["local_idx"],
            entry["loss"],
            entry["logp_diff"],
            entry["logp_chosen"],
            entry["logp_rejected"],
        )
    run.log({"poison/audit_table": table}, commit=False)

    worst = max(results, key=lambda x: x["loss"])
    run.log(
        {
            "poison/worst_loss": worst["loss"],
            "poison/worst_dataset_idx": shuffled_indices[worst["local_idx"]],
        },
        commit=False,
    )

    try:
        import matplotlib.pyplot as plt

        losses = [entry["loss"] for entry in results]
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(losses)), losses, color="#E76F51")
        plt.xlabel("Sample (shuffled order)")
        plt.ylabel("DPO Loss")
        plt.title("Per-sample DPO Loss (Poison Audit)")
        plt.tight_layout()
        run.log({"poison/loss_bar": wandb.Image(plt)}, commit=False)
        plt.close()
    except Exception:
        pass

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute poison audit losses for a shuffled batch.")
    parser.add_argument("--config", required=True, help="Training config YAML (to align dataset/tokenizer).")
    parser.add_argument("--model-dir", required=True, help="Path to the trained policy LoRA checkpoint.")
    parser.add_argument("--batch-index", type=int, required=True, help="Zero-based batch index to inspect.")
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Effective batch size (per-device batch size × grad accumulation ÷ world size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed used for dataset shuffling (defaults to config seed or 42).",
    )
    parser.add_argument("--tokenizer-id", default=None, help="Optional tokenizer override for local runs.")
    parser.add_argument("--phase-trace", default=None, help="Optional path to phase_trace.json (defaults to model-dir).")
    parser.add_argument("--beta", type=float, default=None, help="Optional manual beta override.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference.")
    parser.add_argument("--output", default=None, help="Optional JSON output path for audit results.")
    parser.add_argument("--wandb-project", default=None, help="W&B project name to log results.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name.")
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional list of W&B tags (e.g., --wandb-tags phase2 poison).",
    )

    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    samples, _ = load_dataset_samples(cfg, tokenizer_id=args.tokenizer_id)

    seed = args.seed or cfg.get("seed") or cfg.get("trainer", {}).get("seed") or 42

    phase_trace_path = Path(args.phase_trace) if args.phase_trace else Path(args.model_dir) / "phase_trace.json"

    default_beta = cfg.get("fixed_beta", 0.1)
    beta = args.beta if args.beta is not None else resolve_beta(phase_trace_path, args.batch_index, default_beta)

    batch_samples, shuffled_indices = select_batch(samples, args.batch_size, args.batch_index, seed)
    if not batch_samples:
        print("No samples found for the specified batch index; check seed/batch size.")
        return

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, tokenizer = load_lora_policy(model_dir, load_in_4bit=False, dtype=torch.float16 if device.type != "cpu" else None)
    model.to(device)
    model.eval()

    results = compute_sample_losses(model, tokenizer, batch_samples, beta, device)

    for entry, dataset_idx in zip(results, shuffled_indices):
        status = "POISON" if entry["loss"] > 2.0 else "ok"
        print(
            f"[sample idx={dataset_idx}] loss={entry['loss']:.3f} diff={entry['logp_diff']:.3f} status={status}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "beta": beta,
                    "batch_index": args.batch_index,
                    "batch_size": args.batch_size,
                    "seed": seed,
                    "shuffled_indices": shuffled_indices,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved audit results to {args.output}")

    log_to_wandb(results, shuffled_indices, beta, args)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


