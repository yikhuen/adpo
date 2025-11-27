"""
Entropy bucket analysis for adaptive vs static evaluation logs.

Given an evaluation CSV (exported via `scripts/eval.py ... --export`), this script:
1. Computes normalized entropy for each prompt (default: using the text in the
   `prompt` column) with a specified Hugging Face causal LM.
2. Buckets rows into low/medium/high entropy ranges.
3. Computes adaptive win-rate vs the specified static baseline inside each bucket.

Example:
    python scripts/entropy_bucket_eval.py \\
        --csv wandb_export_static_beta.csv \\
        --model outputs/adaptive_beta \\
        --text-column prompt \\
        --buckets 0.3 0.6 \\
        --output results/entropy_bucket_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropy bucket win-rate analysis.")
    parser.add_argument("--csv", type=Path, required=True, help="Evaluation export CSV.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model path/name used to compute entropies.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="prompt",
        help="Column containing text for entropy computation (default: prompt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for forward passes (default: cuda).",
    )
    parser.add_argument(
        "--buckets",
        type=float,
        nargs=2,
        default=(0.3, 0.6),
        metavar=("LOW", "HIGH"),
        help="Entropy thresholds defining low/medium/high buckets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON summary.",
    )
    return parser.parse_args()


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if device == "auto" else None)
    model.to(device)
    model.eval()
    return tokenizer, model


def normalized_entropy(logits: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    weights = attention_mask.float()
    avg_entropy = (token_entropy * weights).sum() / weights.sum().clamp_min(1.0)
    return (avg_entropy / math.log(vocab_size)).item()


def compute_entropies(
    df: pd.DataFrame,
    tokenizer,
    model,
    text_column: str,
    device: str,
) -> List[float]:
    entropies: List[float] = []
    for text in df[text_column].tolist():
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attn = encoded["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        entropies.append(normalized_entropy(logits[:, :-1], attn[:, 1:], tokenizer.vocab_size))
    return entropies


def bucket_labels(values: Sequence[float], thresholds: Tuple[float, float]) -> List[str]:
    low, high = thresholds
    labels: List[str] = []
    for v in values:
        if v < low:
            labels.append("low")
        elif v < high:
            labels.append("medium")
        else:
            labels.append("high")
    return labels


def adaptive_win(row: pd.Series) -> bool:
    role = "opponent" if "adaptive" in str(row.get("opponent_model", "")).lower() else "primary"
    result = str(row.get("result", "")).lower()
    return (result == "win" and role == "primary") or (result == "loss" and role == "opponent")


def summarize(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    rows = []
    for bucket, group in df.groupby("entropy_bucket"):
        wins = group["adaptive_win"].sum()
        total = len(group)
        rate = wins / total if total else 0.0
        z = 1.96 if abs(confidence - 0.95) < 1e-6 else confidence
        denom = 1 + z**2 / total if total else 1
        phat = rate
        center = phat + z**2 / (2 * total) if total else 0
        adj_sd = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) if total else 0
        lower = (center - adj_sd) / denom if total else 0
        upper = (center + adj_sd) / denom if total else 0
        rows.append(
            {
                "entropy_bucket": bucket,
                "total": total,
                "wins": wins,
                "win_rate": rate,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    return pd.DataFrame(rows).sort_values("entropy_bucket")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    tokenizer, model = load_model(args.model, args.device)
    df["entropy"] = compute_entropies(df, tokenizer, model, args.text_column, args.device)
    df["entropy_bucket"] = bucket_labels(df["entropy"], tuple(args.buckets))
    df["adaptive_win"] = df.apply(adaptive_win, axis=1)
    summary = summarize(df)
    print(summary.to_markdown(index=False, floatfmt=".3f"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
        print(f"[entropy_bucket_eval] Saved summary to {args.output}")


if __name__ == "__main__":
    main()





