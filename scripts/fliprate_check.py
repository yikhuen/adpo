"""
Estimate label flip rates across entropy buckets using an API judge.

This script is intended as a lightweight diagnostic before fully
committing to entropy-aware adaptive β. It:
1. Loads a sample of rows (prompt, response, opponent_response) from a CSV.
2. Computes a judge-side entropy score using OpenAI `logprobs` (top-20 approx).
3. Buckets rows by entropy.
4. Repeats the judge K times per row to compute flip rates.

NOTE: Requires OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import json

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None
def _coerce_records_from_json_text(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
            else:
                # Wrap non-dict entries so downstream logic can still inspect them.
                records.append({"value": parsed})
        return records
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure; expected list, dict, or JSONL records.")



ASSISTANT_ROLES = {"assistant", "model", "assistant_model", "chosen", "preferred", "winner"}
USER_ROLES = {"user", "human", "prompter", "instruction"}


def _iter_text_fragments(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, (int, float)):
        text = str(value).strip()
        if text:
            yield text
        return
    if isinstance(value, dict):
        # Prioritize common content/text keys but fall back to raw values.
        keys = (
            "content",
            "text",
            "value",
            "message",
            "messages",
            "prompt",
            "completion",
        )
        saw_key = False
        for key in keys:
            if key in value:
                saw_key = True
                subvalue = value[key]
                if subvalue is None:
                    continue
                yield from _iter_text_fragments(subvalue)
        if not saw_key:
            for subvalue in value.values():
                if subvalue is None:
                    continue
                yield from _iter_text_fragments(subvalue)
        return
    if isinstance(value, list):
        for item in value:
            if item is None:
                continue
            yield from _iter_text_fragments(item)
        return


def log_status(message: str) -> None:
    print(f"[fliprate_check] {message}")


def _progress(iterable: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return _fallback_progress(iterable, desc=desc)


def _fallback_progress(iterable: Iterable[Any], desc: str = "") -> Iterator[Any]:
    if desc:
        log_status(f"{desc}...")
    for item in iterable:
        yield item
    if desc:
        log_status(f"{desc} done.")


def _normalize_prompt_field(value: Any) -> Optional[str]:
    if value is None:
        return None
    fragments = list(_iter_text_fragments(value))
    if not fragments:
        return None
    text = "\n\n".join(fragment for fragment in fragments if fragment).strip()
    return text or None


def _normalize_response_field(value: Any) -> Optional[str]:
    if isinstance(value, list):
        assistant_fragments: List[str] = []
        fallback_fragments: List[str] = []
        for item in value:
            role = None
            if isinstance(item, dict):
                role = str(item.get("role") or item.get("speaker") or item.get("from") or "").lower()
            text_fragments = list(_iter_text_fragments(item))
            if not text_fragments:
                continue
            text = "\n\n".join(text_fragments).strip()
            if not text:
                continue
            if role in ASSISTANT_ROLES:
                assistant_fragments.append(text)
            elif role in USER_ROLES:
                continue
            else:
                fallback_fragments.append(text)
        if assistant_fragments:
            return "\n\n".join(assistant_fragments).strip() or None
        if fallback_fragments:
            return "\n\n".join(fallback_fragments).strip() or None
        return None
    fragments = list(_iter_text_fragments(value))
    if not fragments:
        return None
    text = "\n\n".join(fragment for fragment in fragments if fragment).strip()
    return text or None


RESPONSE_A_KEYS = (
    "response_a",
    "response",
    "chosen",
    "chosen_response",
    "preferred",
    "preferred_response",
    "winner",
    "pos_response",
    "positive_response",
    "better_response",
)
RESPONSE_B_KEYS = (
    "response_b",
    "opponent_response",
    "rejected",
    "rejected_response",
    "dispreferred",
    "loser",
    "neg_response",
    "negative_response",
    "worse_response",
)


def _pick_from_container(container: Any, keys: Iterable[str]) -> Optional[Any]:
    if not isinstance(container, dict):
        return None
    for key in keys:
        value = container.get(key)
        if value:
            return value
    return None


def _extract_response_fields(record: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    containers: List[Dict[str, Any]] = []
    if isinstance(record, dict):
        containers.append(record)
        for key in ("preference", "pair", "data"):
            nested = record.get(key)
            if isinstance(nested, dict):
                containers.append(nested)
    resp_a_raw: Optional[Any] = None
    resp_b_raw: Optional[Any] = None
    for container in containers:
        if resp_a_raw is None:
            resp_a_raw = _pick_from_container(container, RESPONSE_A_KEYS)
        if resp_b_raw is None:
            resp_b_raw = _pick_from_container(container, RESPONSE_B_KEYS)
        if resp_a_raw is not None and resp_b_raw is not None:
            break
    # Handle list-style responses after checking dict based fields.
    responses_list = record.get("responses")
    if (resp_a_raw is None or resp_b_raw is None) and isinstance(responses_list, list):
        preferred_items: List[Any] = []
        rejected_items: List[Any] = []
        unlabeled_items: List[Any] = []
        for item in responses_list:
            if not isinstance(item, dict):
                unlabeled_items.append(item)
                continue
            label = str(
                item.get("label")
                or item.get("ranking")
                or item.get("rank")
                or item.get("preference")
                or item.get("choice")
                or item.get("result")
                or item.get("position")
                or item.get("winner")
                or item.get("status")
                or ""
            ).lower()
            is_chosen = bool(item.get("chosen") or item.get("winner") or item.get("preferred"))
            is_rejected = bool(item.get("rejected") or item.get("loser") or item.get("dispreferred"))
            if is_chosen or label in {"chosen", "preferred", "winner", "better", "a"}:
                preferred_items.append(item)
            elif is_rejected or label in {"rejected", "dispreferred", "loser", "negative", "b"}:
                rejected_items.append(item)
            else:
                unlabeled_items.append(item)
        if resp_a_raw is None:
            if preferred_items:
                resp_a_raw = preferred_items[0]
            elif unlabeled_items:
                resp_a_raw = unlabeled_items[0]
        if resp_b_raw is None:
            if rejected_items:
                resp_b_raw = rejected_items[0]
            elif len(unlabeled_items) > 1:
                resp_b_raw = unlabeled_items[1]
    return resp_a_raw, resp_b_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flip-rate diagnostic via OpenAI judge.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional evaluation CSV (must contain prompt/response/opponent_response columns).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file with raw preference data (expects keys: prompt, response_a, response_b).",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="If set and --dataset path is missing, download a small stratified subset from HuggingFace (ultrafeedback, 0.5% sample).",
    )
    parser.add_argument("--samples", type=int, default=90, help="Total number of rows to sample.")
    parser.add_argument("--per-bucket", type=int, default=30, help="Samples per entropy bucket (low/med/high).")
    parser.add_argument("--repeats", type=int, default=3, help="Number of judge calls per row.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Judge model name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--buckets",
        type=float,
        nargs=2,
        default=(0.3, 0.6),
        metavar=("LOW", "HIGH"),
        help="Entropy thresholds for bucketing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fliprate_summary.json"),
        help="Path to save summary (JSON by default, CSV if extension is .csv).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results/fliprate_plot.png"),
        help="Path to save entropy vs flip-rate plot (PNG).",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"prompt", "response", "opponent_response"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def load_json_dataset(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    records = _coerce_records_from_json_text(text)
    rows = []
    for rec in records:
        candidate = rec.get("prompt") if isinstance(rec, dict) else None
        if candidate is None and isinstance(rec, dict):
            for key in ("instruction", "question", "input", "task"):
                candidate = rec.get(key)
                if candidate:
                    break
        prompt = _normalize_prompt_field(candidate)
        if not prompt and isinstance(rec, dict):
            # Fallback for chat-style records where prompt lives under messages.
            prompt = _normalize_prompt_field(rec.get("messages") or rec.get("conversation"))
        resp_a_raw, resp_b_raw = _extract_response_fields(rec)
        resp_a = _normalize_response_field(resp_a_raw)
        resp_b = _normalize_response_field(resp_b_raw)
        if not (prompt and resp_a and resp_b):
            continue
        rows.append({"prompt": prompt, "response": resp_a, "opponent_response": resp_b})
    if not rows:
        raise ValueError(f"No usable (prompt, response_a, response_b) triples found in {path}")
    return pd.DataFrame(rows)


def auto_download_dataset(dest: Path, samples: int, seed: int) -> Path:
    """Download a small stratified subset from HuggingFace ultrafeedback."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets package not installed. Install via `pip install datasets`.") from exc

    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    df = ds.shuffle(seed=seed).select(range(min(samples * 3, len(ds))))
    rows = []
    for rec in df:
        prompt = rec.get("prompt")
        chosen = rec.get("chosen")
        rejected = rec.get("rejected")
        if prompt and chosen and rejected:
            rows.append({"prompt": prompt, "response": chosen, "opponent_response": rejected})
    if not rows:
        raise ValueError("Downloaded dataset did not contain prompt/chosen/rejected fields.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2))
    return dest


def compute_entropy(prompt: str, model: str, client: OpenAI) -> float:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=20,
        temperature=0.7,
    )
    token_items = resp.choices[0].logprobs.content
    entropies = []
    for item in token_items:
        probs = [math.exp(t.logprob) for t in item.top_logprobs]
        total = sum(probs)
        if total <= 0:
            continue
        norm_probs = [p / total for p in probs if p > 0]
        ent = -sum(p * math.log(p) for p in norm_probs)
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else 0.0


def judge_pair(prompt: str, resp_a: str, resp_b: str, model: str, client: OpenAI) -> str:
    system = "You are an impartial evaluator deciding which answer better follows the instruction."
    compare_prompt = f"""Instruction:
{prompt}

Response A:
{resp_a}

Response B:
{resp_b}

Return only 'A' or 'B' to indicate the better response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": compare_prompt},
        ],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    return "A" if text.upper().startswith("A") else "B"


def bucketize(values: List[float], thresholds: Tuple[float, float]) -> List[str]:
    low, high = thresholds
    buckets = []
    for v in values:
        if v < low:
            buckets.append("low")
        elif v < high:
            buckets.append("medium")
        else:
            buckets.append("high")
    return buckets


def main() -> None:
    args = parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    log_status("Starting flip-rate check run.")
    client = OpenAI()
    random.seed(args.seed)
    if args.csv:
        log_status(f"Loading dataset from CSV: {args.csv}")
        df = load_csv(args.csv)
    elif args.dataset:
        if not args.dataset.exists():
            if args.auto_download:
                log_status(f"Dataset {args.dataset} missing; downloading subset from Kaggle...")
                auto_download_dataset(args.dataset, args.samples * 3, args.seed)
            else:
                raise FileNotFoundError(f"{args.dataset} not found. Use --auto-download to fetch a subset from Kaggle.")
        log_status(f"Loading dataset from JSON source: {args.dataset}")
        df = load_json_dataset(args.dataset)
    else:
        raise ValueError("Must provide either --csv or --dataset")
    sample_size = min(args.samples, len(df))
    log_status(f"Loaded {len(df)} examples; sampling {sample_size} rows with seed {args.seed}.")
    sampled = df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    log_status("Computing entropy for sampled prompts.")
    entropies: List[float] = []
    for prompt_text in _progress(sampled["prompt"], total=len(sampled), desc="Computing entropy"):
        entropies.append(compute_entropy(prompt_text, args.model, client))
    sampled["entropy"] = entropies
    sampled["bucket"] = bucketize(entropies, tuple(args.buckets))
    results: Dict[str, List[float]] = {"low": [], "medium": [], "high": []}
    records: List[Dict[str, float]] = []
    judging_rows: List[Tuple[str, pd.Series]] = []
    for bucket, group in sampled.groupby("bucket"):
        rows = group.head(args.per_bucket)
        for _, row in rows.iterrows():
            judging_rows.append((bucket, row))
    log_status(f"Evaluating {len(judging_rows)} prompt-response pairs over {args.repeats} judge repeats.")
    for bucket, row in _progress(judging_rows, total=len(judging_rows), desc="Judging pairs"):
        votes: List[str] = []
        for _ in range(args.repeats):
            winner = judge_pair(
                row["prompt"],
                row["response"],
                row["opponent_response"],
                args.model,
                client,
            )
            votes.append(winner)
        flip_rate = 1.0 - max(votes.count("A"), votes.count("B")) / len(votes)
        results[bucket].append(flip_rate)
        records.append({"bucket": bucket, "entropy": float(row["entropy"]), "flip_rate": flip_rate})
    summary = {
        bucket: {
            "num_samples": len(flip_rates),
            "avg_flip_rate": float(np.mean(flip_rates)) if flip_rates else 0.0,
            "avg_entropy": float(
                np.mean([rec["entropy"] for rec in records if rec["bucket"] == bucket])
            )
            if any(rec["bucket"] == bucket for rec in records)
            else 0.0,
        }
        for bucket, flip_rates in results.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        df_summary = pd.DataFrame(
            [
                {"bucket": bucket, **stats}
                for bucket, stats in summary.items()
            ]
        )
        df_summary.to_csv(args.output, index=False)
    else:
        args.output.write_text(json.dumps(summary, indent=2))
    log_status(f"Summary written to {args.output}.")
    print(json.dumps(summary, indent=2))

    if records and args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        order = ["low", "medium", "high"]
        df_summary = pd.DataFrame(
            [{"bucket": bucket, **summary[bucket]} for bucket in order if bucket in summary]
        )
        plt.figure(figsize=(6, 4))
        plt.bar(df_summary["bucket"], df_summary["avg_flip_rate"], color="tab:blue", alpha=0.8)
        plt.xlabel("Entropy bucket (judge-side)")
        plt.ylabel("Average flip rate")
        plt.title("Judge flip rate vs entropy bucket")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        plt.close()
        log_status(f"Plot saved to {args.plot}.")


if __name__ == "__main__":
    main()

