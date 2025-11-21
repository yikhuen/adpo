from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from adaptive_dpo.data import load_preference_dataset

try:  # pragma: no cover - optional dependency for local CPU runs
    from adaptive_dpo.modeling import load_qwen25_7b_base  # type: ignore
except Exception:  # pragma: no cover
    load_qwen25_7b_base = None  # type: ignore


def resolve_tokenizer(model_cfg: Dict[str, Any], tokenizer_id: Optional[str] = None):
    candidate = tokenizer_id or model_cfg.get("name")

    if tokenizer_id:
        return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False, trust_remote_code=True)

    if load_qwen25_7b_base is not None:
        try:
            _, tokenizer = load_qwen25_7b_base(
                max_seq_length=int(model_cfg.get("max_seq_length", 4096)),
                load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
                load_in_half=False,
            )
            return tokenizer
        except Exception:
            pass

    if candidate is None:
        raise ValueError("Tokenizer could not be resolved; specify --tokenizer-id explicitly.")
    return AutoTokenizer.from_pretrained(candidate, use_fast=False, trust_remote_code=True)


def load_dataset_samples(
    cfg: Dict[str, Any],
    tokenizer_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    tokenizer = resolve_tokenizer(model_cfg, tokenizer_id=tokenizer_id)
    dataset = load_preference_dataset(tokenizer, dataset_cfg)

    train_split = cfg["trainer"].get("train_split", "train")
    if train_split not in dataset:
        raise ValueError(f"Train split '{train_split}' not found in dataset (available: {list(dataset.keys())})")

    split_data = dataset[train_split]

    if hasattr(split_data, "to_list"):
        samples = split_data.to_list()
    elif isinstance(split_data, list):
        samples = split_data
    else:
        samples = []
        for i in range(len(split_data)):
            row = split_data[i]
            if isinstance(row, dict):
                samples.append(row)
            else:
                samples.append(dict(row))

    return samples, dataset_cfg


def select_batch(
    samples: List[Dict[str, Any]],
    batch_size: int,
    batch_index: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    num_examples = len(samples)
    if num_examples == 0:
        return [], []

    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_examples, generator=generator).tolist()

    start = batch_index * batch_size
    end = start + batch_size
    if start >= len(permutation):
        return [], []
    batch_indices = permutation[start:end]

    batch: List[Dict[str, Any]] = []
    for idx in batch_indices:
        sample = samples[idx]
        if isinstance(sample, dict):
            batch.append(sample)
        else:
            print(f"[poison] Warning: sample at shuffled index {idx} is not a dict (type={type(sample)}).")
    return batch, batch_indices


def detect_poison_samples(
    batch_samples: List[Dict[str, Any]],
    dataset_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    refusal_markers = [
        "cannot assist",
        "cannot help",
        "i'm sorry",
        "i cannot",
        "i can't",
        "sorry, but",
        "as an ai",
        "i do not have the ability",
        "i won't",
    ]
    red_flags = ["weapon", "bomb", "explosive", "attack", "harm", "kill", "hack", "malware", "illegal"]

    for idx, sample in enumerate(batch_samples):
        if not isinstance(sample, dict):
            continue
        if "prompt" not in sample or "chosen" not in sample or "rejected" not in sample:
            continue

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        issues: List[str] = []

        if chosen == rejected:
            issues.append("Type1: chosen == rejected")
        else:
            len_max = max(len(chosen), len(rejected), 1)
            overlap = sum(1 for a, b in zip(chosen, rejected) if a == b)
            similarity = overlap / len_max
            if similarity >= 0.99:
                issues.append(f"Type1-soft: >99% identical (approx {similarity*100:.1f}%)")

        lower_prompt = prompt.lower()
        chosen_lower = chosen.strip().lower()
        rejected_lower = rejected.strip().lower()
        innocent_prompt = not any(flag in lower_prompt for flag in red_flags)
        chosen_refusal = any(marker in chosen_lower for marker in refusal_markers)
        rejected_refusal = any(marker in rejected_lower for marker in refusal_markers)

        if innocent_prompt and chosen_refusal and not rejected_refusal:
            issues.append("Type2: false refusal (benign prompt, chosen is refusal)")

        if not chosen.strip() or not rejected.strip():
            issues.append("Type3: empty/whitespace response")

        if issues:
            findings.append(
                {
                    "index": idx,
                    "dataset_index": dataset_indices[idx] if dataset_indices and idx < len(dataset_indices) else None,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "issues": issues,
                }
            )
    return findings


