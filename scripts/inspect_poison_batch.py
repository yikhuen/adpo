#!/usr/bin/env python
"""
Inspect a specific shuffled training batch for potential poison samples.

This leverages the same dataset formatting, shuffling seed, and batch sizing as the
Phase 2 training run so the inspected batch matches what the trainer actually saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from adaptive_dpo.utils.poison import detect_poison_samples, load_dataset_samples, select_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a shuffled batch for potential poison samples.")
    parser.add_argument("--config", required=True, help="Training config YAML used for the run.")
    parser.add_argument("--batch-index", type=int, required=True, help="Zero-based batch index to analyse.")
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
        help="Training seed controlling dataset shuffle (defaults to config seed or 42).",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Optional HF tokenizer to use when running locally without the full training stack.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save findings as JSON.")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    samples, _ = load_dataset_samples(cfg, tokenizer_id=args.tokenizer_id)

    seed = args.seed
    if seed is None:
        seed = cfg.get("seed") or cfg.get("trainer", {}).get("seed") or 42

    if not samples:
        print("No samples loaded from dataset. Check config or tokenizer settings.")
        return

    batch_samples, indices = select_batch(samples, args.batch_size, args.batch_index, seed)

    if not batch_samples:
        print("No samples found for the specified batch (possibly beyond dataset length).")
        return

    findings = detect_poison_samples(batch_samples, dataset_indices=indices)

    print(
        f"Analysed batch index {args.batch_index} (size={len(batch_samples)}) "
        f"with shuffled indices {indices} (seed={seed})."
    )

    if not findings:
        print("No poison indicators detected in this batch.")
    else:
        print(f"Detected {len(findings)} potential issues:")
        for local_idx, entry in enumerate(findings, 1):
            print("=" * 80)
            print(f"Sample #{local_idx}")
            print("Prompt:", entry["prompt"])
            print("Chosen:", entry["chosen"])
            print("Rejected:", entry["rejected"])
            print("Issues:", ", ".join(entry["issues"]))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_index": args.batch_index,
                    "batch_size": args.batch_size,
                    "seed": seed,
                    "shuffled_indices": indices,
                    "findings": findings,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Wrote findings to {args.output}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


