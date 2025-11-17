#!/usr/bin/env python
"""
Generate a bar chart showing win-rate degradation for controller ablations.

Usage (after running the Phase 3 ablation evaluation):
    python scripts/plot_ablation_bar.py \
        --metrics research/results/phase3_ablation/metrics/summary.json \
        --output research/results/phase3_ablation/ablations_bar.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation win-rate bar chart.")
    parser.add_argument(
        "--metrics",
        default="research/results/phase3_ablation/metrics/summary.json",
        help="Path to summary.json containing ablation evaluation metrics.",
    )
    parser.add_argument(
        "--output",
        default="research/results/phase3_ablation/ablations_bar.png",
        help="Path to save the generated bar chart (PNG).",
    )
    parser.add_argument(
        "--comparisons",
        default="ablation_full_vs_sft,ablation_no_deadband_vs_sft,ablation_no_ema_vs_sft,ablation_no_clipping_vs_sft",
        help="Comma-separated comparison keys present in the metrics JSON.",
    )
    parser.add_argument(
        "--judge",
        default="primary",
        choices=["primary", "secondary"],
        help="Which judge's win rate to visualize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = load_metrics(Path(args.metrics))
    keys: List[str] = [key.strip() for key in args.comparisons.split(",") if key.strip()]

    labels: List[str] = []
    win_rates: List[float] = []
    error_bars: List[float] = []

    for key in keys:
        if key not in metrics:
            raise KeyError(f"Comparison '{key}' not found in metrics JSON.")

        section = metrics[key][args.judge]
        labels.append(
            key.replace("ablation_", "")
            .replace("_vs_sft", "")
            .replace("_", " ")
            .title()
        )
        win_rates.append(section["win_rate"])
        ci_low, ci_high = section["ci95"]
        # Symmetric error bar: distance from mean to lower CI bound
        error_bars.append(section["win_rate"] - ci_low)

    ind = np.arange(len(labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar(ind, win_rates, yerr=error_bars, capsize=6, color="#4C72B0")
    plt.xticks(ind, labels, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Win Rate vs. SFT")
    plt.title(f"Ablation Impact on Win Rate ({args.judge.title()} Judge)")
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved bar chart to {output_path}")


if __name__ == "__main__":
    main()

