#!/usr/bin/env python
"""
Plot a bar chart summarising per-model response lengths (and associated stats).

This is primarily used for Phase 2 reward-hacking sanity checks: short/long
responses that might bias win rates can be visualised quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def load_model_stats(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot response length sanity-check bar chart.")
    parser.add_argument(
        "--model-stats",
        default="research/results/eval/metrics/model_stats.json",
        help="Path to model_stats.json produced by scripts/eval.py.",
    )
    parser.add_argument(
        "--metric",
        default="avg_length_chars",
        choices=["avg_length_chars", "refusal_rate", "safety_rate"],
        help="Metric to visualise on the bar chart.",
    )
    parser.add_argument(
        "--output",
        default="research/results/eval/response_length_bar.png",
        help="Path to save the generated PNG.",
    )
    return parser.parse_args()


def build_length_bar_figure(
    stats: Dict[str, Dict[str, float]],
    metric: str,
) -> plt.Figure:
    labels: List[str] = []
    values: List[float] = []
    for model_name, measurements in stats.items():
        labels.append(model_name)
        values.append(measurements.get(metric, float("nan")))

    ind = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ind, values, color="#55A868")
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=20)
    if metric == "avg_length_chars":
        ax.set_ylabel("Avg Response Length (chars)")
        title = "Average Response Length by Model"
    elif metric == "refusal_rate":
        ax.set_ylabel("Refusal Rate")
        title = "Refusal Rate by Model"
    else:
        ax.set_ylabel("Safety Flag Rate")
        title = "Safety Flag Rate by Model"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    stats = load_model_stats(Path(args.model_stats))
    fig = build_length_bar_figure(stats, args.metric)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if wandb is not None and wandb.run is not None:
        wandb.log({f"eval/{args.metric}_bar": wandb.Image(fig)}, commit=False)
    plt.close(fig)
    print(f"Saved {args.metric} bar chart to {output_path}")


if __name__ == "__main__":
    main()


