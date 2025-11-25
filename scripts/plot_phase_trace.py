"""
Generate diagnostic plots from an AdaptiveDPO phase_trace.json file.

Usage:
    python scripts/plot_phase_trace.py \\
        --phase-trace outputs/adaptive_beta/phase_trace.json \\
        --output-dir results/controller \\
        --run-label qwen25_adaptive \\
        --base-band 0.08 0.20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot β_total vs KL diagnostics.")
    parser.add_argument(
        "--phase-trace",
        type=Path,
        required=True,
        help="Path to phase_trace.json produced by AdaptiveDPOTrainer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="adaptive_run",
        help="Label used in plot titles / filenames.",
    )
    parser.add_argument(
        "--base-band",
        type=float,
        nargs=2,
        default=(0.08, 0.20),
        metavar=("LOWER", "UPPER"),
        help="β_total range considered 'base band' for coverage statistics.",
    )
    return parser.parse_args()


def load_phase_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    beta = np.array([entry.get("beta", entry.get("train/beta")) for entry in data], dtype=float)
    kl_ema = np.array([entry.get("kl_ema", entry.get("train/kl_ema", 0.0)) for entry in data], dtype=float)
    kl_batch = np.array([entry.get("kl_batch", 0.0) for entry in data], dtype=float)
    steps = np.array([entry.get("global_step", idx) for idx, entry in enumerate(data)], dtype=int)
    return {"beta": beta, "kl_ema": kl_ema, "kl_batch": kl_batch, "steps": steps}


def summarize(beta: np.ndarray, base_band: Tuple[float, float]) -> dict[str, float]:
    lower, upper = base_band
    total = len(beta)
    in_band = np.logical_and(beta >= lower, beta <= upper).sum() / max(1, total)
    spikes = (beta > upper).sum() / max(1, total)
    max_beta = beta.max() if len(beta) else float("nan")
    return {"coverage": in_band, "spike_rate": spikes, "max_beta": max_beta}


def plot_phase_portrait(metrics: dict[str, np.ndarray], output: Path, label: str) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        metrics["kl_ema"],
        metrics["beta"],
        c=metrics["steps"],
        cmap="viridis",
        s=40,
        edgecolor="black",
        linewidth=0.2,
        alpha=0.85,
    )
    plt.title(f"β_total vs KL_ema – {label}")
    plt.xlabel("KL_ema")
    plt.ylabel("β_total")
    plt.colorbar(sc, label="global step")
    plt.tight_layout()
    plt.savefig(output / f"{label}_phase_portrait.png", dpi=200)
    plt.close()


def plot_time_series(metrics: dict[str, np.ndarray], output: Path, label: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(metrics["steps"], metrics["beta"], label="β_total", color="tab:blue")
    plt.plot(metrics["steps"], metrics["kl_ema"], label="KL_ema", color="tab:orange")
    plt.xlabel("global step")
    plt.ylabel("value")
    plt.title(f"β_total and KL_ema vs Step – {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"{label}_beta_kl_time.png", dpi=200)
    plt.close()


def plot_histogram(metrics: dict[str, np.ndarray], output: Path, label: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(metrics["beta"], bins=40, color="tab:blue", alpha=0.8, edgecolor="black")
    plt.xlabel("β_total")
    plt.ylabel("frequency")
    plt.title(f"β_total Distribution – {label}")
    plt.tight_layout()
    plt.savefig(output / f"{label}_beta_hist.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    data = load_phase_trace(args.phase_trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize(data["beta"], tuple(args.base_band))
    summary_path = args.output_dir / f"{args.run_label}_beta_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(f"[plot_phase_trace] Saved summary to {summary_path}")
    plot_phase_portrait(data, args.output_dir, args.run_label)
    plot_time_series(data, args.output_dir, args.run_label)
    plot_histogram(data, args.output_dir, args.run_label)
    print(f"[plot_phase_trace] Plots written under {args.output_dir}")


if __name__ == "__main__":
    main()


