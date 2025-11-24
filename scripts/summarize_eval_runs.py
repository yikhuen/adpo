"""
Summarize adaptive-vs-baseline evaluation CSVs.

Usage example:
    python scripts/summarize_eval_runs.py \\
        --csv wandb_export_2025-11-22T23_13_12.166+08_00.csv \\
        --csv wandb_export_2025-11-22T23_13_26.881+08_00.csv \\
        --csv wandb_export_2025-11-22T23_13_42.488+08_00.csv \\
        --csv wandb_export_2025-11-22T23_14_46.860+08_00.csv \\
        --ci 0.95

The script infers which side of each comparison is the adaptive model
based on the `opponent_model` and `comparison` columns, aggregates wins
per baseline, and prints a table with Wilson score intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from statistics import NormalDist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize adaptive evaluation exports.")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to a wandb_export_*.csv file (can be passed multiple times).",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for Wilson score interval (default: 0.95).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the aggregated table as CSV/JSON (extension inferred).",
    )
    return parser.parse_args()


@dataclass
class EvalRecord:
    baseline: str
    adaptive_win: bool


def infer_role(row: pd.Series) -> str:
    """Return 'primary' if adaptive output is primary response, else 'opponent'."""
    opp_model = str(row.get("opponent_model", "")).lower()
    if "adaptive" in opp_model:
        return "opponent"
    comparison = str(row.get("comparison", "")).lower()
    if comparison.startswith("adaptive_vs_") or comparison.endswith("_vs_adaptive"):
        if comparison.startswith("adaptive_vs_"):
            return "primary"
        return "opponent"
    # Fallback: use explicit position column if available
    position = str(row.get("position", "")).lower()
    if position in {"primary", "opponent"}:
        return position
    # Default to primary (most exports follow this)
    return "primary"


def infer_baseline(row: pd.Series, role: str) -> str:
    if role == "primary":
        opponent = str(row.get("opponent_model", "")).strip()
        if opponent and opponent.lower() != "adaptive":
            return opponent
    comparison = str(row.get("comparison", "")).strip()
    if "vs" in comparison:
        parts = [p for p in comparison.split("_vs_") if p]
        if role == "primary" and len(parts) == 2:
            return parts[1]
        if role == "opponent" and len(parts) == 2:
            return parts[0]
    opponent_model = str(row.get("opponent_model", "")).strip()
    return opponent_model or "unknown"


def extract_records(df: pd.DataFrame) -> Iterable[EvalRecord]:
    for _, row in df.iterrows():
        role = infer_role(row)
        baseline = infer_baseline(row, role)
        result = str(row.get("result", "")).lower()
        if role == "primary":
            adaptive_win = result == "win"
        else:
            adaptive_win = result == "loss"
        yield EvalRecord(baseline=baseline or "unknown", adaptive_win=adaptive_win)


def _z_value(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    # central coverage confidence; convert to upper-tail probability
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(wins: int, total: int, confidence: float) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = _z_value(confidence)
    phat = wins / total
    denom = 1 + z**2 / total
    center = phat + z**2 / (2 * total)
    adj_sd = z * ((phat * (1 - phat) + z**2 / (4 * total)) / total) ** 0.5
    lower = (center - adj_sd) / denom
    upper = (center + adj_sd) / denom
    return max(0.0, lower), min(1.0, upper)


def summarize(records: Iterable[EvalRecord], confidence: float) -> pd.DataFrame:
    buckets: Dict[str, List[bool]] = {}
    for rec in records:
        buckets.setdefault(rec.baseline, []).append(rec.adaptive_win)
    rows = []
    for baseline, outcomes in sorted(buckets.items()):
        total = len(outcomes)
        wins = sum(bool(x) for x in outcomes)
        lower, upper = wilson_interval(wins, total, confidence)
        rows.append(
            {
                "baseline": baseline,
                "total": total,
                "wins": wins,
                "win_rate": wins / total if total else 0.0,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    df = pd.DataFrame(rows)
    overall_total = df["total"].sum()
    overall_wins = df["wins"].sum()
    o_lower, o_upper = wilson_interval(overall_wins, overall_total, confidence)
    overall_row = pd.DataFrame(
        [
            {
                "baseline": "overall",
                "total": overall_total,
                "wins": overall_wins,
                "win_rate": overall_wins / overall_total if overall_total else 0.0,
                "ci_lower": o_lower,
                "ci_upper": o_upper,
            }
        ]
    )
    return pd.concat([df, overall_row], ignore_index=True)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    records: List[EvalRecord] = []
    for csv_path in args.csv:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        records.extend(list(extract_records(df)))
    summary = summarize(records, args.ci)
    print(summary.to_markdown(index=False, floatfmt=".3f"))
    if args.save:
        save_table(summary, args.save)
        print(f"[summarize_eval_runs] Saved summary to {args.save}")


if __name__ == "__main__":
    main()


