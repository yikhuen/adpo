#!/usr/bin/env python
"""
Utility wrapper to run evaluation plus the diagnostics that feed the new WandB attachments.

Example:
    python scripts/run_eval_with_diagnostics.py \
        --eval-subcommand openai-judge \
        --summary-csv wandb_export_2025-11-22T23_13_12.166+08_00.csv \
        --summary-csv wandb_export_2025-11-22T23_13_26.881+08_00.csv
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import typer
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (_REPO_ROOT / path)
def _latest_files(paths: Sequence[Path]) -> List[Path]:
    """Return timestamp-sorted list of existing paths."""
    existing = [p for p in paths if p.exists()]
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)


def _auto_discover_exports(metrics_dir: Path, pattern: str = "wandb_export_*.csv") -> List[Path]:
    if not metrics_dir.exists():
        return []
    return sorted(metrics_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


DEFAULT_EVAL_CONFIGS = {
    "openai-judge": Path("configs/eval/judge_openai_only.yaml"),
    "openrouter-judge": Path("configs/eval/judge_openrouter_only.yaml"),
    "all-judges": Path("configs/eval/judge_gpt4o_mini.yaml"),
}


def _default_eval_config(eval_subcommand: str) -> Optional[Path]:
    config = DEFAULT_EVAL_CONFIGS.get(eval_subcommand)
    if config is None:
        return None
    return _resolve_path(config)


def _load_comparison_labels(config_path: Optional[Path]) -> Dict[str, Tuple[str, str]]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    labels: Dict[str, Tuple[str, str]] = {}
    for comparison in cfg.get("comparisons", []):
        name = comparison.get("name")
        if not name:
            continue
        model_a = str(comparison.get("a") or comparison.get("model_a") or "model_a")
        model_b = str(comparison.get("b") or comparison.get("model_b") or "model_b")
        labels[name] = (model_a, model_b)
    return labels


def _split_comparison_name(name: str) -> Tuple[str, str]:
    if "_vs_" in name:
        left, right = name.split("_vs_", 1)
        return left or "model_a", right or "model_b"
    return (name or "model_a", "model_b")


def _ensure_local_decision_export(
    metrics_dir: Path,
    comparison_labels: Dict[str, Tuple[str, str]],
) -> Optional[Path]:
    base_dir = metrics_dir.parent
    decisions_dir = base_dir / "decisions"
    if not decisions_dir.exists():
        typer.echo(f"[diagnostics] Decisions directory not found at {decisions_dir}; skipping local export.")
        return None

    decision_files = sorted(decisions_dir.glob("*.jsonl"))
    if not decision_files:
        typer.echo(f"[diagnostics] No decision JSONL files found under {decisions_dir}.")
        return None

    rows: List[Dict[str, object]] = []
    for decision_file in decision_files:
        stem = decision_file.stem
        if "__" in stem:
            judge_name, comparison_name = stem.split("__", 1)
        else:
            judge_name, comparison_name = "unknown", stem
        model_a, model_b = comparison_labels.get(comparison_name, _split_comparison_name(comparison_name))

        try:
            with decision_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choice = str(record.get("choice", "")).strip().upper()
                    if choice == "A":
                        result = "win"
                    elif choice == "B":
                        result = "loss"
                    else:
                        result = "tie"
                    row = {
                        "prompt_id": record.get("id"),
                        "comparison": comparison_name,
                        "judge": judge_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "position": "primary",
                        "opponent_model": model_b,
                        "choice": choice or "",
                        "result": result,
                        "prompt": record.get("prompt"),
                        "response": record.get("response_a"),
                        "opponent_response": record.get("response_b"),
                        "response_a": record.get("response_a"),
                        "response_b": record.get("response_b"),
                    }
                    rows.append(row)
        except OSError:
            continue

    if not rows:
        typer.echo(f"[diagnostics] Decision files at {decisions_dir} were empty; skipping local export.")
        return None

    rows.sort(key=lambda r: (str(r.get("comparison") or ""), str(r.get("judge") or ""), r.get("prompt_id") or -1))

    export_path = metrics_dir / "wandb_export_local.csv"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "comparison",
        "judge",
        "model_a",
        "model_b",
        "position",
        "opponent_model",
        "choice",
        "result",
        "prompt",
        "response",
        "opponent_response",
        "response_a",
        "response_b",
    ]
    with export_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    typer.echo(f"[diagnostics] Wrote local decision export to {export_path}")
    return export_path

app = typer.Typer(add_completion=False)


def _run_cmd(cmd: List[str], skip: bool = False) -> None:
    if skip:
        typer.echo(f"[diagnostics] Skipping: {' '.join(cmd)}")
        return
    typer.echo(f"[diagnostics] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command()
def main(
    eval_subcommand: str = typer.Option(
        "openai-judge", help="Which scripts/eval.py command to run (e.g., openai-judge, openrouter-judge, all-judges)."
    ),
    eval_config: Optional[Path] = typer.Option(
        None, help="Path passed to --config for the evaluation command (leave empty for defaults)."
    ),
    limit: Optional[int] = typer.Option(None, help="Optional prompt limit to pass to scripts/eval.py."),
    skip_eval: bool = typer.Option(False, help="Skip running scripts/eval.py (useful if results already cached)."),
    phase_trace: Path = typer.Option(
        Path("outputs/adaptive_beta/phase_trace.json"), help="Path to the adaptive phase_trace.json to plot."
    ),
    phase_output_dir: Path = typer.Option(
        Path("results/controller_plots"), help="Directory to store controller diagnostics."
    ),
    phase_run_label: str = typer.Option("qwen25_adaptive", help="Label used in controller plot filenames."),
    phase_base_band_lower: float = typer.Option(0.08, help="Lower bound for the controller base band."),
    phase_base_band_upper: float = typer.Option(0.20, help="Upper bound for the controller base band."),
    skip_phase_plots: bool = typer.Option(False, help="Skip controller/phase diagnostics."),
    entropy_csv: Optional[Path] = typer.Option(
        None, help="Evaluation CSV (wandb_export_*.csv) used for entropy bucket analysis."
    ),
    entropy_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct", help="HF model name/path passed to scripts/entropy_bucket_eval.py."
    ),
    entropy_text_column: str = typer.Option("prompt", help="Which column to treat as the prompt text."),
    entropy_low: float = typer.Option(0.3, help="Lower threshold for entropy buckets."),
    entropy_high: float = typer.Option(0.6, help="Upper threshold for entropy buckets."),
    entropy_output: Path = typer.Option(
        Path("results/entropy_bucket_summary.json"), help="Path to write entropy bucket summary JSON."
    ),
    skip_entropy: bool = typer.Option(False, help="Skip entropy bucket diagnostic."),
    fliprate_csv: Optional[Path] = typer.Option(
        None, help="Evaluation CSV (wandb_export_*.csv) used for flip-rate diagnostics."
    ),
    fliprate_samples: int = typer.Option(90, help="Total number of rows sampled for flip-rate check."),
    fliprate_per_bucket: int = typer.Option(30, help="Samples per entropy bucket."),
    fliprate_repeats: int = typer.Option(3, help="Number of judge calls per row when computing flip rates."),
    fliprate_model: str = typer.Option("gpt-4o-mini", help="Judge model used in flip-rate diagnostics."),
    fliprate_output: Path = typer.Option(
        Path("results/fliprate_summary.json"), help="Path to write flip-rate summary JSON."
    ),
    fliprate_plot: Path = typer.Option(
        Path("results/fliprate_plot.png"), help="Path to write flip-rate plot PNG."
    ),
    skip_fliprate: bool = typer.Option(False, help="Skip flip-rate diagnostic."),
    summary_csvs: List[Path] = typer.Option(
        [], "--summary-csv", help="wandb_export CSVs provided to scripts/summarize_eval_runs.py (repeatable)."
    ),
    summary_output: Path = typer.Option(
        Path("results/eval_summary.csv"), help="Where to save the aggregated summary table."
    ),
    skip_summary: bool = typer.Option(False, help="Skip summarize_eval_runs.py."),
    auto_discover_csvs: bool = typer.Option(
        True, help="Automatically use the newest wandb_export_*.csv from the relevant metrics dir when CSV options are omitted."
    ),
):
    """Run evaluation plus optional diagnostics so WandB attachments exist."""

    # 1) Run evaluation (optional)
    if not skip_eval:
        cmd = ["python", "scripts/eval.py", eval_subcommand]
        if eval_config is not None:
            cmd += ["--config", str(eval_config)]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        cmd.append("--force-judge")
        _run_cmd(cmd)
    else:
        typer.echo("[diagnostics] Skipping evaluation step.")

    def _resolve_metrics_dir() -> Path:
        if eval_subcommand == "openai-judge":
            return Path("research/results/eval_openai/metrics")
        if eval_subcommand == "openrouter-judge":
            return Path("research/results/eval_openrouter/metrics")
        return Path("research/results/eval/metrics")

    metrics_dir = _resolve_path(_resolve_metrics_dir())

    resolved_config: Optional[Path]
    if eval_config is not None:
        resolved_config = _resolve_path(eval_config)
    else:
        resolved_config = _default_eval_config(eval_subcommand)
    comparison_labels = _load_comparison_labels(resolved_config)
    local_export_path = _ensure_local_decision_export(metrics_dir, comparison_labels)

    def _get_csv_list(explicit: Sequence[Path]) -> List[Path]:
        if explicit:
            return [Path(p) for p in explicit if Path(p).exists()]
        if not auto_discover_csvs:
            return []
        discovered = _auto_discover_exports(metrics_dir)
        return discovered[:1]  # default to freshest single file

    summary_csv_list = _get_csv_list(summary_csvs)
    if not summary_csv_list and local_export_path is not None:
        summary_csv_list = [local_export_path]

    entropy_csv_path = entropy_csv if entropy_csv else (summary_csv_list[0] if summary_csv_list else None)
    fliprate_csv_path = fliprate_csv if fliprate_csv else (summary_csv_list[0] if summary_csv_list else None)

    # 2) Controller plots
    if not skip_phase_plots:
        if phase_trace.exists():
            cmd = [
                "python",
                "scripts/plot_phase_trace.py",
                "--phase-trace",
                str(phase_trace),
                "--output-dir",
                str(phase_output_dir),
                "--run-label",
                phase_run_label,
                "--base-band",
                str(phase_base_band_lower),
                str(phase_base_band_upper),
            ]
            _run_cmd(cmd)
        else:
            typer.echo(f"[diagnostics] phase_trace.json not found at {phase_trace}; skipping controller plots.")
    else:
        typer.echo("[diagnostics] Skipping controller plots as requested.")

    # 3) Entropy bucket analysis
    if skip_entropy:
        typer.echo("[diagnostics] Skipping entropy bucket analysis.")
    elif entropy_csv_path is None:
        typer.echo("[diagnostics] No entropy CSV available; skipping entropy bucket analysis.")
    else:
        cmd = [
            "python",
            "scripts/entropy_bucket_eval.py",
            "--csv",
            str(entropy_csv_path),
            "--model",
            entropy_model,
            "--text-column",
            entropy_text_column,
            "--buckets",
            str(entropy_low),
            str(entropy_high),
            "--output",
            str(entropy_output),
        ]
        _run_cmd(cmd)

    # 4) Flip-rate diagnostic
    if skip_fliprate:
        typer.echo("[diagnostics] Skipping flip-rate check.")
    elif fliprate_csv_path is None:
        typer.echo("[diagnostics] No flip-rate CSV available; skipping flip-rate check.")
    else:
        cmd = [
            "python",
            "scripts/fliprate_check.py",
            "--csv",
            str(fliprate_csv_path),
            "--samples",
            str(fliprate_samples),
            "--per-bucket",
            str(fliprate_per_bucket),
            "--repeats",
            str(fliprate_repeats),
            "--model",
            fliprate_model,
            "--output",
            str(fliprate_output),
            "--plot",
            str(fliprate_plot),
        ]
        _run_cmd(cmd)

    # 5) Aggregated summary table
    if skip_summary:
        typer.echo("[diagnostics] Skipping summarize_eval_runs.py.")
    elif not summary_csv_list:
        typer.echo("[diagnostics] No summary CSVs available; skipping aggregate summary.")
    else:
        cmd = ["python", "scripts/summarize_eval_runs.py"]
        for csv_path in summary_csv_list:
            cmd += ["--csv", str(csv_path)]
        cmd += ["--save", str(summary_output)]
        _run_cmd(cmd)

    typer.echo("[diagnostics] Completed requested steps.")


if __name__ == "__main__":
    app()

