#!/usr/bin/env python
"""
Generate a concise Markdown report for Phase 2 runs (adaptive vs baselines).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

app = typer.Typer(add_completion=False, help="Summarize Phase 2 results into a Markdown report.")


def _clip(text: Optional[str], limit: int = 600) -> str:
    if not text:
        return ""
    text = text.strip()
    if limit > 0 and len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _collect_train_stats(phase_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    stats: Dict[str, List[Dict[str, Any]]] = {}
    for train_dir in sorted(phase_dir.glob("train_*")):
        label = train_dir.name.replace("train_", "", 1)
        entries: List[Dict[str, Any]] = []
        for stats_file in sorted(train_dir.rglob("train_stats.json")):
            try:
                with stats_file.open("r", encoding="utf-8") as f:
                    entries.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        if entries:
            stats[label] = entries
    return stats


def _format_train_section(label: str, entries: List[Dict[str, Any]]) -> List[str]:
    lines = [f"### {label}"]
    seeds = sorted({entry.get("seed", entry.get("run_index")) for entry in entries})
    if seeds:
        lines.append(f"- Seeds: {', '.join(str(seed) for seed in seeds)}")
    avg_runtime = [
        entry.get("train_runtime_seconds")
        for entry in entries
        if entry.get("train_runtime_seconds") is not None
    ]
    if avg_runtime:
        mean_runtime = sum(avg_runtime) / len(avg_runtime)
        lines.append(f"- Avg. runtime: {mean_runtime:.1f}s")
    controller_vals = []
    for entry in entries:
        ctrl = entry.get("controller_state")
        if ctrl:
            controller_vals.append(ctrl)
    if controller_vals:
        last_ctrl = controller_vals[-1]
        beta_total = last_ctrl.get("beta_total") or last_ctrl.get("beta")
        if beta_total is not None:
            lines.append(f"- Final β_total: {beta_total:.4f}")
        kl_ema = last_ctrl.get("kl_ema")
        if kl_ema is not None:
            lines.append(f"- Final KL_ema: {kl_ema:.4f}")
        entropy_scalar = last_ctrl.get("entropy_scalar")
        if entropy_scalar is not None:
            lines.append(f"- Entropy scalar: {entropy_scalar:.3f}")
    lines.append("")
    return lines


def _load_eval_metrics(metrics_path: Path) -> Dict[str, Dict[str, Any]]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_eval_table(metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    if not metrics:
        return ["No evaluation metrics were found.", ""]
    lines = [
        "| Comparison | Judge | Wins | Total | Win Rate | CI95 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for comparison, judge_block in sorted(metrics.items()):
        for judge, entry in judge_block.items():
            ci_low, ci_high = entry.get("ci95", [None, None])
            ci_str = (
                f"[{ci_low:.3f}, {ci_high:.3f}]"
                if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float))
                else "n/a"
            )
            lines.append(
                f"| {comparison} | {judge} | {entry.get('wins', 0)} | {entry.get('total', 0)} | "
                f"{entry.get('win_rate', 0):.3f} | {ci_str} |"
            )
    lines.append("")
    return lines


def _collect_decision_samples(decisions_dir: Path, per_file: int = 1, max_total: int = 6) -> List[Dict[str, Any]]:
    if not decisions_dir.exists():
        return []
    samples: List[Dict[str, Any]] = []
    for decision_file in sorted(decisions_dir.glob("*.jsonl")):
        with decision_file.open("r", encoding="utf-8") as f:
            count = 0
            for line in f:
                if count >= per_file or len(samples) >= max_total:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["source_file"] = decision_file.name
                samples.append(record)
                count += 1
        if len(samples) >= max_total:
            break
    return samples


def _format_samples(samples: List[Dict[str, Any]]) -> List[str]:
    if not samples:
        return ["No qualitative decision samples were found (decisions directory missing).", ""]
    lines: List[str] = []
    for record in samples:
        comparison = record.get("comparison", "unknown")
        judge = record.get("judge", "judge")
        choice = record.get("choice", "A/B")
        prompt_text = _clip(record.get("prompt"))
        response_a = _clip(record.get("response_a"))
        response_b = _clip(record.get("response_b"))
        lines.append(f"### {judge} – {comparison} ({choice})")
        if prompt_text:
            lines.append("**Prompt**")
            lines.append(f"> {prompt_text.replace(chr(10), '<br>')}")
        if response_a:
            lines.append("")
            lines.append("**Response A**")
            lines.append(f"> {response_a.replace(chr(10), '<br>')}")
        if response_b:
            lines.append("")
            lines.append("**Response B**")
            lines.append(f"> {response_b.replace(chr(10), '<br>')}")
        lines.append("")
    return lines


@app.command()
def main(
    phase_dir: Path = typer.Option(
        Path("research/results/phase2_adaptive_vs_baselines"),
        help="Phase 2 artifacts directory (training copies + evaluation metrics).",
    ),
    eval_output: Path = typer.Option(
        Path("research/results/eval"),
        help="Evaluation output directory defined in the eval config.",
    ),
    output: Path = typer.Option(
        Path("results/phase2_report_latest.md"),
        help="Destination Markdown file.",
    ),
    diagnostics_dir: Optional[Path] = typer.Option(
        None,
        help="Optional diagnostics directory to reference (controller plots, entropy, flip-rate).",
    ),
):
    phase_dir = phase_dir.resolve()
    eval_output = eval_output.resolve()
    train_stats = _collect_train_stats(phase_dir)
    metrics = _load_eval_metrics(phase_dir / "evaluation" / "metrics" / "summary.json")
    decisions_dir = eval_output / "decisions"
    samples = _collect_decision_samples(decisions_dir)

    lines: List[str] = [
        "# Phase 2 Evaluation Report",
        "",
        f"_Phase directory: `{phase_dir}`_",
        "",
        "## Training Overview",
    ]
    if not train_stats:
        lines.append("No training stats were found.")
        lines.append("")
    else:
        for label, entries in train_stats.items():
            lines.extend(_format_train_section(label, entries))

    lines.append("## Evaluation Metrics")
    lines.extend(_format_eval_table(metrics))

    lines.append("## Qualitative Samples")
    lines.extend(_format_samples(samples))

    lines.append("## Diagnostics")
    if diagnostics_dir:
        lines.append(f"- Diagnostics artifacts: `{diagnostics_dir.resolve()}`")
    else:
        lines.append("- Diagnostics artifacts were not provided.")
    lines.append(f"- Decisions directory: `{decisions_dir}`")
    lines.append("")

    output_path = output if output.is_absolute() else (Path.cwd() / output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"[report] Phase 2 report written to {output_path}")


if __name__ == "__main__":
    app()

