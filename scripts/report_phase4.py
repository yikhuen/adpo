#!/usr/bin/env python
"""
Generate a Markdown report for Phase 4 (generalisation) runs, including LLM-curated highlights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from adaptive_dpo.reporting.highlights import curate_highlights, highlights_to_markdown

app = typer.Typer(add_completion=False, help="Summarize Phase 4 results into a Markdown report.")


def _load_metrics(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)):
                ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
            else:
                ci_str = "n/a"
            lines.append(
                f"| {comparison} | {judge} | {entry.get('wins', 0)} | {entry.get('total', 0)} | "
                f"{entry.get('win_rate', 0):.3f} | {ci_str} |"
            )
    lines.append("")
    return lines


@app.command()
def main(
    phase_dir: Path = typer.Option(
        Path("research/results/phase4_generalization"),
        help="Phase 4 output directory.",
    ),
    output: Path = typer.Option(
        Path("results/phase4_report_latest.md"),
        help="Destination Markdown file.",
    ),
    llm_model: Optional[str] = typer.Option(
        "gpt-5-mini",
        help="OpenAI model to curate qualitative highlights ('none' to skip).",
    ),
    llm_max_examples: int = typer.Option(
        4,
        help="Maximum number of highlights per dataset.",
    ),
):
    phase_dir = phase_dir.resolve()
    output = output.resolve()

    lines: List[str] = [
        "# Phase 4 Generalization Report",
        "",
        f"_Phase directory: `{phase_dir}`_",
        "",
    ]

    dataset_dirs = sorted([p for p in phase_dir.iterdir() if p.is_dir()])
    if not dataset_dirs:
        lines.append("No dataset evaluations were found.")
        lines.append("")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        eval_dir = dataset_dir / "evaluation"
        metrics_path = eval_dir / "metrics" / "summary.json"
        decisions_dir = eval_dir / "decisions"
        metrics = _load_metrics(metrics_path)

        lines.append(f"## Dataset: {dataset_name}")
        lines.extend(_format_eval_table(metrics))

        heading = f"### LLM-Curated Highlights – {dataset_name}"
        if llm_model and llm_model.lower() != "none":
            try:
                highlights = curate_highlights(
                    decisions_dir,
                    metrics,
                    model=llm_model,
                    max_examples=llm_max_examples,
                )
                lines.extend(highlights_to_markdown(highlights, heading=heading))
            except Exception as exc:  # pragma: no cover - external API
                typer.echo(f"[report] Dataset {dataset_name}: LLM highlights skipped ({exc})")
                lines.extend(highlights_to_markdown([], heading=heading))
                lines.append(f"_LLM generation failed: {exc}_")
                lines.append("")
        else:
            lines.extend(highlights_to_markdown([], heading=heading))

    summary_path = phase_dir / "summary.json"
    if summary_path.exists():
        summary_metrics = _load_metrics(summary_path)
        lines.append("## Aggregate Metrics")
        lines.extend(_format_eval_table(summary_metrics))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")
    typer.echo(f"[report] Wrote Phase 4 report to {output}")


if __name__ == "__main__":
    app()

