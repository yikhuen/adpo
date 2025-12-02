#!/usr/bin/env python
"""
Compute inter-judge agreement metrics (percent agreement and Cohen's kappa)
from two JSONL decision logs (e.g., primary__*.jsonl and secondary__*.jsonl).

Each input file is expected to contain one JSON object per line with at least
the field:
    - "choice": the judge's preference label (e.g., "A" or "B").

The script assumes the two files contain the same number of records in the
same order (as produced by the evaluation pipeline), and reports:
    - number of paired decisions
    - raw percent agreement
    - Cohen's kappa
    - the confusion matrix over labels
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import typer

app = typer.Typer(add_completion=False, help="Compute inter-judge agreement metrics.")


def _load_choices(path: Path) -> List[str]:
    """Load the sequence of judge choices from a JSONL decisions file."""
    choices: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = record.get("choice")
            if choice is None:
                continue
            choices.append(str(choice))
    return choices


def _compute_kappa(labels1: List[str], labels2: List[str]) -> Tuple[float, float, Dict[Tuple[str, str], int]]:
    """Compute percent agreement and Cohen's kappa for two label sequences."""
    if len(labels1) != len(labels2):
        raise ValueError(f"Label sequences have different lengths: {len(labels1)} vs {len(labels2)}")
    n = len(labels1)
    if n == 0:
        return 0.0, 0.0, {}

    # Confusion matrix over (label1, label2)
    counts: Counter[Tuple[str, str]] = Counter()
    for a, b in zip(labels1, labels2):
        counts[(a, b)] += 1

    # Observed agreement
    agree = sum(v for (a, b), v in counts.items() if a == b)
    p_o = agree / n

    # Marginal distributions
    marg1: Counter[str] = Counter(labels1)
    marg2: Counter[str] = Counter(labels2)
    labels = sorted(set(marg1.keys()) | set(marg2.keys()))

    # Expected agreement under independence
    p_e = 0.0
    for lab in labels:
        p1 = marg1[lab] / n
        p2 = marg2[lab] / n
        p_e += p1 * p2

    if 1.0 - p_e <= 0.0:
        kappa = 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return p_o, kappa, dict(counts)


@app.command("pair")
def pair(
    primary: Path = typer.Argument(..., help="Path to primary judge decisions JSONL (e.g., primary__*.jsonl)."),
    secondary: Path = typer.Argument(..., help="Path to secondary judge decisions JSONL (e.g., secondary__*.jsonl)."),
) -> None:
    """Compute inter-judge agreement between two decision logs and print summary."""
    primary = primary.resolve()
    secondary = secondary.resolve()

    if not primary.exists():
        raise typer.Exit(code=1)
    if not secondary.exists():
        raise typer.Exit(code=1)

    labels1 = _load_choices(primary)
    labels2 = _load_choices(secondary)

    try:
        p_o, kappa, confusion = _compute_kappa(labels1, labels2)
    except ValueError as exc:
        typer.echo(f"[agreement] error: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"[agreement] primary:   {primary}")
    typer.echo(f"[agreement] secondary: {secondary}")
    typer.echo(f"[agreement] n_pairs:   {len(labels1)}")
    typer.echo(f"[agreement] percent agreement: {p_o * 100:.2f}%")
    typer.echo(f"[agreement] Cohen's kappa:     {kappa:.3f}")
    typer.echo("[agreement] confusion matrix (primary, secondary) counts:")
    for (a, b), count in sorted(confusion.items()):
        typer.echo(f"  ({a}, {b}): {count}")


def _collect_primary_secondary_pairs(decisions_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Discover (primary, secondary) decision file pairs in a directory.

    We look for files named 'primary__*.jsonl' and expect a corresponding
    'secondary__*.jsonl' file with the same suffix.
    """
    pairs: List[Tuple[Path, Path]] = []
    for primary_path in sorted(decisions_dir.glob("primary__*.jsonl")):
        suffix = primary_path.name.replace("primary__", "", 1)
        secondary_path = decisions_dir / f"secondary__{suffix}"
        if secondary_path.exists():
            pairs.append((primary_path, secondary_path))
    return pairs


@app.command("overall")
def overall(
    decisions_dirs: List[Path] = typer.Argument(
        ...,
        help=(
            "One or more directories containing primary__*.jsonl and "
            "secondary__*.jsonl decision files (e.g., eval_results, "
            "all_results/phase_4_results/phase4_helpsteer2/decisions, ...)."
        ),
    ),
) -> None:
    """
    Aggregate inter-judge agreement across multiple evaluation sets.

    For each directory, this command finds matching primary/secondary decision
    files, concatenates all paired choices, and reports overall percent
    agreement and Cohen's kappa, plus the total number of decision pairs.
    """
    all_labels1: List[str] = []
    all_labels2: List[str] = []

    total_pairs = 0
    total_files = 0

    for decisions_dir in decisions_dirs:
        decisions_dir = decisions_dir.resolve()
        if not decisions_dir.exists():
            typer.echo(f"[overall] warning: directory not found: {decisions_dir}")
            continue
        pairs = _collect_primary_secondary_pairs(decisions_dir)
        if not pairs:
            typer.echo(f"[overall] warning: no primary__/secondary__ pairs in {decisions_dir}")
            continue
        typer.echo(f"[overall] scanning {decisions_dir} ({len(pairs)} file pairs)")
        for primary_path, secondary_path in pairs:
            labels1 = _load_choices(primary_path)
            labels2 = _load_choices(secondary_path)
            if len(labels1) != len(labels2):
                typer.echo(
                    f"[overall] skipping mismatched pair: {primary_path.name} "
                    f"({len(labels1)}) vs {secondary_path.name} ({len(labels2)})"
                )
                continue
            all_labels1.extend(labels1)
            all_labels2.extend(labels2)
            total_pairs += len(labels1)
            total_files += 1

    if not all_labels1:
        typer.echo("[overall] no valid decision pairs found.")
        raise typer.Exit(code=1)

    p_o, kappa, _ = _compute_kappa(all_labels1, all_labels2)
    typer.echo(f"[overall] directories scanned: {len(decisions_dirs)}")
    typer.echo(f"[overall] file pairs used:    {total_files}")
    typer.echo(f"[overall] n_pairs:           {total_pairs}")
    typer.echo(f"[overall] percent agreement: {p_o * 100:.2f}%")
    typer.echo(f"[overall] Cohen's kappa:     {kappa:.3f}")


if __name__ == "__main__":
    app()


