from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from adaptive_dpo.pipelines.orchestration import (
    run_phase1,
    run_phase2,
    run_phase3,
    run_phase4,
)

app = typer.Typer(help="Orchestrate multi-phase Adaptive DPO experiments.")


@app.command()
def phase1(
    base_config: Path = typer.Option(
        Path("configs/train/qwen25_7b_fixed_beta.yaml"),
        help="Base fixed-beta config.",
    ),
    betas: List[float] = typer.Option(
        [0.05, 0.10, 0.20],
        help="Fixed beta values to sweep.",
    ),
):
    run_phase1(base_config, betas)


@app.command()
def phase2(
    adaptive_config: Path = typer.Option(
        Path("configs/train/qwen25_7b_adaptive_beta.yaml"),
        help="Adaptive controller config path.",
    ),
    annealed_config: Path = typer.Option(
        Path("configs/train/qwen25_7b_annealed_beta.yaml"),
        help="Annealed beta baseline config path.",
    ),
    fixed_config: Path = typer.Option(
        Path("configs/train/qwen25_7b_fixed_beta.yaml"),
        help="Oracle fixed beta config path.",
    ),
    eval_config: Path = typer.Option(
        Path("configs/eval/judge_gpt4o_mini.yaml"),
        help="Evaluation config for win-rate comparison.",
    ),
    include_fixed_beta: bool = typer.Option(True, help="Run the fixed-beta baseline as part of phase 2."),
    force_eval: bool = typer.Option(False, help="Force re-run evaluation even if cached."),
    oracle_beta: Optional[float] = typer.Option(
        None,
        help="Override the fixed-beta baseline with the supplied beta value (e.g. best from Phase 1).",
    ),
    audit_batch_index: int = typer.Option(
        15, help="Zero-based batch index to inspect during the poison audit (default targets Step 16)."
    ),
    audit_wandb_project: Optional[str] = typer.Option(
        None, help="Optional W&B project name for logging poison audit results."
    ),
    eval_prompt_size: int = typer.Option(
        200, help="Number of evaluation prompts to generate for Phase 2 judges."
    ),
):
    run_phase2(
        adaptive_config,
        annealed_config,
        fixed_config,
        eval_config,
        include_fixed_beta,
        force_eval,
        oracle_beta,
        audit_batch_index,
        audit_wandb_project,
        eval_prompt_size,
    )


@app.command()
def phase3(
    base_config: Path = typer.Option(
        Path("configs/train/qwen25_7b_adaptive_beta.yaml"),
        help="Base adaptive config used for ablations.",
    ),
    ablations: List[str] = typer.Option(
        ["full", "no_deadband", "no_ema", "no_clipping", "no_fast_loop"],
        help="Controller variants to run.",
    ),
):
    run_phase3(base_config, ablations)


@app.command()
def phase4(
    eval_config: Path = typer.Option(..., help="Base evaluation config to clone per dataset."),
    datasets: List[str] = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Dataset spec 'name=path/to/prompts.jsonl'. Repeat per dataset.",
    ),
    models: List[str] = typer.Option(
        [],
        "--model",
        "-m",
        help="Model spec 'name=kind:lora,checkpoint:path'. Overrides config models when provided.",
    ),
    force_eval: bool = typer.Option(False, help="Force re-run judges even if cached decisions exist."),
):
    run_phase4(eval_config, datasets, models, force_eval)


def entrypoint():
    app()


if __name__ == "__main__":
    entrypoint()

