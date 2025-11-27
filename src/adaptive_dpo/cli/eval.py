from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from adaptive_dpo.eval.runner import run_evaluation

DEFAULT_ALL_JUDGES_CONFIG = Path("configs/eval/judge_gpt4o_mini.yaml")
DEFAULT_OPENAI_ONLY_CONFIG = Path("configs/eval/judge_openai_only.yaml")
DEFAULT_OPENROUTER_ONLY_CONFIG = Path("configs/eval/judge_openrouter_only.yaml")

app = typer.Typer(help="Evaluate preference models with multiple judges and export metrics.")


def _run(config: Path, limit: Optional[int], force_generate: bool, force_judge: bool):
    typer.echo(f"[eval] Using config: {config}")
    run_evaluation(str(config), limit, force_generate, force_judge)


@app.command()
def main(
    config: Path = typer.Option(..., exists=True, readable=True, help="Path to evaluation YAML config."),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    _run(config, limit, force_generate, force_judge)


@app.command("openai-judge")
def run_openai_judge(
    config: Path = typer.Option(
        DEFAULT_OPENAI_ONLY_CONFIG,
        "--config",
        "-c",
        help="Path to an OpenAI-only eval config.",
        show_default=False,
    ),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    _run(config, limit, force_generate, force_judge)


@app.command("openrouter-judge")
def run_openrouter_judge(
    config: Path = typer.Option(
        DEFAULT_OPENROUTER_ONLY_CONFIG,
        "--config",
        "-c",
        help="Path to an OpenRouter-only eval config.",
        show_default=False,
    ),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    _run(config, limit, force_generate, force_judge)


@app.command("all-judges")
def run_all_judges(
    config: Path = typer.Option(
        DEFAULT_ALL_JUDGES_CONFIG,
        "--config",
        "-c",
        help="Path to the combined judge eval config.",
        show_default=False,
    ),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    _run(config, limit, force_generate, force_judge)


def entrypoint():
    app()


if __name__ == "__main__":
    entrypoint()

