from __future__ import annotations

from pathlib import Path

import typer
import yaml
import unsloth  # noqa: F401

from adaptive_dpo.pipelines.train import run_training

app = typer.Typer(help="Train adaptive or baseline DPO models with flexible beta control.")


@app.command()
def main(config: Path = typer.Option(..., exists=True, readable=True, help="Path to training YAML config")):
    with config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    typer.echo(f"[train] Loaded config from {config}")
    results = run_training(cfg)
    if results:
        typer.echo(f"[train] Completed {len(results)} run(s).")


def entrypoint():
    app()


if __name__ == "__main__":
    entrypoint()

