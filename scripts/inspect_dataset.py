import json
from typing import Optional

import typer
import yaml
from transformers import AutoTokenizer

from adaptive_dpo.data import load_preference_dataset

app = typer.Typer(help="Inspect preference datasets after formatting into prompt/chosen/rejected.")


def _load_dataset_config(config_path: Optional[str], alias: Optional[str], path: Optional[str]) -> dict:
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dataset_cfg = cfg.get("dataset", cfg)
    else:
        if not alias:
            raise typer.BadParameter("Alias is required when --config is not provided.")
        dataset_cfg = {
            "alias": alias,
            "path": path,
            "splits": {"train": "train"},
        }
    return dataset_cfg


@app.command()
def main(
    config: Optional[str] = typer.Option(
        None, help="YAML file containing dataset section (same as training config)."
    ),
    alias: Optional[str] = typer.Option(
        None, help="Dataset alias (ultrafeedback, anthropic_hh, sycophancy, ...)"
    ),
    path: Optional[str] = typer.Option(
        None, help="Override Hugging Face dataset path when not using --config."
    ),
    split: str = typer.Option("train", help="Split name to inspect after formatting."),
    samples: int = typer.Option(3, help="Number of formatted examples to print."),
    tokenizer_name: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer used for chat template rendering.",
    ),
    format_kwargs: Optional[str] = typer.Option(
        None,
        help="JSON/YAML string of formatter overrides (e.g., column names for sycophancy).",
    ),
    show_prompt: bool = typer.Option(True, help="Whether to print rendered prompt text."),
    show_responses: bool = typer.Option(True, help="Whether to print chosen/rejected responses."),
):
    """
    Preview dataset formatting to validate preprocessing assumptions before training/evaluation.
    """
    dataset_cfg = _load_dataset_config(config, alias, path)
    if format_kwargs:
        overrides = yaml.safe_load(format_kwargs)
        dataset_cfg.setdefault("format_kwargs", {}).update(overrides)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    ds_dict = load_preference_dataset(tokenizer, dataset_cfg)
    if split not in ds_dict:
        available = ", ".join(ds_dict.keys())
        raise typer.BadParameter(f"Split '{split}' not found. Available splits: {available}.")

    subset = ds_dict[split].select(range(min(samples, len(ds_dict[split]))))
    for idx, row in enumerate(subset):
        typer.echo(f"\nExample {idx+1}/{len(subset)} ---")
        metadata = {k: v for k, v in row.items() if k not in {"prompt", "chosen", "rejected"}}
        if metadata:
            typer.echo(f"Meta: {json.dumps(metadata, ensure_ascii=False)}")
        if show_prompt:
            typer.echo("Prompt:")
            typer.echo(row["prompt"])
        if show_responses:
            typer.echo("\nChosen:")
            typer.echo(row["chosen"])
            typer.echo("\nRejected:")
            typer.echo(row["rejected"])


if __name__ == "__main__":
    app()

