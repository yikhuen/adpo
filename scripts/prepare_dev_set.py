import json
import os
from typing import Optional

import typer
import yaml
from transformers import AutoTokenizer

from adaptive_dpo.data import FORMATTERS, load_preference_dataset

DEFAULT_SPLITS = {
    "ultrafeedback": {"train": "train_prefs", "test": "test_prefs"},
    "anthropic_hh": {"train": "train", "test": "test"},
    "sycophancy": {"train": "train", "test": "test"},
}

app = typer.Typer()


@app.command()
def main(
    dataset: Optional[str] = typer.Option(
        None,
        help="Dataset alias or path; if omitted, load from YAML config via --config.",
    ),
    size: int = typer.Option(200, help="Number of prompts to export."),
    out: str = typer.Option("data/dev.jsonl", help="Destination JSONL file."),
    config: Optional[str] = typer.Option(
        None,
        help="YAML file containing a 'dataset' section compatible with training config.",
    ),
    split: str = typer.Option("test", help="Which split to export prompts from."),
    tokenizer_name: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer name/path used for chat template.",
    ),
    respect_config_sampling: bool = typer.Option(
        False,
        help="If False (default), ignore sample_frac/sample_size from training config so the full split is available.",
    ),
):
    """
    Export a JSONL file of prompts for evaluation/dev set generation.

    The loader reuses preference dataset formatting utilities so prompts
    match the training pipeline (system + user chat template applied).
    """
    if not dataset and not config:
        raise typer.BadParameter("Provide either --dataset alias/path or --config YAML with dataset settings.")

    if config:
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dataset_cfg = cfg.get("dataset", cfg)
    else:
        if dataset in FORMATTERS:
            alias = dataset
            path_override = None
        else:
            alias = next(
                (key for key, value in FORMATTERS.items() if dataset == value["default_path"]),
                None,
            )
            if alias is None:
                raise typer.BadParameter(
                    f"Unknown dataset alias/path '{dataset}'. Provide a YAML config to describe custom datasets."
                )
            path_override = dataset
        dataset_cfg = {
            "alias": alias,
            "path": path_override,
            "splits": DEFAULT_SPLITS.get(alias, {"train": "train"}),
        }

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    if not respect_config_sampling:
        # Remove sampling directives inherited from the training config so we can export
        # the full evaluation split (or control size purely via --size).
        dataset_cfg.pop("sample_frac", None)
        dataset_cfg.pop("sample_size", None)

    ds_dict = load_preference_dataset(tokenizer, dataset_cfg)

    if split not in ds_dict:
        available = ", ".join(ds_dict.keys())
        raise typer.BadParameter(f"Split '{split}' not available. Options: {available}.")

    ds_split = ds_dict[split]
    n = min(size, len(ds_split))
    subset = ds_split.select(range(n))

    prompts = [{"prompt": row["prompt"]} for row in subset]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
