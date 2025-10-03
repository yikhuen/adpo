import os
import sys
# Ensure local src/ is importable when running as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import unsloth
import json
import yaml
import typer
from typing import Optional

from transformers import TrainingArguments
from trl import DPOConfig

from adaptive_dpo.beta_controller import BetaControllerConfig, AdaptiveBetaController
from adaptive_dpo.modeling import load_qwen25_7b
from adaptive_dpo.data import load_ultrafeedback_subset_formatted
from adaptive_dpo.trainer import AdaptiveDPOTrainer
from adaptive_dpo.utils.repro import set_global_seed

app = typer.Typer()


@app.command()
def main(config: str = typer.Option(..., help="Path to training YAML config")):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_global_seed(int(cfg.get("seed", 42)))

    # Load model
    model_cfg = cfg["model"]
    model, tokenizer = load_qwen25_7b(
        max_seq_length=int(model_cfg.get("max_seq_length", 4096)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        dtype=model_cfg.get("dtype", None),
    )

    # Load dataset subset and format
    ds_cfg = cfg["dataset"]
    ds = load_ultrafeedback_subset_formatted(
        tokenizer=tokenizer,
        sample_frac=float(ds_cfg.get("sample_frac", 0.005)),
        splits=ds_cfg.get("splits", ["train_prefs", "test_prefs"]),
    )

    # Controller or fixed beta
    if "beta_controller" in cfg:
        bc_cfg = BetaControllerConfig(**cfg["beta_controller"]) 
        controller = AdaptiveBetaController(bc_cfg)
        beta_init = bc_cfg.beta_init
    else:
        controller = None
        beta_init = float(cfg.get("fixed_beta", 0.1))

    # Training args
    tr_cfg = cfg["trainer"]
    args = DPOConfig(
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 12)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.1)),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 1)),
        learning_rate=float(tr_cfg.get("learning_rate", 5e-6)),
        logging_steps=int(tr_cfg.get("logging_steps", 1)),
        optim=str(tr_cfg.get("optim", "adamw_8bit")),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tr_cfg.get("lr_scheduler_type", "linear")),
        seed=int(cfg.get("seed", 42)),
        output_dir=str(tr_cfg.get("output_dir", "outputs")),
        report_to=str(tr_cfg.get("report_to", "wandb")),
    )

    # Build trainer with a separate frozen ref_model for KL measurement
    from transformers import AutoModelForCausalLM
    ref_model, _ = load_qwen25_7b(
        max_seq_length=int(model_cfg.get("max_seq_length", 4096)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        dtype=model_cfg.get("dtype", None),
    )

    if controller is not None:
        trainer = AdaptiveDPOTrainer(
            beta_controller=controller,
            model=model,
            ref_model=ref_model,
            args=args,
            beta=beta_init,
            train_dataset=ds["train"],
            tokenizer=tokenizer,
            max_length=int(tr_cfg.get("max_length", 1024)),
            max_prompt_length=int(tr_cfg.get("max_prompt_length", 512)),
        )
    else:
        from trl import DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            beta=beta_init,
            train_dataset=ds["train"],
            tokenizer=tokenizer,
            max_length=int(tr_cfg.get("max_length", 1024)),
            max_prompt_length=int(tr_cfg.get("max_prompt_length", 512)),
        )

    trainer.train()

    # Save adapters and tokenizer
    output_dir = tr_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    try:
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    app()
