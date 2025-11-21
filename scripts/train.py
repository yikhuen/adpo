import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure local src/ is importable when running as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import unsloth  # noqa: F401  (ensures fast transformers patches are applied)
import typer
import yaml
from trl import DPOConfig, DPOTrainer

from adaptive_dpo.beta_controller import AdaptiveBetaController, BetaControllerConfig
from adaptive_dpo.data import load_preference_dataset, load_ultrafeedback_subset_formatted
from adaptive_dpo.modeling import load_qwen25_7b
from adaptive_dpo.trainer import AdaptiveDPOTrainer, LoggingDPOTrainer
from adaptive_dpo.utils.repro import set_global_seed
from adaptive_dpo.utils.schedules import AnnealedBetaCallback, AnnealedBetaConfig

app = typer.Typer(help="Train adaptive or baseline DPO models with flexible beta control.")


def _load_dataset(tokenizer, ds_cfg: Dict[str, Any]):
    if any(key in ds_cfg for key in ("alias", "path", "splits", "format_kwargs")):
        return load_preference_dataset(tokenizer, ds_cfg)
    return load_ultrafeedback_subset_formatted(
        tokenizer=tokenizer,
        sample_frac=float(ds_cfg.get("sample_frac", 0.005)),
        splits=ds_cfg.get("splits", ["train_prefs", "test_prefs"]),
    )


def _build_training_args(tr_cfg: Dict[str, Any], seed: int, output_dir: Path) -> DPOConfig:
    return DPOConfig(
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 12)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.1)),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 1)),
        learning_rate=float(tr_cfg.get("learning_rate", 5e-6)),
        logging_steps=int(tr_cfg.get("logging_steps", 1)),
        optim=str(tr_cfg.get("optim", "adamw_8bit")),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tr_cfg.get("lr_scheduler_type", "linear")),
        seed=seed,
        output_dir=str(output_dir),
        report_to=str(tr_cfg.get("report_to", "wandb")),
    )


def _save_phase_trace(run_output_dir: Path, phase_trace: List[Dict[str, Any]]) -> Path:
    phase_path = run_output_dir / "phase_trace.json"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with open(phase_path, "w", encoding="utf-8") as f:
        json.dump(phase_trace, f, indent=2)
    return phase_path


def _log_phase_plot(
    phase_trace: List[Dict[str, Any]],
    run_output_dir: Path,
    run_label: str,
) -> None:
    if not phase_trace:
        return

    points = []
    for entry in phase_trace:
        kl_val = entry.get("kl_ema")
        if kl_val is None:
            kl_val = entry.get("kl_batch")
        beta_val = entry.get("beta")
        step_val = entry.get("global_step", 0)
        if kl_val is None or beta_val is None:
            continue
        points.append((float(kl_val), float(beta_val), int(step_val)))

    if not points:
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        typer.echo("[train] Phase plot skipped (matplotlib/seaborn not installed).")
        return

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))

    kl_values = np.array([p[0] for p in points])
    beta_values = np.array([p[1] for p in points])
    step_values = np.array([p[2] for p in points])

    plt.plot(kl_values, beta_values, color="orange", alpha=0.4, linewidth=1, zorder=1)
    sc = plt.scatter(
        kl_values,
        beta_values,
        c=step_values,
        cmap="viridis",
        s=60,
        edgecolors="black",
        zorder=2,
        alpha=0.85,
    )

    max_idx = int(np.argmax(kl_values))
    plt.annotate(
        "Poison Batch Impact",
        xy=(kl_values[max_idx], beta_values[max_idx]),
        xytext=(kl_values[max_idx], beta_values[max_idx] + 0.05),
        arrowprops=dict(facecolor="red", shrink=0.05),
        fontsize=10,
        fontweight="bold",
    )

    plt.title("Controller Phase Portrait: Beta Response to KL Divergence", fontsize=14)
    plt.xlabel("KL Divergence (Error Signal)", fontsize=12)
    plt.ylabel("Beta Value (Control Output)", fontsize=12)
    plt.colorbar(sc, label="Global Step")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = run_output_dir / "phase_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)

    try:
        import wandb

        if wandb.run is not None:
            safe_label = run_label.replace("/", "_")
            wandb.log({f"phase/{safe_label}_plot": wandb.Image(output_path)}, commit=False)
    except Exception:
        pass
    finally:
        plt.close()

def _train_single_run(cfg: Dict[str, Any], seed: int, run_idx: int, total_runs: int) -> Dict[str, Any]:
    model_cfg = cfg["model"]
    tr_cfg = cfg["trainer"]
    ds_cfg = cfg["dataset"]
    schedule_cfg = cfg.get("beta_schedule")
    controller_cfg = cfg.get("beta_controller")
    fixed_beta = cfg.get("fixed_beta", 0.1)

    set_global_seed(seed)

    model, tokenizer = load_qwen25_7b(
        max_seq_length=int(model_cfg.get("max_seq_length", 4096)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        dtype=model_cfg.get("dtype", None),
    )
    ref_model, _ = load_qwen25_7b(
        max_seq_length=int(model_cfg.get("max_seq_length", 4096)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
        dtype=model_cfg.get("dtype", None),
    )

    dataset = _load_dataset(tokenizer, ds_cfg)
    train_split_name = tr_cfg.get("train_split", "train")
    if train_split_name not in dataset:
        raise KeyError(f"Requested train split '{train_split_name}' not found in dataset splits {list(dataset.keys())}.")
    train_dataset = dataset[train_split_name]
    train_size = len(train_dataset)

    base_output_dir = Path(tr_cfg.get("output_dir", "outputs"))
    if total_runs > 1:
        run_output_dir = base_output_dir / f"seed_{seed}"
    else:
        run_output_dir = base_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    args = _build_training_args(tr_cfg, seed, run_output_dir)

    controller: Optional[AdaptiveBetaController] = None
    beta_init = float(fixed_beta)
    if controller_cfg:
        bc_cfg = BetaControllerConfig(**controller_cfg)
        controller = AdaptiveBetaController(bc_cfg)
        beta_init = bc_cfg.beta_init

    kl_log_alpha = float(tr_cfg.get("kl_log_alpha", 0.10))

    if controller:
        trainer: DPOTrainer = AdaptiveDPOTrainer(
            beta_controller=controller,
            model=model,
            ref_model=ref_model,
            args=args,
            beta=beta_init,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_length=int(tr_cfg.get("max_length", 1024)),
            max_prompt_length=int(tr_cfg.get("max_prompt_length", 512)),
            kl_log_alpha=kl_log_alpha,
            fixed_beta_value=beta_init,
        )
    else:
        trainer = LoggingDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            beta=beta_init,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_length=int(tr_cfg.get("max_length", 1024)),
            max_prompt_length=int(tr_cfg.get("max_prompt_length", 512)),
            kl_log_alpha=kl_log_alpha,
            fixed_beta_value=beta_init,
        )

    # Optional beta scheduling baseline
    schedule_callback = None
    if schedule_cfg:
        schedule_callback = AnnealedBetaCallback(AnnealedBetaConfig(**schedule_cfg))
        schedule_callback.trainer = trainer
        trainer.add_callback(schedule_callback)

    start_time = time.time()
    trainer.train()
    wall_time = time.time() - start_time

    # Persist adapters/tokenizer per run
    try:
        trainer.model.save_pretrained(run_output_dir)
        tokenizer.save_pretrained(run_output_dir)
    except Exception:
        pass

    # Collect metrics
    log_history = trainer.state.log_history or []
    final_log = {}
    for entry in reversed(log_history):
        if "train_runtime" in entry or "loss" in entry:
            final_log = entry
            break

    phase_trace = getattr(trainer, "phase_trace", [])
    if phase_trace:
        _save_phase_trace(run_output_dir, phase_trace)
        run_label = getattr(trainer.args, "run_name", None) or run_output_dir.name
        _log_phase_plot(phase_trace, run_output_dir, run_label)

    stats = {
        "seed": seed,
        "run_index": run_idx,
        "output_dir": str(run_output_dir),
        "train_global_step": trainer.state.global_step,
        "train_examples": train_size,
        "train_runtime_seconds": final_log.get("train_runtime"),
        "train_samples_per_second": final_log.get("train_samples_per_second"),
        "train_steps_per_second": final_log.get("train_steps_per_second"),
        "wall_clock_seconds": wall_time,
        "dataset_alias": ds_cfg.get("alias", "ultrafeedback"),
        "beta_initial": beta_init,
        "beta_final": getattr(trainer, "beta", beta_init),
    }
    if controller:
        stats["controller_state"] = controller.state()
    if schedule_cfg:
        stats["beta_schedule"] = schedule_cfg

    with open(run_output_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


@app.command()
def main(config: str = typer.Option(..., help="Path to training YAML config")):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seeds_cfg = cfg.get("seeds")
    if seeds_cfg is None:
        seeds: List[int] = [int(cfg.get("seed", 42))]
    elif isinstance(seeds_cfg, (list, tuple)):
        seeds = [int(s) for s in seeds_cfg]
    else:
        seeds = [int(seeds_cfg)]

    results: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        typer.echo(f"[train] Starting run {idx+1}/{len(seeds)} with seed={seed}")
        stats = _train_single_run(cfg, int(seed), idx, len(seeds))
        results.append(stats)

    trainer_cfg = cfg.get("trainer", {})
    base_output_dir = Path(trainer_cfg.get("output_dir", "outputs"))
    if len(results) > 1:
        summary_path = base_output_dir / "train_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"[train] Wrote multi-run summary to {summary_path}")


if __name__ == "__main__":
    app()
