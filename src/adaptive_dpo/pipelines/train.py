from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from trl import DPOConfig, DPOTrainer
from trl.trainer.kto_config import KTOConfig
from trl.trainer.kto_trainer import KTOTrainer

from adaptive_dpo.controllers import (
    AdaptiveBetaController,
    BetaControllerConfig,
    HybridAdaptiveKLController,
    HybridControllerConfig,
    MethodSpec,
    RobustHybridConfig,
    RobustHybridController,
    resolve_method_config,
)
from adaptive_dpo.data import load_preference_dataset, load_ultrafeedback_subset_formatted
from adaptive_dpo.modeling import load_qwen25_7b
from adaptive_dpo.trainer import AdaptiveDPOTrainer, LoggingDPOTrainer
from adaptive_dpo.utils.repro import set_global_seed
from adaptive_dpo.utils.schedules import AnnealedBetaCallback, AnnealedBetaConfig


@dataclass
class TrainerArtifacts:
    trainer: Any
    run_output_dir: Path
    controller: Optional[object]
    tokenizer: Any
    train_dataset: Any
    schedule_cfg: Optional[Dict[str, Any]]
    trainer_cfg: Dict[str, Any]
    method_label: str


def _load_dataset(tokenizer, ds_cfg: Dict[str, Any]):
    if any(key in ds_cfg for key in ("alias", "path", "splits", "format_kwargs")):
        return load_preference_dataset(tokenizer, ds_cfg)
    return load_ultrafeedback_subset_formatted(
        tokenizer=tokenizer,
        sample_frac=float(ds_cfg.get("sample_frac", 0.005)),
        splits=ds_cfg.get("splits", ["train_prefs", "test_prefs"]),
    )


def _build_training_args(
    tr_cfg: Dict[str, Any],
    seed: int,
    output_dir: Path,
    *,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
) -> DPOConfig:
    return DPOConfig(
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 12)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.1)),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 1)),
        max_steps=int(tr_cfg.get("max_steps", -1)),
        learning_rate=float(tr_cfg.get("learning_rate", 5e-6)),
        logging_steps=int(tr_cfg.get("logging_steps", 1)),
        optim=str(tr_cfg.get("optim", "adamw_8bit")),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tr_cfg.get("lr_scheduler_type", "linear")),
        seed=seed,
        output_dir=str(output_dir),
        report_to=str(tr_cfg.get("report_to", "wandb")),
        loss_type=[loss_type],
        reference_free=reference_free,
    )


def _build_kto_training_args(
    tr_cfg: Dict[str, Any],
    seed: int,
    output_dir: Path,
    *,
    desirable_weight: float,
    undesirable_weight: float,
) -> KTOConfig:
    return KTOConfig(
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 12)),
        warmup_ratio=float(tr_cfg.get("warmup_ratio", 0.1)),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 1)),
        max_steps=int(tr_cfg.get("max_steps", -1)),
        learning_rate=float(tr_cfg.get("learning_rate", 5e-6)),
        logging_steps=int(tr_cfg.get("logging_steps", 1)),
        optim=str(tr_cfg.get("optim", "adamw_8bit")),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=str(tr_cfg.get("lr_scheduler_type", "linear")),
        seed=seed,
        output_dir=str(output_dir),
        report_to=str(tr_cfg.get("report_to", "wandb")),
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )


def _instantiate_controller(controller_cfg: Optional[Dict[str, Any]]) -> Tuple[Optional[object], Optional[float]]:
    if not controller_cfg:
        return None, None
    controller_kind = controller_cfg.get("kind", "pid")
    controller_payload = {k: v for k, v in controller_cfg.items() if k != "kind"}
    if controller_kind == "hybrid_entropy":
        cfg = HybridControllerConfig(**controller_payload)
        return HybridAdaptiveKLController(cfg), cfg.beta_init
    if controller_kind == "robust_hybrid":
        cfg = RobustHybridConfig(**controller_payload)
        return RobustHybridController(cfg), cfg.beta_init
    cfg = BetaControllerConfig(**controller_payload)
    return AdaptiveBetaController(cfg), cfg.beta_init


def _load_policy_and_reference(model_cfg: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    kwargs = {
        "max_seq_length": int(model_cfg.get("max_seq_length", 4096)),
        "load_in_4bit": bool(model_cfg.get("load_in_4bit", True)),
        "dtype": model_cfg.get("dtype", None),
    }
    model, tokenizer = load_qwen25_7b(**kwargs)
    ref_model, _ = load_qwen25_7b(**kwargs)
    return model, ref_model, tokenizer


def _prepare_run_output_dir(tr_cfg: Dict[str, Any], seed: int, total_runs: int) -> Path:
    base_output_dir = Path(tr_cfg.get("output_dir", "outputs"))
    run_output_dir = base_output_dir / f"seed_{seed}" if total_runs > 1 else base_output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def _save_phase_trace(run_output_dir: Path, phase_trace: List[Dict[str, Any]]) -> Path:
    phase_path = run_output_dir / "phase_trace.json"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with open(phase_path, "w", encoding="utf-8") as f:
        json.dump(phase_trace, f, indent=2)
    return phase_path


def _log_phase_plot(phase_trace: List[Dict[str, Any]], run_output_dir: Path, run_label: str) -> None:
    if not phase_trace:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        return

    points = []
    for entry in phase_trace:
        kl_val = entry.get("kl_ema") or entry.get("kl_batch")
        beta_val = entry.get("beta")
        step_val = entry.get("global_step", 0)
        if kl_val is None or beta_val is None:
            continue
        points.append((float(kl_val), float(beta_val), int(step_val)))
    if not points:
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


def _build_trainer(
    cfg: Dict[str, Any],
    seed: int,
    run_idx: int,
    total_runs: int,
) -> TrainerArtifacts:
    method_spec = resolve_method_config(cfg)
    if method_spec.trainer_kind == "kto":
        return _build_kto_trainer(cfg, seed, run_idx, total_runs, method_spec)
    return _build_dpo_trainer(cfg, seed, run_idx, total_runs, method_spec)


def _build_dpo_trainer(
    cfg: Dict[str, Any],
    seed: int,
    run_idx: int,
    total_runs: int,
    method_spec: MethodSpec,
) -> TrainerArtifacts:
    model_cfg = cfg["model"]
    tr_cfg = cfg["trainer"]
    ds_cfg = cfg["dataset"]
    controller_cfg = cfg.get("beta_controller") if method_spec.name == "adaptive" else None

    set_global_seed(seed)

    model, ref_model, tokenizer = _load_policy_and_reference(model_cfg)

    dataset = _load_dataset(tokenizer, ds_cfg)
    train_split_name = tr_cfg.get("train_split", "train")
    if train_split_name not in dataset:
        raise KeyError(f"Requested train split '{train_split_name}' not found in dataset splits {list(dataset.keys())}.")
    train_dataset = dataset[train_split_name]

    run_output_dir = _prepare_run_output_dir(tr_cfg, seed, total_runs)

    args = _build_training_args(
        tr_cfg,
        seed,
        run_output_dir,
        loss_type=method_spec.loss_type_arg,
        reference_free=method_spec.reference_free,
    )

    controller = None
    controller_beta = None

    if method_spec.name == "adaptive":
        controller, controller_beta = _instantiate_controller(controller_cfg)
    elif method_spec.name == "beta_dpo" and method_spec.beta_dpo_config:
        controller = BetaDPOController(method_spec.beta_dpo_config)
        controller_beta = method_spec.beta_dpo_config.beta_min
    elif method_spec.name == "epsilon_dpo" and method_spec.epsilon_dpo_config:
        controller = EpsilonDPOController(method_spec.epsilon_dpo_config)
        controller_beta = method_spec.epsilon_dpo_config.beta_init
    else:
        # Fixed/SimPO/IPO/Annealed (handled via callback or static beta)
        pass

    beta_init = controller_beta if controller_beta is not None else float(cfg.get("fixed_beta", 0.1))
    kl_log_alpha = float(tr_cfg.get("kl_log_alpha", 0.10))
    high_kl_threshold = float(tr_cfg.get("high_kl_threshold")) if tr_cfg.get("high_kl_threshold") is not None else None

    trainer_kwargs = dict(
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

    if controller:
        trainer: DPOTrainer = AdaptiveDPOTrainer(
            beta_controller=controller,
            high_kl_threshold=high_kl_threshold,
            **trainer_kwargs,
        )
    else:
        trainer = LoggingDPOTrainer(
            loss_type_override=method_spec.loss_override,
            simpo_gamma=method_spec.simpo_gamma,
            **trainer_kwargs,
        )

    schedule_cfg = cfg.get("beta_schedule")
    if schedule_cfg:
        schedule_callback = AnnealedBetaCallback(AnnealedBetaConfig(**schedule_cfg))
        schedule_callback.trainer = trainer
        trainer.add_callback(schedule_callback)

    return TrainerArtifacts(
        trainer=trainer,
        run_output_dir=run_output_dir,
        controller=controller,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        schedule_cfg=schedule_cfg,
        trainer_cfg=tr_cfg,
        method_label=method_spec.label,
    )


def _build_kto_trainer(
    cfg: Dict[str, Any],
    seed: int,
    run_idx: int,
    total_runs: int,
    method_spec: MethodSpec,
) -> TrainerArtifacts:
    model_cfg = cfg["model"]
    tr_cfg = cfg["trainer"]
    ds_cfg = cfg["dataset"]

    set_global_seed(seed)

    model, ref_model, tokenizer = _load_policy_and_reference(model_cfg)

    dataset = _load_dataset(tokenizer, ds_cfg)
    train_split_name = tr_cfg.get("train_split", "train")
    if train_split_name not in dataset:
        raise KeyError(f"Requested train split '{train_split_name}' not found in dataset splits {list(dataset.keys())}.")
    train_dataset = dataset[train_split_name]

    run_output_dir = _prepare_run_output_dir(tr_cfg, seed, total_runs)

    args = _build_kto_training_args(
        tr_cfg,
        seed,
        run_output_dir,
        desirable_weight=method_spec.desirable_weight,
        undesirable_weight=method_spec.undesirable_weight,
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    return TrainerArtifacts(
        trainer=trainer,
        run_output_dir=run_output_dir,
        controller=None,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        schedule_cfg=None,
        trainer_cfg=tr_cfg,
        method_label=method_spec.label,
    )


def _train_single_run(cfg: Dict[str, Any], seed: int, run_idx: int, total_runs: int) -> Dict[str, Any]:
    artifacts = _build_trainer(cfg, seed, run_idx, total_runs)
    trainer = artifacts.trainer
    run_output_dir = artifacts.run_output_dir
    controller = artifacts.controller
    tokenizer = artifacts.tokenizer
    schedule_cfg = artifacts.schedule_cfg

    start_time = time.time()
    trainer.train()
    wall_time = time.time() - start_time

    try:
        trainer.model.save_pretrained(run_output_dir)
        tokenizer.save_pretrained(run_output_dir)
    except Exception:
        pass

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
        "train_examples": len(artifacts.train_dataset),
        "train_runtime_seconds": final_log.get("train_runtime"),
        "train_samples_per_second": final_log.get("train_samples_per_second"),
        "train_steps_per_second": final_log.get("train_steps_per_second"),
        "wall_clock_seconds": wall_time,
        "dataset_alias": cfg["dataset"].get("alias", "ultrafeedback"),
        "beta_initial": getattr(trainer, "beta", None),
        "beta_final": getattr(trainer, "beta", None),
        "method": artifacts.method_label,
    }
    if controller:
        stats["controller_state"] = controller.state()
    if schedule_cfg:
        stats["beta_schedule"] = schedule_cfg

    with open(run_output_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def run_training(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    seeds_cfg = cfg.get("seeds")
    if seeds_cfg is None:
        seeds: List[int] = [int(cfg.get("seed", 42))]
    elif isinstance(seeds_cfg, (list, tuple)):
        seeds = [int(s) for s in seeds_cfg]
    else:
        seeds = [int(seeds_cfg)]

    results: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        stats = _train_single_run(cfg, int(seed), idx, len(seeds))
        results.append(stats)

    trainer_cfg = cfg.get("trainer", {})
    base_output_dir = Path(trainer_cfg.get("output_dir", "outputs"))
    if len(results) > 1:
        summary_path = base_output_dir / "train_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results

