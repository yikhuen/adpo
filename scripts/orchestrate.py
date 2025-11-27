import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_ROOT = _REPO_ROOT / "research" / "results"
CONFIG_CACHE = RESULTS_ROOT / "configs"

app = typer.Typer(help="Orchestrate multi-phase Adaptive DPO experiments.")


def _slug(value: str) -> str:
    return value.replace(".", "p").replace("/", "_").replace(" ", "_")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (_REPO_ROOT / path)


def _write_config(config: Dict, filename: str) -> Path:
    CONFIG_CACHE.mkdir(parents=True, exist_ok=True)
    path = CONFIG_CACHE / filename
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def _relativize(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_dataset_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise typer.BadParameter(
            f"Dataset spec '{spec}' must be in the form name=path/to/prompts.jsonl."
        )
    label, path_value = (part.strip() for part in spec.split("=", 1))
    if not label:
        raise typer.BadParameter(f"Dataset spec '{spec}' is missing a name before '='.")
    if not path_value:
        raise typer.BadParameter(f"Dataset spec '{spec}' is missing a path after '='.")
    resolved = _resolve_path(Path(path_value))
    if not resolved.exists():
        raise typer.BadParameter(f"Dataset prompt file '{resolved}' does not exist.")
    return label, _relativize(resolved)


def _parse_model_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    if "=" not in spec:
        raise typer.BadParameter(
            f"Model spec '{spec}' must be in the form name=kind:lora,checkpoint:path"
        )
    label, payload = (part.strip() for part in spec.split("=", 1))
    if not label or not payload:
        raise typer.BadParameter(f"Malformed model spec '{spec}'.")

    entry: Dict[str, Any] = {}
    segments = [segment.strip() for segment in payload.split(",") if segment.strip()]
    if not segments:
        raise typer.BadParameter(f"Model spec '{spec}' must define at least one key/value pair.")

    default_kind = "lora"
    if len(segments) == 1 and ":" not in segments[0]:
        entry["kind"] = default_kind
        entry["checkpoint"] = segments[0]
    else:
        for segment in segments:
            if ":" not in segment:
                raise typer.BadParameter(
                    f"Model spec segment '{segment}' must be key:value (from '{spec}')."
                )
            key, value = (part.strip() for part in segment.split(":", 1))
            if not key:
                raise typer.BadParameter(f"Model spec '{spec}' has an empty key.")
            if not value:
                raise typer.BadParameter(f"Model spec '{spec}' missing value for key '{key}'.")
            entry[key] = value
        entry.setdefault("kind", default_kind)

    kind = entry.get("kind")
    if kind == "lora":
        if "checkpoint" not in entry:
            raise typer.BadParameter(f"Model '{label}' of kind 'lora' requires 'checkpoint'.")
    elif kind == "hf":
        if "model" not in entry:
            raise typer.BadParameter(f"Model '{label}' of kind 'hf' requires 'model'.")
    elif kind == "base":
        # Base models optionally supply identifier
        pass
    else:
        raise typer.BadParameter(
            f"Unsupported model kind '{kind}' for '{label}'. Expected one of lora, hf, base."
        )
    return label, entry


def _validate_comparisons_models(cfg: Dict[str, Any]) -> None:
    models = cfg.get("models") or {}
    if not models:
        raise typer.BadParameter("Evaluation config must define at least one model.")
    model_names = set(models.keys())
    missing: List[str] = []
    for comparison in cfg.get("comparisons", []):
        comp_name = comparison.get("name", "comparison")
        for side in ("a", "b"):
            model_ref = comparison.get(side)
            if model_ref not in model_names:
                missing.append(f"{comp_name}:{side} -> '{model_ref}'")
    if missing:
        missing_list = ", ".join(missing)
        raise typer.BadParameter(
            f"Comparison references undefined models: {missing_list}. "
            "Ensure --model specs include every referenced model."
        )


def _run_training(config_path: Path) -> None:
    typer.echo(f"[orchestrate] Training with config {config_path}")
    start = time.time()
    subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        check=True,
        cwd=_REPO_ROOT,
    )
    typer.echo(f"[orchestrate] Training finished in {time.time() - start:.1f}s")


def _collect_train_artifacts(output_dir: str, dest_dir: Path) -> None:
    output_path = Path(output_dir)
    if not output_path.exists():
        typer.echo(f"[orchestrate] Warning: output dir {output_path} does not exist.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stats_file in output_path.rglob("train_stats.json"):
        rel = stats_file.relative_to(output_path)
        dst = dest_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stats_file, dst)
    summary = output_path / "train_summary.json"
    if summary.exists():
        shutil.copy2(summary, dest_dir / summary.name)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_eval_prompts(eval_cfg: Dict, dataset_config: Path, prompt_count: int) -> Path:
    prompts_cfg = eval_cfg.get("prompts") or {}
    prompt_path = prompts_cfg.get("path")
    if not prompt_path:
        raise typer.BadParameter("Evaluation config must define 'prompts.path'.")
    split = prompts_cfg.get("split", "eval")
    prompt_abs_path = _resolve_path(Path(prompt_path))
    prompt_abs_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[orchestrate] Preparing evaluation prompts ({prompt_count}) at {prompt_abs_path} (split='{split}')"
    )
    cmd = [
        sys.executable,
        "scripts/prepare_dev_set.py",
        "--config",
        str(dataset_config),
        "--size",
        str(prompt_count),
        "--split",
        split,
        "--out",
        str(prompt_abs_path),
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return prompt_abs_path


def _phase_output_dir(phase: str) -> Path:
    dir_path = RESULTS_ROOT / phase
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


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
    """Run Phase 1 brittle β grid (SFT baseline + fixed β sweep)."""
    base_cfg = _load_yaml(base_config)
    phase_dir = _phase_output_dir("phase1_fixed_beta_grid")

    for beta in betas:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg["fixed_beta"] = float(beta)
        output_dir = Path(cfg["trainer"].get("output_dir", "outputs/fixed_beta"))
        run_dir = output_dir / f"beta_{_slug(f'{beta:.3f}')}"
        cfg["trainer"]["output_dir"] = str(run_dir)
        cfg["seed"] = cfg.get("seed", 42)
        cfg["seeds"] = cfg.get("seeds", [cfg["seed"]])

        config_path = _write_config(cfg, f"phase1_beta_{_slug(f'{beta:.3f}')}.yaml")
        _run_training(config_path)
        _collect_train_artifacts(str(_resolve_path(run_dir)), phase_dir / f"beta_{_slug(f'{beta:.3f}')}")


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
    """Run Phase 2 adaptive vs baselines and evaluate."""
    phase_dir = _phase_output_dir("phase2_adaptive_vs_baselines")

    configs = [
        ("adaptive", adaptive_config),
        ("annealed", annealed_config),
    ]
    if include_fixed_beta:
        configs.append(("fixed", fixed_config))

    run_outputs: Dict[str, Path] = {}
    config_map: Dict[str, Dict[str, Any]] = {}

    for label, path in configs:
        cfg = _load_yaml(path)
        trainer_cfg = cfg.setdefault("trainer", {})
        run_output = Path(trainer_cfg.get("output_dir", f"outputs/{label}"))

        if label == "fixed" and oracle_beta is not None:
            beta_value = float(oracle_beta)
            cfg["fixed_beta"] = beta_value
            beta_slug = _slug(f"{beta_value:.3f}")
            run_output = run_output / f"beta_{beta_slug}"
            trainer_cfg["output_dir"] = str(run_output)
            existing_name = trainer_cfg.get("run_name")
            trainer_cfg["run_name"] = (existing_name + f"_beta_{beta_slug}") if existing_name else f"phase2_fixed_beta_{beta_slug}"

        if len(cfg.get("seeds", [])) <= 1:
            cfg["seeds"] = cfg.get("seeds", [cfg.get("seed", 42)])
        config_path = _write_config(cfg, f"phase2_{label}.yaml")
        _run_training(config_path)
        _collect_train_artifacts(str(_resolve_path(run_output)), phase_dir / f"train_{label}")
        run_outputs[label] = _resolve_path(run_output)
        config_map[label] = cfg

    typer.echo("[orchestrate] Running evaluation for Phase 2")
    eval_output_dir = phase_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = _load_yaml(eval_config)
    _prepare_eval_prompts(eval_cfg, adaptive_config, eval_prompt_size)
    eval_cmd = [
        sys.executable,
        "scripts/eval.py",
        "--config",
        str(eval_config),
    ]
    eval_cmd.extend(["--limit", str(eval_prompt_size)])
    if force_eval:
        eval_cmd.append("--force-judge")
    subprocess.run(eval_cmd, check=True, cwd=_REPO_ROOT)
    # Copy metrics into phase results
    eval_results_root = eval_cfg.get("output", {}).get("dir", "research/results/eval")
    metrics_dir = _resolve_path(Path(eval_results_root)) / "metrics"
    if metrics_dir.exists():
        shutil.copytree(metrics_dir, eval_output_dir / "metrics", dirs_exist_ok=True)

    adaptive_output = run_outputs.get("adaptive")
    adaptive_cfg = config_map.get("adaptive")
    if adaptive_output and adaptive_cfg:
        trainer_cfg = adaptive_cfg.get("trainer", {})
        per_device = int(trainer_cfg.get("per_device_train_batch_size", 1))
        grad_accum = int(trainer_cfg.get("gradient_accumulation_steps", 1))
        effective_batch = per_device * grad_accum
        seed = int(adaptive_cfg.get("seed") or trainer_cfg.get("seed", 42))

        phase_trace_path = adaptive_output / "phase_trace.json"
        audit_cmd = [
            sys.executable,
            "scripts/poison_audit.py",
            "--config",
            str(adaptive_config),
            "--model-dir",
            str(adaptive_output),
            "--batch-index",
            str(audit_batch_index),
            "--batch-size",
            str(effective_batch),
            "--seed",
            str(seed),
            "--phase-trace",
            str(phase_trace_path),
        ]
        if audit_wandb_project:
            audit_cmd.extend(["--wandb-project", audit_wandb_project])
            audit_cmd.extend(["--wandb-name", f"phase2_poison_audit_seed{seed}"])

        typer.echo("[orchestrate] Running poison audit for Phase 2 adaptive controller")
        subprocess.run(audit_cmd, check=True, cwd=_REPO_ROOT)


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
    """Run Phase 3 controller ablations (EMA, deadband, clipping, entropy gain)."""
    base_cfg = _load_yaml(base_config)
    phase_dir = _phase_output_dir("phase3_ablation")

    variant_overrides = {
        "no_deadband": {"deadband": 0.0},
        "no_ema": {"ema_alpha": 1.0},
        "no_clipping": {"beta_min": 0.0, "beta_max": 10000.0},
        "no_fast_loop": {"lambda_entropy": 0.0},
    }

    for variant in ablations:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        overrides = cfg.setdefault("beta_controller", {})
        label = variant
        if variant not in variant_overrides and variant != "full":
            raise typer.BadParameter(f"Unknown ablation variant '{variant}'.")

        variant_values = variant_overrides.get(variant)
        if variant_values:
            overrides.update(variant_values)

        run_output = Path(cfg["trainer"].get("output_dir", "outputs/adaptive_beta")) / f"ablation_{label}"
        cfg["trainer"]["output_dir"] = str(run_output)
        cfg["seed"] = cfg.get("seed", 42)
        cfg["seeds"] = cfg.get("seeds", [cfg["seed"]])

        config_path = _write_config(cfg, f"phase3_{label}.yaml")
        _run_training(config_path)
        _collect_train_artifacts(str(_resolve_path(run_output)), phase_dir / f"ablation_{label}")


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
    """Run Phase 4 generalization sweep across dataset/model combinations."""
    base_cfg = _load_yaml(eval_config)
    phase_dir = _phase_output_dir("phase4_generalization")

    dataset_specs = [_parse_dataset_spec(spec) for spec in datasets]
    if not dataset_specs:
        raise typer.BadParameter("At least one --dataset entry is required for Phase 4.")

    if models:
        model_map = {}
        for spec in models:
            label, entry = _parse_model_spec(spec)
            model_map[label] = entry
    else:
        model_map = base_cfg.get("models") or {}

    if not model_map:
        raise typer.BadParameter(
            "No models defined. Provide --model entries or ensure the base eval config defines models."
        )

    for dataset_label, dataset_path in dataset_specs:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        cfg["models"] = yaml.safe_load(yaml.dump(model_map))
        prompts_cfg = cfg.setdefault("prompts", {})
        prompts_cfg["path"] = dataset_path

        dataset_slug = _slug(dataset_label)
        dataset_phase_dir = phase_dir / dataset_slug
        dataset_phase_dir.mkdir(parents=True, exist_ok=True)
        eval_output_dir = dataset_phase_dir / "evaluation"
        cfg.setdefault("output", {})
        cfg["output"]["dir"] = str(eval_output_dir)

        _validate_comparisons_models(cfg)

        config_path = _write_config(cfg, f"phase4_{dataset_slug}.yaml")
        typer.echo(
            f"[orchestrate] Phase 4 evaluation for dataset '{dataset_label}' "
            f"-> config {config_path}"
        )

        cmd = [
            sys.executable,
            "scripts/eval.py",
            "--config",
            str(config_path),
        ]
        if force_eval:
            cmd.append("--force-judge")
        subprocess.run(cmd, check=True, cwd=_REPO_ROOT)

        metrics_dir = eval_output_dir / "metrics"
        if metrics_dir.exists():
            dest_metrics = dataset_phase_dir / "metrics"
            shutil.copytree(metrics_dir, dest_metrics, dirs_exist_ok=True)


if __name__ == "__main__":
    app()

