from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = _REPO_ROOT / "research" / "results"
CONFIG_CACHE = RESULTS_ROOT / "configs"


@dataclass
class PhaseTrainingJob:
    label: str
    config: Dict[str, Any]
    output_dir: Path
    config_filename: str


@dataclass
class DatasetPromptOptions:
    size: int
    split: str
    tokenizer: str


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


def _run_training_job(job: PhaseTrainingJob, artifact_dest: Path) -> Path:
    config_path = _write_config(job.config, job.config_filename)
    _run_training(config_path)
    _collect_train_artifacts(str(_resolve_path(job.output_dir)), artifact_dest)
    return config_path


def _relativize(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _prepare_dataset_prompts_from_alias(label: str, alias: str, options: DatasetPromptOptions) -> Path:
    dest_dir = RESULTS_ROOT / "phase4_datasets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{_slug(label)}.jsonl"
    cmd = [
        sys.executable,
        "scripts/prepare_dev_set.py",
        "--dataset",
        alias,
        "--size",
        str(options.size),
        "--split",
        options.split,
        "--tokenizer-name",
        options.tokenizer,
        "--out",
        str(dest_path),
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return dest_path


def _prepare_dataset_prompts_from_config(label: str, config_path: Path, options: DatasetPromptOptions) -> Path:
    dest_dir = RESULTS_ROOT / "phase4_datasets"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{_slug(label)}.jsonl"
    resolved_config = config_path if config_path.is_absolute() else (_REPO_ROOT / config_path)
    cmd = [
        sys.executable,
        "scripts/prepare_dev_set.py",
        "--config",
        str(resolved_config),
        "--size",
        str(options.size),
        "--split",
        options.split,
        "--tokenizer-name",
        options.tokenizer,
        "--out",
        str(dest_path),
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    return dest_path


def _parse_dataset_spec(spec: str, prompt_options: Optional[DatasetPromptOptions] = None) -> Tuple[str, str]:
    if "=" not in spec:
        raise typer.BadParameter(
            f"Dataset spec '{spec}' must be in the form name=path/to/prompts.jsonl."
        )
    label, path_value = (part.strip() for part in spec.split("=", 1))
    if not label:
        raise typer.BadParameter(f"Dataset spec '{spec}' is missing a name before '='.")
    if not path_value:
        raise typer.BadParameter(f"Dataset spec '{spec}' is missing a path after '='.")
    if path_value.startswith("alias:"):
        if prompt_options is None:
            raise typer.BadParameter(
                f"Dataset spec '{spec}' uses alias:, which requires prompt preparation options."
            )
        alias = path_value.split(":", 1)[1]
        prepared = _prepare_dataset_prompts_from_alias(label, alias, prompt_options)
        return label, _relativize(prepared)
    if path_value.startswith("config:"):
        if prompt_options is None:
            raise typer.BadParameter(
                f"Dataset spec '{spec}' uses config:, which requires prompt preparation options."
            )
        config_path = Path(path_value.split(":", 1)[1])
        prepared = _prepare_dataset_prompts_from_config(label, config_path, prompt_options)
        return label, _relativize(prepared)
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


def run_phase1(
    base_config: Path,
    betas: List[float],
    eval_config: Optional[Path],
    eval_prompt_size: int,
    eval_model_slot: str,
    selection_comparison: str,
    selection_judge: str,
    force_eval: bool,
    results_path: Optional[Path],
    run_eval: bool,
) -> None:
    base_cfg = _load_yaml(base_config)
    phase_dir = _phase_output_dir("phase1_fixed_beta_grid")
    evaluation_enabled = run_eval and eval_config is not None
    eval_cfg_template: Optional[Dict[str, Any]] = None
    eval_records: List[Dict[str, Any]] = []
    results_destination = results_path
    if results_destination is None:
        results_destination = phase_dir / "phase1_results.json"
    elif not results_destination.is_absolute():
        results_destination = _REPO_ROOT / results_destination

    if evaluation_enabled:
        eval_cfg_template = _load_yaml(eval_config)  # type: ignore[arg-type]
        prompts_cfg = eval_cfg_template.setdefault("prompts", {})
        if eval_prompt_size > 0:
            prompts_cfg["limit"] = eval_prompt_size
        _prepare_eval_prompts(eval_cfg_template, base_config, eval_prompt_size)

    for beta in betas:
        cfg = copy.deepcopy(base_cfg)
        cfg["fixed_beta"] = float(beta)
        output_dir = Path(cfg["trainer"].get("output_dir", "outputs/fixed_beta"))
        run_dir = output_dir / f"beta_{_slug(f'{beta:.3f}')}"
        cfg["trainer"]["output_dir"] = str(run_dir)
        cfg["seed"] = cfg.get("seed", 42)
        cfg["seeds"] = cfg.get("seeds", [cfg["seed"]])

        job = PhaseTrainingJob(
            label=f"beta_{_slug(f'{beta:.3f}')}",
            config=cfg,
            output_dir=run_dir,
            config_filename=f"phase1_beta_{_slug(f'{beta:.3f}')}.yaml",
        )
        _run_training_job(job, phase_dir / job.label)

        if eval_cfg_template:
            eval_cfg = copy.deepcopy(eval_cfg_template)
            eval_output_dir = phase_dir / "evaluation" / job.label
            eval_cfg.setdefault("output", {})
            eval_cfg["output"]["dir"] = str(eval_output_dir)
            models = eval_cfg.get("models") or {}
            if eval_model_slot not in models:
                raise typer.BadParameter(
                    f"Eval config is missing model slot '{eval_model_slot}' required for Phase 1."
                )
            candidate_path = _resolve_path(job.output_dir)
            models[eval_model_slot]["checkpoint"] = str(candidate_path)
            _validate_comparisons_models(eval_cfg)
            eval_cfg_path = _write_config(eval_cfg, f"phase1_eval_{job.label}.yaml")
            eval_cmd = [
                sys.executable,
                "scripts/eval.py",
                "all-judges",
                "--config",
                str(eval_cfg_path),
            ]
            if eval_prompt_size > 0:
                eval_cmd.extend(["--limit", str(eval_prompt_size)])
            if force_eval:
                eval_cmd.append("--force-judge")
            typer.echo(f"[orchestrate] Evaluating Phase 1 beta={beta}")
            subprocess.run(eval_cmd, check=True, cwd=_REPO_ROOT)

            metrics_path = eval_output_dir / "metrics" / "summary.json"
            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"Expected metrics summary at {metrics_path} for beta={beta}."
                )
            with metrics_path.open("r", encoding="utf-8") as metrics_file:
                metrics_summary = json.load(metrics_file)

            comparison_metrics = metrics_summary.get(selection_comparison)
            if comparison_metrics is None:
                raise typer.BadParameter(
                    f"Comparison '{selection_comparison}' not found in eval metrics for beta={beta}."
                )
            judge_metrics = comparison_metrics.get(selection_judge)
            if judge_metrics is None:
                raise typer.BadParameter(
                    f"Judge '{selection_judge}' not found in eval metrics for beta={beta}."
                )

            record = {
                "beta": float(beta),
                "label": job.label,
                "win_rate": float(judge_metrics.get("win_rate", 0.0)),
                "wins": int(judge_metrics.get("wins", 0)),
                "total": int(judge_metrics.get("total", 0)),
                "comparison": selection_comparison,
                "judge": selection_judge,
                "metrics_file": _relativize(metrics_path),
                "config_file": _relativize(eval_cfg_path),
                "judges": comparison_metrics,
            }
            eval_records.append(record)

    if eval_records:
        eval_records.sort(key=lambda entry: entry["beta"])
        best_record = max(eval_records, key=lambda entry: entry["win_rate"])
        results_payload = {
            "selection_comparison": selection_comparison,
            "selection_judge": selection_judge,
            "best_beta": best_record["beta"],
            "best_record": best_record,
            "records": eval_records,
        }
        results_destination.parent.mkdir(parents=True, exist_ok=True)
        with results_destination.open("w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        typer.echo(
            "[orchestrate] Phase 1 best beta={beta:.3f} (win_rate={rate:.3f}, judge={judge}, comparison={comparison}) "
            "-> results stored at {path}".format(
                beta=best_record["beta"],
                rate=best_record["win_rate"],
                judge=selection_judge,
                comparison=selection_comparison,
                path=_relativize(results_destination),
            )
        )


def run_phase2(
    adaptive_config: Path,
    annealed_config: Path,
    fixed_config: Path,
    eval_config: Path,
    include_fixed_beta: bool,
    force_eval: bool,
    oracle_beta: Optional[float],
    audit_batch_index: int,
    audit_wandb_project: Optional[str],
    eval_prompt_size: int,
    run_diagnostics: bool,
    diagnostics_dir: Optional[Path],
    report_output: Path,
) -> None:
    phase_dir = _phase_output_dir("phase2_adaptive_vs_baselines")

    configs = [
        ("adaptive", adaptive_config),
        ("annealed", annealed_config),
    ]
    if include_fixed_beta:
        configs.append(("fixed", fixed_config))

    run_outputs: Dict[str, Path] = {}
    config_map: Dict[str, Dict[str, Any]] = {}

    training_jobs: List[PhaseTrainingJob] = []
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
            trainer_cfg["run_name"] = (
                existing_name + f"_beta_{beta_slug}" if existing_name else f"phase2_fixed_beta_{beta_slug}"
            )

        if len(cfg.get("seeds", [])) <= 1:
            cfg["seeds"] = cfg.get("seeds", [cfg.get("seed", 42)])

        training_jobs.append(
            PhaseTrainingJob(
                label=label,
                config=cfg,
                output_dir=run_output,
                config_filename=f"phase2_{label}.yaml",
            )
        )

    for job in training_jobs:
        _run_training_job(job, phase_dir / f"train_{job.label}")
        run_outputs[job.label] = _resolve_path(job.output_dir)
        config_map[job.label] = job.config

    typer.echo("[orchestrate] Running evaluation for Phase 2")
    eval_output_dir = phase_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = _load_yaml(eval_config)
    _prepare_eval_prompts(eval_cfg, adaptive_config, eval_prompt_size)
    eval_cmd = [
        sys.executable,
        "scripts/eval.py",
        "all-judges",
        "--config",
        str(eval_config),
    ]
    eval_cmd.extend(["--limit", str(eval_prompt_size)])
    if force_eval:
        eval_cmd.append("--force-judge")
    subprocess.run(eval_cmd, check=True, cwd=_REPO_ROOT)
    eval_results_root = eval_cfg.get("output", {}).get("dir", "research/results/eval")
    eval_results_root_path = _resolve_path(Path(eval_results_root))
    metrics_dir = eval_results_root_path / "metrics"
    if metrics_dir.exists():
        shutil.copytree(metrics_dir, eval_output_dir / "metrics", dirs_exist_ok=True)

    adaptive_output = run_outputs.get("adaptive")
    adaptive_cfg = config_map.get("adaptive")
    diagnostics_path: Optional[Path] = None
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

    if run_diagnostics:
        diagnostics_path = diagnostics_dir
        if diagnostics_path is None:
            diagnostics_path = phase_dir / "diagnostics"
        elif not diagnostics_path.is_absolute():
            diagnostics_path = _REPO_ROOT / diagnostics_path
        diagnostics_path.mkdir(parents=True, exist_ok=True)
        controller_dir = diagnostics_path / "controller_plots"
        controller_dir.mkdir(parents=True, exist_ok=True)
        phase_trace_file = adaptive_output / "phase_trace.json" if adaptive_output else None

        diag_cmd = [
            sys.executable,
            "scripts/run_eval_with_diagnostics.py",
            "--eval-subcommand",
            "all-judges",
            "--eval-config",
            str(eval_config),
            "--limit",
            str(eval_prompt_size),
            "--skip-eval",
            "--phase-output-dir",
            str(controller_dir),
            "--phase-run-label",
            "phase2_adaptive",
            "--summary-output",
            str(diagnostics_path / "eval_summary.csv"),
            "--entropy-output",
            str(diagnostics_path / "entropy_bucket_summary.json"),
            "--fliprate-output",
            str(diagnostics_path / "fliprate_summary.json"),
            "--fliprate-plot",
            str(diagnostics_path / "fliprate_plot.png"),
        ]
        if phase_trace_file and phase_trace_file.exists():
            diag_cmd.extend(["--phase-trace", str(phase_trace_file)])
        subprocess.run(diag_cmd, check=True, cwd=_REPO_ROOT)

    resolved_report = report_output if report_output.is_absolute() else _REPO_ROOT / report_output
    report_cmd = [
        sys.executable,
        "scripts/report_phase2.py",
        "--phase-dir",
        str(phase_dir),
        "--eval-output",
        str(eval_results_root_path),
        "--output",
        str(resolved_report),
    ]
    if diagnostics_path:
        report_cmd.extend(["--diagnostics-dir", str(diagnostics_path)])
    subprocess.run(report_cmd, check=True, cwd=_REPO_ROOT)


def run_phase3(
    base_config: Path,
    ablations: List[str],
    eval_config: Optional[Path],
    eval_prompt_size: int,
    eval_model_slot: str,
    force_eval: bool,
    plot_phase_traces: bool,
) -> None:
    base_cfg = _load_yaml(base_config)
    phase_dir = _phase_output_dir("phase3_ablation")
    eval_cfg_template: Optional[Dict[str, Any]] = None
    aggregate_metrics: Dict[str, Dict[str, Any]] = {}

    if eval_config:
        eval_cfg_template = _load_yaml(eval_config)
        prompts_cfg = eval_cfg_template.setdefault("prompts", {})
        if eval_prompt_size > 0:
            prompts_cfg["limit"] = eval_prompt_size
        _prepare_eval_prompts(eval_cfg_template, base_config, eval_prompt_size)

    variant_overrides = {
        "no_deadband": {"deadband": 0.0},
        "no_ema": {"ema_alpha": 1.0},
        "no_clipping": {"beta_min": 0.0, "beta_max": 10000.0},
        "no_fast_loop": {"lambda_entropy": 0.0},
    }

    ablation_variants = list(ablations)
    with typer.progressbar(
        ablation_variants,
        label="Phase 3 ablations",
        item_show_func=lambda variant: str(variant),
    ) as progress_iter:
        for variant in progress_iter:
            cfg = copy.deepcopy(base_cfg)
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

            job = PhaseTrainingJob(
                label=f"ablation_{label}",
                config=cfg,
                output_dir=run_output,
                config_filename=f"phase3_{label}.yaml",
            )
            _run_training_job(job, phase_dir / job.label)

            run_output_path = _resolve_path(job.output_dir)

            if plot_phase_traces:
                phase_trace_path = run_output_path / "phase_trace.json"
                if phase_trace_path.exists():
                    trace_dir = phase_dir / "phase_traces" / job.label
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace_cmd = [
                        sys.executable,
                        "scripts/plot_phase_trace.py",
                        "--phase-trace",
                        str(phase_trace_path),
                        "--output-dir",
                        str(trace_dir),
                        "--run-label",
                        job.label,
                    ]
                    subprocess.run(trace_cmd, check=True, cwd=_REPO_ROOT)

            if eval_cfg_template:
                eval_cfg = copy.deepcopy(eval_cfg_template)
                eval_output_dir = phase_dir / "evaluation" / job.label
                eval_cfg.setdefault("output", {})
                eval_cfg["output"]["dir"] = str(eval_output_dir)
                models = eval_cfg.get("models") or {}
                if eval_model_slot not in models:
                    raise typer.BadParameter(
                        f"Eval config is missing model slot '{eval_model_slot}' required for Phase 3."
                    )
                models[eval_model_slot]["checkpoint"] = str(run_output_path)
                _validate_comparisons_models(eval_cfg)
                eval_cfg_path = _write_config(eval_cfg, f"phase3_eval_{job.label}.yaml")
                eval_cmd = [
                    sys.executable,
                    "scripts/eval.py",
                    "all-judges",
                    "--config",
                    str(eval_cfg_path),
                ]
                if eval_prompt_size > 0:
                    eval_cmd.extend(["--limit", str(eval_prompt_size)])
                if force_eval:
                    eval_cmd.append("--force-judge")
                subprocess.run(eval_cmd, check=True, cwd=_REPO_ROOT)

                metrics_path = eval_output_dir / "metrics" / "summary.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(
                        f"Missing evaluation metrics for {job.label} at {metrics_path}."
                    )
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics_summary = json.load(f)
                for comparison, judge_block in metrics_summary.items():
                    agg_entry = aggregate_metrics.setdefault(comparison, {})
                    agg_entry.update(judge_block)

    if aggregate_metrics:
        metrics_root = phase_dir / "metrics"
        metrics_root.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_root / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(aggregate_metrics, f, indent=2)

        bar_output = phase_dir / "ablation_win_rates.png"
        comparisons_arg = ",".join(sorted(aggregate_metrics.keys()))
        bar_cmd = [
            sys.executable,
            "scripts/plot_ablation_bar.py",
            "--metrics",
            str(summary_path),
            "--output",
            str(bar_output),
            "--comparisons",
            comparisons_arg,
        ]
        subprocess.run(bar_cmd, check=True, cwd=_REPO_ROOT)


def run_phase4(
    eval_config: Path,
    datasets: List[str],
    models: List[str],
    force_eval: bool,
    dataset_prompt_size: int,
    dataset_split: str,
    dataset_tokenizer: str,
) -> None:
    base_cfg = _load_yaml(eval_config)
    phase_dir = _phase_output_dir("phase4_generalization")

    prompt_options = DatasetPromptOptions(
        size=dataset_prompt_size,
        split=dataset_split,
        tokenizer=dataset_tokenizer,
    )
    dataset_specs = [_parse_dataset_spec(spec, prompt_options) for spec in datasets]
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

    dataset_metrics: Dict[str, Dict[str, Any]] = {}

    dataset_entries = list(dataset_specs)
    with typer.progressbar(
        dataset_entries,
        label="Phase 4 datasets",
        item_show_func=lambda item: str(item[0]),
    ) as progress_iter:
        for dataset_label, dataset_path in progress_iter:
            cfg = copy.deepcopy(base_cfg)
            cfg["models"] = copy.deepcopy(model_map)
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
                summary_path = dest_metrics / "summary.json"
                if summary_path.exists():
                    with summary_path.open("r", encoding="utf-8") as f:
                        dataset_metrics[dataset_label] = json.load(f)

    if dataset_metrics:
        summary_root = phase_dir / "summary.json"
        with summary_root.open("w", encoding="utf-8") as f:
            json.dump(dataset_metrics, f, indent=2)


__all__ = [
    "run_phase1",
    "run_phase2",
    "run_phase3",
    "run_phase4",
]

