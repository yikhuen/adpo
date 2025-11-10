import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
):
    """Run Phase 2 adaptive vs baselines and evaluate."""
    phase_dir = _phase_output_dir("phase2_adaptive_vs_baselines")

    configs = [
        ("adaptive", adaptive_config),
        ("annealed", annealed_config),
    ]
    if include_fixed_beta:
        configs.append(("fixed", fixed_config))

    for label, path in configs:
        cfg = _load_yaml(path)
        run_output = Path(cfg["trainer"].get("output_dir", f"outputs/{label}"))
        if len(cfg.get("seeds", [])) <= 1:
            cfg["seeds"] = cfg.get("seeds", [cfg.get("seed", 42)])
        config_path = _write_config(cfg, f"phase2_{label}.yaml")
        _run_training(config_path)
        _collect_train_artifacts(str(_resolve_path(run_output)), phase_dir / f"train_{label}")

    typer.echo("[orchestrate] Running evaluation for Phase 2")
    eval_output_dir = phase_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        sys.executable,
        "scripts/eval.py",
        "--config",
        str(eval_config),
    ]
    if force_eval:
        eval_cmd.append("--force-judge")
    subprocess.run(eval_cmd, check=True, cwd=_REPO_ROOT)
    # Copy metrics into phase results
    metrics_dir = _REPO_ROOT / "research" / "results" / "eval" / "metrics"
    if metrics_dir.exists():
        shutil.copytree(metrics_dir, eval_output_dir / "metrics", dirs_exist_ok=True)


@app.command()
def phase3(
    base_config: Path = typer.Option(
        Path("configs/train/qwen25_7b_adaptive_beta.yaml"),
        help="Base adaptive config used for ablations.",
    ),
    ablations: List[str] = typer.Option(
        ["full", "no_deadband", "no_ema", "no_clipping"],
        help="Controller variants to run.",
    ),
):
    """Run Phase 3 controller ablations (EMA, deadband, clipping)."""
    base_cfg = _load_yaml(base_config)
    phase_dir = _phase_output_dir("phase3_ablation")

    for variant in ablations:
        cfg = yaml.safe_load(yaml.dump(base_cfg))
        overrides = cfg.setdefault("beta_controller", {})
        label = variant
        if variant == "no_deadband":
            overrides["use_deadband"] = False
        elif variant == "no_ema":
            overrides["use_ema"] = False
        elif variant == "no_clipping":
            overrides["use_clipping"] = False
        elif variant != "full":
            raise typer.BadParameter(f"Unknown ablation variant '{variant}'.")

        run_output = Path(cfg["trainer"].get("output_dir", "outputs/adaptive_beta")) / f"ablation_{label}"
        cfg["trainer"]["output_dir"] = str(run_output)
        cfg["seed"] = cfg.get("seed", 42)
        cfg["seeds"] = cfg.get("seeds", [cfg["seed"]])

        config_path = _write_config(cfg, f"phase3_{label}.yaml")
        _run_training(config_path)
        _collect_train_artifacts(str(_resolve_path(run_output)), phase_dir / f"ablation_{label}")


@app.command()
def phase4(
    eval_config: Path = typer.Option(..., help="Evaluation config covering generalization datasets."),
    force_eval: bool = typer.Option(False, help="Force re-run evaluation even if cached."),
):
    """Run Phase 4 generalization evaluation only (models assumed trained)."""
    phase_dir = _phase_output_dir("phase4_generalization")
    eval_output_dir = phase_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--config",
        str(eval_config),
    ]
    if force_eval:
        cmd.append("--force-judge")
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)
    metrics_dir = _REPO_ROOT / "research" / "results" / "eval" / "metrics"
    if metrics_dir.exists():
        shutil.copytree(metrics_dir, eval_output_dir / "metrics", dirs_exist_ok=True)


if __name__ == "__main__":
    app()

