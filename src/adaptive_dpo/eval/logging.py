from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer


def log_results_to_wandb(
    cfg: Dict[str, Any],
    metrics_summary: Dict[str, Dict[str, Any]],
    summary_path: Path,
    csv_path: Path,
    prompt_count: int,
    config_path: str,
    model_stats: Dict[str, Dict[str, float]],
    model_stats_path: Optional[Path],
    reward_hacking_report: Optional[Dict[str, Any]],
    reward_hacking_path: Optional[Path],
    qualitative_samples_by_model: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    output_dir: Optional[Path] = None,
):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled"):
        return

    try:
        import wandb
    except ImportError:
        typer.echo("[eval] wandb is not installed; skipping wandb logging.")
        return

    init_kwargs: Dict[str, Any] = {}
    for key in ("project", "entity", "group", "job_type", "name", "notes"):
        value = wandb_cfg.get(key)
        if value is not None:
            init_kwargs[key] = value
    if wandb_cfg.get("tags"):
        init_kwargs["tags"] = wandb_cfg["tags"]
    if wandb_cfg.get("settings"):
        init_kwargs["settings"] = wandb_cfg["settings"]
    if wandb_cfg.get("resume"):
        init_kwargs["resume"] = wandb_cfg["resume"]

    typer.echo("[eval] Initialising wandb run for evaluation logging.")
    run = wandb.init(**init_kwargs)

    extra_config = wandb_cfg.get("config")
    if isinstance(extra_config, dict):
        run.config.update(extra_config, allow_val_change=True)
    run.config.update(
        {
            "eval_config_path": config_path,
            "prompt_count": prompt_count,
            "comparisons": list(metrics_summary.keys()),
        },
        allow_val_change=True,
    )

    wandb_output_dir = Path(output_dir) if output_dir else None

    def _truncate_text(value: Any, limit: int) -> str:
        if value is None:
            return ""
        text = str(value)
        if limit > 0 and len(text) > limit:
            return text[:limit] + "…"
        return text

    def _safe_path(path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(path_value)
        return path if path.exists() else None

    def _safe_add_path(artifact: "wandb.Artifact", path: Path, *, name: Optional[str] = None) -> None:
        if not path.exists():
            typer.echo(f"[eval] Attachment missing, skipping: {path}")
            return
        if path.is_dir():
            artifact.add_dir(str(path), name=name or path.name)
        else:
            artifact.add_file(str(path), name=name or path.name)

    def _log_json_table(table_key: str, json_path: Path) -> None:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            typer.echo(f"[eval] Unable to read JSON summary at {json_path}")
            return
        if not isinstance(data, list) or not data:
            return
        columns = sorted({key for row in data if isinstance(row, dict) for key in row.keys()})
        if not columns:
            return
        table = wandb.Table(columns=columns)
        for row in data:
            table.add_data(*[row.get(col) for col in columns])
        run.log({table_key: table}, commit=False)

    if wandb_cfg.get("log_table", True):
        table = wandb.Table(
            columns=[
                "comparison",
                "judge",
                "wins",
                "total",
                "win_rate",
                "ci95_low",
                "ci95_high",
                "model_a",
                "model_b",
            ]
        )
        for comparison_name, judge_metrics in metrics_summary.items():
            for judge_name, metric_entry in judge_metrics.items():
                ci_low, ci_high = metric_entry["ci95"]
                table.add_data(
                    comparison_name,
                    judge_name,
                    metric_entry["wins"],
                    metric_entry["total"],
                    metric_entry["win_rate"],
                    ci_low,
                    ci_high,
                    metric_entry["model_a"],
                    metric_entry["model_b"],
                )
        run.log({"eval/summary_table": table}, commit=False)

    if model_stats:
        stats_table = wandb.Table(
            columns=["model", "avg_length_chars", "refusal_rate", "safety_rate", "responses"]
        )
        model_names = []
        avg_lengths = []
        for model_name, stats in model_stats.items():
            stats_table.add_data(
                model_name,
                stats.get("avg_length_chars", float("nan")),
                stats.get("refusal_rate", float("nan")),
                stats.get("safety_rate", float("nan")),
                int(stats.get("count", 0)),
            )
            model_names.append(model_name)
            avg_lengths.append(stats.get("avg_length_chars", float("nan")))
        run.log({"eval/model_stats_table": stats_table}, commit=False)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            typer.echo("[eval] matplotlib not installed; skipping response length bar chart.")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(model_names, avg_lengths, color="#55A868")
            ax.set_ylabel("Avg Response Length (chars)")
            ax.set_title("Response Length Sanity Check")
            ax.set_xticklabels(model_names, rotation=20)
            fig.tight_layout()
            run.log({"eval/avg_response_length": wandb.Image(fig)}, commit=False)
            plt.close(fig)

    if reward_hacking_report:
        rh_table = wandb.Table(
            columns=[
                "model",
                "status",
                "issues",
                "avg_length_chars",
                "refusal_rate",
                "safety_rate",
                "responses",
            ]
        )
        for model_name, info in reward_hacking_report.get("per_model", {}).items():
            metrics = info.get("metrics", {})
            issues = info.get("issues") or []
            rh_table.add_data(
                model_name,
                info.get("status", "unknown"),
                "; ".join(issues) if issues else "",
                metrics.get("avg_length_chars", float("nan")),
                metrics.get("refusal_rate", float("nan")),
                metrics.get("safety_rate", float("nan")),
                metrics.get("responses", 0),
            )
        run.log({"eval/reward_hacking_checks": rh_table}, commit=False)
        run.summary["reward_hacking_status"] = reward_hacking_report.get("overall_status", "unknown")
        if reward_hacking_report.get("alerts"):
            run.summary["reward_hacking_alerts"] = reward_hacking_report["alerts"]
        run.config.update({"reward_hacking_thresholds": reward_hacking_report.get("thresholds", {})}, allow_val_change=True)

    if qualitative_samples_by_model:
        for model_name, rows in qualitative_samples_by_model.items():
            if not rows:
                continue
            table = wandb.Table(
                columns=[
                    "comparison",
                    "judge",
                    "result",
                    "prompt_id",
                    "prompt",
                    "response",
                    "opponent_model",
                    "opponent_response",
                    "position",
                    "judge_choice",
                ]
            )
            for row in rows:
                table.add_data(
                    row.get("comparison"),
                    row.get("judge"),
                    row.get("result"),
                    row.get("prompt_id"),
                    row.get("prompt"),
                    row.get("response"),
                    row.get("opponent_model"),
                    row.get("opponent_response"),
                    row.get("position"),
                    row.get("choice"),
                )
            run.log({f"qualitative/{model_name}": table}, commit=False)

    bar_cfg = wandb_cfg.get("bar_chart", {})
    if bar_cfg.get("enabled"):
        try:
            from plot_ablation_bar import build_ablation_bar_figure
        except ImportError:
            typer.echo("[eval] plot_ablation_bar helper unavailable; skipping wandb bar chart logging.")
        else:
            comparisons = bar_cfg.get("comparisons") or list(metrics_summary.keys())
            judge = bar_cfg.get("judge")
            if not judge and metrics_summary:
                first_metrics = next(iter(metrics_summary.values()))
                judge = next(iter(first_metrics.keys()), None)
            if not judge:
                typer.echo("[eval] No judge specified for wandb bar chart; skipping.")
            else:
                try:
                    fig = build_ablation_bar_figure(metrics_summary, comparisons, judge)
                except (KeyError, ValueError) as exc:
                    typer.echo(f"[eval] Failed to build bar chart for wandb: {exc}")
                else:
                    try:
                        import matplotlib.pyplot as plt
                    except ImportError:
                        typer.echo("[eval] matplotlib not available; skipping bar chart logging.")
                    else:
                        chart_name = bar_cfg.get("name", "eval/ablation_bar")
                        run.log({chart_name: wandb.Image(fig)}, commit=False)
                        plt.close(fig)

    matrix_cfg = wandb_cfg.get("matrix", {})
    if matrix_cfg.get("enabled"):
        cell_mapping = matrix_cfg.get("cell_mapping")
        if not isinstance(cell_mapping, dict):
            typer.echo("[eval] wandb matrix logging requires 'cell_mapping'; skipping.")
        else:
            judge = matrix_cfg.get("judge")
            if not judge and metrics_summary:
                first_metrics = next(iter(metrics_summary.values()))
                judge = next(iter(first_metrics.keys()), None)
            if not judge:
                typer.echo("[eval] No judge specified for wandb matrix; skipping.")
            else:
                rows = matrix_cfg.get("rows") or sorted(
                    {cell.get("row") for cell in cell_mapping.values() if isinstance(cell, dict) and "row" in cell}
                )
                cols = matrix_cfg.get("cols") or sorted(
                    {cell.get("col") for cell in cell_mapping.values() if isinstance(cell, dict) and "col" in cell}
                )
                if not rows or not cols:
                    typer.echo("[eval] Wandb matrix logging requires non-empty rows/cols; skipping.")
                else:
                    value_key = matrix_cfg.get("value_key", "win_rate")
                    default_val = float("nan")
                    matrix_values: Dict[Tuple[str, str], float] = {}
                    for comparison_name, cell in cell_mapping.items():
                        if (
                            comparison_name not in metrics_summary
                            or not isinstance(cell, dict)
                            or "row" not in cell
                            or "col" not in cell
                        ):
                            continue
                        metric_entry = metrics_summary[comparison_name].get(judge)
                        if not metric_entry:
                            continue
                        matrix_values[(cell["row"], cell["col"])] = float(metric_entry.get(value_key, default_val))

                    heatmap_table = wandb.Table(columns=["row", "col", value_key])
                    matrix_table = wandb.Table(columns=["row"] + list(cols))
                    for row_label in rows:
                        row_values: List[float] = []
                        for col_label in cols:
                            value = matrix_values.get((row_label, col_label), default_val)
                            heatmap_table.add_data(row_label, col_label, value)
                            row_values.append(value)
                        matrix_table.add_data(row_label, *row_values)

                    heatmap_key = matrix_cfg.get("name", "eval/generalization_heatmap")
                    run.log({heatmap_key: wandb.plot.heatmap(heatmap_table, "row", "col", value_key)}, commit=False)
                    run.log({f"{heatmap_key}_table": matrix_table}, commit=False)

    artifact_name = wandb_cfg.get("artifact_name", "eval-metrics")
    artifact_type = wandb_cfg.get("artifact_type", "evaluation")
    full_artifact_name = f"{artifact_name}-{run.id}"
    artifact = wandb.Artifact(full_artifact_name, type=artifact_type)
    artifact.add_file(str(summary_path), name="summary.json")
    artifact.add_file(str(csv_path), name="summary.csv")
    if model_stats_path is not None and model_stats:
        artifact.add_file(str(model_stats_path), name="model_stats.json")
    if reward_hacking_path is not None and reward_hacking_report:
        artifact.add_file(str(reward_hacking_path), name="reward_hacking.json")

    decision_cfg = wandb_cfg.get("decision_table", {})
    decision_paths: List[Path] = []
    if wandb_output_dir:
        decisions_dir = wandb_output_dir / "decisions"
        if decisions_dir.exists():
            decision_paths.extend(sorted(decisions_dir.glob("*.jsonl")))
    for custom in decision_cfg.get("paths", []):
        custom_path = _safe_path(custom)
        if custom_path:
            decision_paths.append(custom_path)
    deduped_paths: List[Path] = []
    seen_paths: set = set()
    for path in decision_paths:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        deduped_paths.append(path)
    decision_paths = deduped_paths

    if decision_cfg.get("enabled"):
        max_rows = int(decision_cfg.get("max_rows", 200))
        prompt_chars = int(decision_cfg.get("prompt_chars", 512))
        if decision_paths:
            table = wandb.Table(
                columns=[
                    "comparison",
                    "judge",
                    "prompt_id",
                    "prompt",
                    "response_a",
                    "response_b",
                    "choice",
                ]
            )
            rows_added = 0
            for path in decision_paths:
                stem = path.stem
                if "__" in stem:
                    judge_name, comparison_name = stem.split("__", 1)
                else:
                    judge_name, comparison_name = "unknown", stem
                try:
                    with path.open("r", encoding="utf-8") as f:
                        for line in f:
                            if rows_added >= max_rows:
                                break
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            table.add_data(
                                comparison_name,
                                judge_name,
                                record.get("id"),
                                _truncate_text(record.get("prompt"), prompt_chars),
                                _truncate_text(record.get("response_a"), prompt_chars),
                                _truncate_text(record.get("response_b"), prompt_chars),
                                record.get("choice"),
                            )
                            rows_added += 1
                except OSError:
                    continue
                if rows_added >= max_rows:
                    break
            if rows_added:
                run.log({"eval/decision_table": table}, commit=False)

    raw_decisions_cfg = wandb_cfg.get("raw_decisions", {})
    if raw_decisions_cfg.get("enabled", True) and decision_paths:
        table_name = raw_decisions_cfg.get("table_name", "eval/raw_decisions_full")
        prompt_chars = int(raw_decisions_cfg.get("prompt_chars", 1024))
        response_chars = int(raw_decisions_cfg.get("response_chars", 1024))
        raw_table = wandb.Table(
            columns=[
                "comparison",
                "judge",
                "prompt_id",
                "prompt",
                "response_a",
                "response_b",
                "choice",
            ]
        )
        raw_rows = 0
        for path in decision_paths:
            stem = path.stem
            if "__" in stem:
                judge_name, comparison_name = stem.split("__", 1)
            else:
                judge_name, comparison_name = "unknown", stem
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        record_line = line.strip()
                        if not record_line:
                            continue
                        try:
                            record = json.loads(record_line)
                        except json.JSONDecodeError:
                            continue
                        raw_table.add_data(
                            comparison_name,
                            judge_name,
                            record.get("id"),
                            _truncate_text(record.get("prompt"), prompt_chars),
                            _truncate_text(record.get("response_a"), response_chars),
                            _truncate_text(record.get("response_b"), response_chars),
                            record.get("choice"),
                        )
                        raw_rows += 1
            except OSError:
                continue
        if raw_rows:
            run.log({table_name: raw_table}, commit=False)
        artifact_prefix = raw_decisions_cfg.get("artifact_prefix", "raw_decisions")
        for path in decision_paths:
            try:
                artifact.add_file(str(path), name=f"{artifact_prefix}/{path.name}")
            except Exception:
                pass

    attachments_cfg = wandb_cfg.get("attachments", {})
    prompt_manifest_path: Optional[Path] = None
    if attachments_cfg.get("include_prompt_file"):
        prompt_file_cfg = attachments_cfg.get("prompt_file") or cfg.get("prompts", {}).get("path")
        prompt_manifest_path = _safe_path(prompt_file_cfg)
        if prompt_manifest_path:
            _safe_add_path(artifact, prompt_manifest_path, name=attachments_cfg.get("prompt_file_name") or "prompts.jsonl")
            try:
                prompt_hash = hashlib.sha256(prompt_manifest_path.read_bytes()).hexdigest()
                run.config.update({"prompt_manifest_sha256": prompt_hash}, allow_val_change=True)
            except OSError:
                typer.echo(f"[eval] Unable to hash prompt file at {prompt_manifest_path}")

    if attachments_cfg.get("include_decisions_dir") and wandb_output_dir:
        decisions_dir = wandb_output_dir / "decisions"
        if decisions_dir.exists():
            _safe_add_path(artifact, decisions_dir, name=attachments_cfg.get("decisions_name") or "decisions")

    if attachments_cfg.get("include_responses_dir") and wandb_output_dir:
        responses_dir = wandb_output_dir / "responses"
        if responses_dir.exists():
            _safe_add_path(artifact, responses_dir, name=attachments_cfg.get("responses_name") or "responses")

    if attachments_cfg.get("include_generations_dir") and wandb_output_dir:
        generations_dir = wandb_output_dir / "generations"
        if generations_dir.exists():
            _safe_add_path(artifact, generations_dir, name=attachments_cfg.get("generations_name") or "generations")

    if attachments_cfg.get("include_prompts") and wandb_output_dir:
        prompts_dir = wandb_output_dir / "prompts"
        if prompts_dir.exists():
            _safe_add_path(artifact, prompts_dir, name=attachments_cfg.get("prompts_name") or "prompts")

    for extra_entry in attachments_cfg.get("extra_files", []):
        path_value = extra_entry.get("path") if isinstance(extra_entry, dict) else extra_entry
        if not path_value:
            continue
        extra_path = _safe_path(path_value)
        if not extra_path:
            continue
        name_override = extra_entry.get("name") if isinstance(extra_entry, dict) else None
        _safe_add_path(artifact, extra_path, name=name_override)
        if isinstance(extra_entry, dict) and extra_entry.get("log_image"):
            try:
                run.log({f"eval/{name_override or extra_path.stem}": wandb.Image(str(extra_path))}, commit=False)
            except Exception:
                typer.echo(f"[eval] Failed to log image for attachment {extra_path}")

    phase_cfg = wandb_cfg.get("phase_trace", {}) or {}
    phase_json_path = _safe_path(phase_cfg.get("json"))
    if phase_json_path:
        _safe_add_path(artifact, phase_json_path, name=phase_cfg.get("json_name") or "phase_trace.json")
        try:
            phase_stats = json.loads(phase_json_path.read_text(encoding="utf-8"))
            run.summary["phase_trace_stats"] = phase_stats
        except Exception:
            typer.echo(f"[eval] Unable to parse phase trace stats from {phase_json_path}")
    for plot_path in phase_cfg.get("plots", []):
        plot = _safe_path(plot_path)
        if not plot:
            continue
        try:
            run.log({f"phase_trace/{plot.stem}": wandb.Image(str(plot))}, commit=False)
        except Exception:
            typer.echo(f"[eval] Failed to log phase trace plot {plot}")
        _safe_add_path(artifact, plot, name=plot.name)

    def _handle_summary_block(block_name: str, cfg_key: str) -> None:
        block = wandb_cfg.get(cfg_key)
        if not isinstance(block, dict):
            return
        summary_path = _safe_path(block.get("summary"))
        if summary_path:
            _log_json_table(block_name, summary_path)
            _safe_add_path(artifact, summary_path, name=block.get("summary_name") or summary_path.name)
        plot_path = _safe_path(block.get("plot"))
        if plot_path:
            try:
                run.log({f"{block_name}_plot": wandb.Image(str(plot_path))}, commit=False)
            except Exception:
                typer.echo(f"[eval] Failed to log plot for {cfg_key} at {plot_path}")
            _safe_add_path(artifact, plot_path, name=plot_path.name)

    _handle_summary_block("eval/entropy_buckets", "entropy_buckets")
    _handle_summary_block("eval/fliprate", "fliprate")

    eval_summary_cfg = wandb_cfg.get("eval_summary") or {}
    eval_summary_path = _safe_path(eval_summary_cfg.get("path"))
    if eval_summary_path:
        try:
            with eval_summary_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    summary_table = wandb.Table(columns=reader.fieldnames)
                    for row in reader:
                        summary_table.add_data(*[row.get(col) for col in reader.fieldnames])
                    run.log({"eval/aggregated_summary": summary_table}, commit=False)
        except Exception:
            typer.echo(f"[eval] Failed to build aggregated summary table from {eval_summary_path}")
        _safe_add_path(artifact, eval_summary_path, name=eval_summary_cfg.get("name") or eval_summary_path.name)

    run.log_artifact(artifact)
    run.finish()

