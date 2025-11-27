from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import progress


def wilson_ci(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    adj = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def evaluate_comparison(
    comparison: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    responses: Dict[str, List[Dict[str, Any]]],
    judges: List[Any],
    output_dir: Path,
    force: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    decisions_dir = output_dir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    qualitative_samples: Dict[str, List[Dict[str, Any]]] = {}

    model_a = comparison["a"]
    model_b = comparison["b"]
    responses_a = responses[model_a]
    responses_b = responses[model_b]

    if len(responses_a) != len(prompts) or len(responses_b) != len(prompts):
        raise ValueError(f"Response count mismatch for comparison {comparison['name']}.")

    for judge in judges:
        decision_path = decisions_dir / f"{judge.name}__{comparison['name']}.jsonl"
        if decision_path.exists() and not force:
            with decision_path.open("r", encoding="utf-8") as f:
                decisions = [json.loads(line) for line in f]
        else:
            decisions = []
            decision_iter = progress(
                zip(prompts, responses_a, responses_b),
                total=len(prompts),
                desc=f"{judge.name}: {comparison['name']}",
            )
            for prompt_obj, resp_a, resp_b in decision_iter:
                choice = judge.judge(prompt_obj["prompt"], resp_a["response"], resp_b["response"])
                decisions.append(
                    {
                        "id": prompt_obj.get("id"),
                        "prompt": prompt_obj["prompt"],
                        "response_a": resp_a["response"],
                        "response_b": resp_b["response"],
                        "choice": choice,
                    }
                )
            with decision_path.open("w", encoding="utf-8") as f:
                for record in decisions:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        qualitative_samples[judge.name] = decisions[:20]

        wins = sum(1 for record in decisions if record["choice"] == "A")
        total = len(decisions)
        wr = wins / total if total else 0.0
        ci = wilson_ci(wins, total)
        metrics.setdefault(judge.name, {})
        metrics[judge.name][comparison["name"]] = {
            "wins": wins,
            "total": total,
            "win_rate": wr,
            "ci95": ci,
            "model_a": model_a,
            "model_b": model_b,
            "decision_path": str(decision_path),
        }

    return metrics, qualitative_samples


def percentage_agreement(a: List[str], b: List[str]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a) if a else 0.0


def compute_cohen_kappa(a: List[str], b: List[str]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    total = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    obs = agree / total
    p_a = sum(1 for x in a if x == "A") / total
    p_b = 1 - p_a
    q_a = sum(1 for y in b if y == "A") / total
    q_b = 1 - q_a
    expected = p_a * q_a + p_b * q_b
    if expected >= 1.0:
        return 0.0
    denom = 1 - expected
    if denom == 0:
        return 0.0
    return (obs - expected) / denom


def compute_judge_agreement(decisions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(decisions) < 2:
        return None
    judge_names = list(decisions.keys())
    primary, secondary = judge_names[0], judge_names[1]

    agreement_report = {
        "judges": [primary, secondary],
        "overall": {},
        "by_comparison": {},
    }

    overall_choices_primary: List[str] = []
    overall_choices_secondary: List[str] = []

    for comparison, primary_metrics in decisions[primary].items():
        decision_path_primary = Path(primary_metrics["decision_path"])
        decision_path_secondary = Path(decisions[secondary][comparison]["decision_path"])
        with decision_path_primary.open("r", encoding="utf-8") as f:
            primary_decisions = [json.loads(line) for line in f]
        with decision_path_secondary.open("r", encoding="utf-8") as f:
            secondary_decisions = [json.loads(line) for line in f]

        sec_map = {d["id"]: d["choice"] for d in secondary_decisions}
        matches = 0
        count = 0
        judge1_choices: List[str] = []
        judge2_choices: List[str] = []
        for record in primary_decisions:
            pid = record["id"]
            if pid not in sec_map:
                continue
            choice1 = record["choice"]
            choice2 = sec_map[pid]
            judge1_choices.append(choice1)
            judge2_choices.append(choice2)
            overall_choices_primary.append(choice1)
            overall_choices_secondary.append(choice2)
            count += 1
            if choice1 == choice2:
                matches += 1
        agreement = matches / count if count else 0.0
        kappa = compute_cohen_kappa(judge1_choices, judge2_choices)
        agreement_report["by_comparison"][comparison] = {
            "n": count,
            "agreement": agreement,
            "cohen_kappa": kappa,
        }

    agreement_report["overall"] = {
        "n": len(overall_choices_primary),
        "agreement": percentage_agreement(overall_choices_primary, overall_choices_secondary),
        "cohen_kappa": compute_cohen_kappa(overall_choices_primary, overall_choices_secondary),
    }
    return agreement_report

