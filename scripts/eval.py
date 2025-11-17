import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure local src/ is importable when running as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import unsloth  # noqa: F401  (ensures FastLanguageModel patches are applied)
import openai
import torch
import typer
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_dpo.modeling import load_qwen25_7b_base
from adaptive_dpo.utils.generate import generate_batch

app = typer.Typer(help="Evaluate preference models with multiple judges and export metrics.")

DEFAULT_PROMPT_TEMPLATE = (
    "Prompt:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\nWhich is better? Reply with only A or B."
)


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


def load_prompts(cfg: Dict[str, Any], override_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    limit = override_limit or cfg.get("limit")
    if limit:
        limit = min(limit, len(data))
        if cfg.get("shuffle", False):
            random.seed(cfg.get("seed", 42))
            random.shuffle(data)
        data = data[:limit]
    return data


def strip_prompt(prompt_text: str, full_text: str) -> str:
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :].strip()
    return full_text.strip()


def load_lora_model(ckpt_dir: str, max_seq_length: int = 4096, load_in_4bit: bool = True):
    model, tokenizer = load_qwen25_7b_base(max_seq_length=max_seq_length, load_in_4bit=load_in_4bit)
    try:
        model.load_adapter(ckpt_dir)
    except Exception:
        pass
    return model, tokenizer


def load_model_entry(name: str, entry: Dict[str, Any]):
    kind = entry.get("kind", "lora")
    max_seq_length = int(entry.get("max_seq_length", 4096))
    load_in_4bit = bool(entry.get("load_in_4bit", True))

    if kind == "base":
        model, tokenizer = load_qwen25_7b_base(max_seq_length=max_seq_length, load_in_4bit=load_in_4bit)
    elif kind == "lora":
        ckpt_dir = entry.get("checkpoint")
        if not ckpt_dir:
            raise ValueError(f"Model '{name}' of kind 'lora' requires a 'checkpoint' path.")
        model, tokenizer = load_lora_model(ckpt_dir, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit)
    elif kind == "hf":
        model_id = entry.get("model")
        if not model_id:
            raise ValueError(f"Model '{name}' of kind 'hf' requires a 'model' identifier.")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported model kind '{kind}' for model '{name}'.")

    model.eval()
    return model, tokenizer


def ensure_responses(
    name: str,
    entry: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    generation_cfg: Dict[str, Any],
    output_dir: Path,
    force: bool = False,
) -> List[Dict[str, Any]]:
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    cache_path = responses_dir / f"{name}.jsonl"

    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    batch_size = int(generation_cfg.get("batch_size", 8))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 256))

    model, tokenizer = load_model_entry(name, entry)

    prompt_texts = [p["prompt"] for p in prompts]
    outputs: List[str] = []
    for i in range(0, len(prompt_texts), batch_size):
        chunk = prompt_texts[i : i + batch_size]
        batch_outputs = generate_batch(model, tokenizer, chunk, max_new_tokens=max_new_tokens)
        for prompt_text, full_text in zip(chunk, batch_outputs):
            outputs.append(strip_prompt(prompt_text, full_text))

    records = []
    for idx, (prompt_obj, response_text) in enumerate(zip(prompts, outputs)):
        records.append(
            {
                "id": prompt_obj.get("id", idx),
                "prompt": prompt_obj["prompt"],
                "response": response_text,
            }
        )

    with cache_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records


class PairwiseJudge:
    def __init__(self, cfg: Dict[str, Any]):
        self.name = cfg.get("name") or cfg.get("model")
        self.provider = cfg.get("provider", "openai")
        self.model_name = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 64))
        self.system_prompt = cfg.get("system_prompt")
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

        if self.provider == "openai":
            api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set for OpenAI judge.")
            self.client = openai.OpenAI(api_key=api_key)
            self.kwargs = cfg
        elif self.provider == "hf_causal":
            if not self.model_name:
                raise ValueError("hf_causal judge requires 'model' identifier.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False, trust_remote_code=True
            )
            dtype = torch.float16 if torch.cuda.is_available() else None
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=dtype,
            )
            self.model.eval()
        else:
            raise ValueError(f"Unsupported judge provider '{self.provider}'.")

    def _format_prompt(self, prompt: str, response_a: str, response_b: str) -> str:
        return self.prompt_template.format(prompt=prompt, a=response_a, b=response_b)

    def _parse_choice(self, raw: str) -> str:
        text = raw.strip().upper()
        if "A" in text and "B" not in text:
            return "A"
        if "B" in text:
            return "B"
        return "A"

    def judge(self, prompt: str, response_a: str, response_b: str) -> str:
        formatted = self._format_prompt(prompt, response_a, response_b)
        if self.provider == "openai":
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            else:
                messages.append({"role": "system", "content": DEFAULT_PROMPT_TEMPLATE})
            messages.append({"role": "user", "content": formatted})
            resp = self.client.chat.completions.create(
                model=self.model_name or "gpt-4o-mini",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = resp.choices[0].message.content or ""
            return self._parse_choice(text)

        # hf_causal
        input_text = formatted
        if self.system_prompt:
            input_text = f"{self.system_prompt.strip()}\n\n{input_text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )
            generated = outputs[0][inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_choice(text)


def evaluate_comparison(
    comparison: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    responses: Dict[str, List[Dict[str, Any]]],
    judges: List[PairwiseJudge],
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    decisions_dir = output_dir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    model_a = comparison["a"]
    model_b = comparison["b"]
    responses_a = responses[model_a]
    responses_b = responses[model_b]

    # Ensure lengths match
    if len(responses_a) != len(prompts) or len(responses_b) != len(prompts):
        raise ValueError(f"Response count mismatch for comparison {comparison['name']}.")

    for judge in judges:
        decision_path = decisions_dir / f"{judge.name}__{comparison['name']}.jsonl"
        decisions: List[Dict[str, Any]]

        if decision_path.exists() and not force:
            with decision_path.open("r", encoding="utf-8") as f:
                decisions = [json.loads(line) for line in f]
        else:
            decisions = []
            for prompt_obj, resp_a, resp_b in zip(prompts, responses_a, responses_b):
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

    return metrics


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


@app.command()
def main(
    config: str = typer.Option(..., help="Path to evaluation YAML config."),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts = load_prompts(cfg["prompts"], override_limit=limit)
    output_dir = Path(cfg.get("output", {}).get("dir", "research/results/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_cfg = cfg.get("generation", {})

    # Generate responses for each model
    responses: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, model_entry in cfg["models"].items():
        typer.echo(f"[eval] Generating responses for model '{model_name}'")
        start = time.time()
        records = ensure_responses(
            model_name,
            model_entry,
            prompts,
            generation_cfg,
            output_dir,
            force=force_generate or not generation_cfg.get("cache", True),
        )
        responses[model_name] = records
        typer.echo(f"[eval] Model '{model_name}' responses ready ({time.time()-start:.1f}s)")

    # Prepare judges
    judges = [PairwiseJudge(entry) for entry in cfg.get("judges", [])]
    if not judges:
        raise ValueError("At least one judge must be specified in the config.")

    # Evaluate comparisons
    metrics_summary: Dict[str, Dict[str, Any]] = {}
    per_judge_decisions: Dict[str, Dict[str, Any]] = {judge.name: {} for judge in judges}

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for comparison in cfg.get("comparisons", []):
        typer.echo(f"[eval] Evaluating comparison {comparison['name']}")
        comparison_metrics = evaluate_comparison(
            comparison,
            prompts,
            responses,
            judges,
            output_dir,
            force=force_judge,
        )
        for judge_name, judge_metrics in comparison_metrics.items():
            metric_entry = judge_metrics[comparison["name"]]
            per_judge_decisions[judge_name][comparison["name"]] = metric_entry
            metrics_summary.setdefault(comparison["name"], {})[judge_name] = metric_entry

    # Save summary metrics
    summary_path = metrics_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    for comparison_name, judge_metrics in metrics_summary.items():
        for judge_name, metric_entry in judge_metrics.items():
            typer.echo(
                f"[eval] {comparison_name} | judge={judge_name} | wins={metric_entry['wins']}/{metric_entry['total']} "
                f"({metric_entry['win_rate']:.3f})"
            )
    typer.echo(f"[eval] Wrote metrics summary to {summary_path}")

    csv_path = metrics_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["comparison", "judge", "wins", "total", "win_rate", "ci95_low", "ci95_high", "model_a", "model_b"]
        )
        for comparison_name, judge_metrics in metrics_summary.items():
            for judge_name, metric_entry in judge_metrics.items():
                ci_low, ci_high = metric_entry["ci95"]
                writer.writerow(
                    [
                        comparison_name,
                        judge_name,
                        metric_entry["wins"],
                        metric_entry["total"],
                        metric_entry["win_rate"],
                        ci_low,
                        ci_high,
                        metric_entry["model_a"],
                        metric_entry["model_b"],
                    ]
                )
    typer.echo(f"[eval] Wrote metrics CSV to {csv_path}")

    # Judge agreement
    agreement = compute_judge_agreement(per_judge_decisions)
    if agreement:
        agreement_path = metrics_dir / "judge_agreement.json"
        with agreement_path.open("w", encoding="utf-8") as f:
            json.dump(agreement, f, indent=2)
        typer.echo(f"[eval] Wrote judge agreement report to {agreement_path}")
    else:
        typer.echo("[eval] Agreement metrics skipped (need at least two judges).")


if __name__ == "__main__":
    app()
