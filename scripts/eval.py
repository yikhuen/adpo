import csv
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
import hashlib
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
from unsloth import FastLanguageModel

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None
def _progress(iterable, *, total: Optional[int] = None, desc: Optional[str] = None):
    """Wrap iterable with tqdm when available."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)

try:
    from plot_ablation_bar import build_ablation_bar_figure
except ImportError:  # pragma: no cover - helper script may be unavailable in some contexts
    build_ablation_bar_figure = None

app = typer.Typer(help="Evaluate preference models with multiple judges and export metrics.")

DEFAULT_PROMPT_TEMPLATE = (
    "Prompt:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\nWhich is better? Reply with only A or B."
)

DEFAULT_ALL_JUDGES_CONFIG = str(Path("configs/eval/judge_gpt4o_mini.yaml"))
DEFAULT_OPENAI_ONLY_CONFIG = str(Path("configs/eval/judge_openai_only.yaml"))
DEFAULT_GEMINI_ONLY_CONFIG = str(Path("configs/eval/judge_gemini_only.yaml"))

DEFAULT_REWARD_HACKING_THRESHOLDS: Dict[str, Any] = {
    "avg_length_chars": {"max": 1500},
    "refusal_rate": {"max": 0.3},
    "safety_rate": {"max": 0.1},
    "length_ratio_max": 2.5,
}


def _merge_thresholds(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(defaults)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def assess_reward_hacking(
    model_stats: Dict[str, Dict[str, float]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not model_stats:
        return {}

    cfg = cfg or {}
    if cfg.get("enabled") is False:
        return {}

    thresholds_cfg = cfg.get("thresholds")
    thresholds = _merge_thresholds(DEFAULT_REWARD_HACKING_THRESHOLDS, thresholds_cfg)

    lengths = {name: stats.get("avg_length_chars", 0.0) or 0.0 for name, stats in model_stats.items()}
    shortest_length = min((length for length in lengths.values() if length > 0), default=0.0)

    per_model: Dict[str, Any] = {}
    alerts: List[str] = []
    any_warnings = False

    for model_name, stats in model_stats.items():
        issues: List[str] = []
        avg_length = stats.get("avg_length_chars", float("nan"))
        refusal_rate = stats.get("refusal_rate", float("nan"))
        safety_rate = stats.get("safety_rate", float("nan"))

        length_cfg = thresholds.get("avg_length_chars", {})
        max_len = length_cfg.get("max")
        min_len = length_cfg.get("min")
        if max_len is not None and avg_length and avg_length > max_len:
            issues.append(f"avg_length_chars={avg_length:.1f} exceeds max {max_len}")
        if min_len is not None and avg_length and avg_length < min_len:
            issues.append(f"avg_length_chars={avg_length:.1f} below min {min_len}")

        if not isinstance(refusal_rate, float):
            try:
                refusal_rate = float(refusal_rate)
            except (TypeError, ValueError):
                refusal_rate = float("nan")
        refusal_cfg = thresholds.get("refusal_rate", {})
        max_refusal = refusal_cfg.get("max")
        if max_refusal is not None and refusal_rate == refusal_rate and refusal_rate > max_refusal:
            issues.append(f"refusal_rate={refusal_rate:.3f} exceeds max {max_refusal}")

        if not isinstance(safety_rate, float):
            try:
                safety_rate = float(safety_rate)
            except (TypeError, ValueError):
                safety_rate = float("nan")
        safety_cfg = thresholds.get("safety_rate", {})
        max_safety = safety_cfg.get("max")
        if max_safety is not None and safety_rate == safety_rate and safety_rate > max_safety:
            issues.append(f"safety_rate={safety_rate:.3f} exceeds max {max_safety}")

        length_ratio_max = thresholds.get("length_ratio_max")
        if length_ratio_max and shortest_length > 0 and avg_length and avg_length / shortest_length > length_ratio_max:
            issues.append(
                f"avg_length_chars ratio {avg_length/shortest_length:.2f} exceeds max {length_ratio_max}"
            )

        status = "warning" if issues else "ok"
        if status == "warning":
            any_warnings = True
            alerts.append(f"{model_name}: {', '.join(issues)}")

        per_model[model_name] = {
            "status": status,
            "issues": issues,
            "metrics": {
                "avg_length_chars": avg_length,
                "refusal_rate": refusal_rate,
                "safety_rate": safety_rate,
                "responses": stats.get("count", 0),
            },
        }

    report = {
        "thresholds": thresholds,
        "per_model": per_model,
        "alerts": alerts,
        "overall_status": "warning" if any_warnings else "ok",
    }
    return report


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
    ckpt_path = Path(ckpt_dir)
    adapter_path = ckpt_path / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Expected LoRA adapter weights at '{adapter_path}'. "
            "Verify the checkpoint directory is correct and training completed."
        )

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(ckpt_path),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    except Exception as exc:
        raise RuntimeError(f"Failed to load LoRA adapter from '{ckpt_dir}'.") from exc

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

    def _prompt_key(prompt_obj: Dict[str, Any], fallback_idx: int) -> str:
        prompt_id = prompt_obj.get("id", fallback_idx)
        return str(prompt_id)

    records_by_id: Dict[str, Dict[str, Any]] = {}
    cache_loaded = cache_path.exists() and not force
    if cache_loaded:
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = record.get("id")
                key = str(record_id)
                records_by_id[key] = record

    missing_prompts: List[Tuple[int, Dict[str, Any]]] = []
    for idx, prompt_obj in enumerate(prompts):
        key = _prompt_key(prompt_obj, idx)
        if key not in records_by_id:
            missing_prompts.append((idx, prompt_obj))

    if cache_loaded and not missing_prompts:
        typer.echo(
            f"[eval] Using cached responses for model '{name}'. "
            "Run with --force-generate or delete the cache to regenerate."
        )
        return [records_by_id[_prompt_key(prompt_obj, idx)] for idx, prompt_obj in enumerate(prompts)]

    if cache_loaded and missing_prompts:
        typer.echo(
            f"[eval] Resuming cached responses for model '{name}' "
            f"({len(missing_prompts)} missing of {len(prompts)} prompts)."
        )

    if missing_prompts:
        batch_size = int(generation_cfg.get("batch_size", 8))
        max_new_tokens = int(generation_cfg.get("max_new_tokens", 256))

        model, tokenizer = load_model_entry(name, entry)

        prompt_texts = [prompt_obj["prompt"] for _, prompt_obj in missing_prompts]
        generated_texts: List[str] = []
        total_batches = max(1, (len(prompt_texts) + batch_size - 1) // batch_size)
        batch_iter = _progress(
            range(0, len(prompt_texts), batch_size),
            total=total_batches,
            desc=f"{name} generation",
        )
        for i in batch_iter:
            chunk = prompt_texts[i : i + batch_size]
            batch_outputs = generate_batch(model, tokenizer, chunk, max_new_tokens=max_new_tokens)
            for prompt_text, full_text in zip(chunk, batch_outputs):
                generated_texts.append(strip_prompt(prompt_text, full_text))

        for (idx, prompt_obj), response_text in zip(missing_prompts, generated_texts):
            key = _prompt_key(prompt_obj, idx)
            records_by_id[key] = {
                "id": prompt_obj.get("id", idx),
                "prompt": prompt_obj["prompt"],
                "response": response_text,
            }

    ordered_records: List[Dict[str, Any]] = []
    for idx, prompt_obj in enumerate(prompts):
        key = _prompt_key(prompt_obj, idx)
        record = records_by_id.get(key)
        if record is None:
            raise ValueError(
                f"Missing response for prompt id '{key}' after generation. "
                f"Delete cache at {cache_path} and retry."
            )
        ordered_records.append(record)

    with cache_path.open("w", encoding="utf-8") as f:
        for record in ordered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return ordered_records


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
            # Add timeout and max_retries to handle connection issues
            timeout = cfg.get("timeout", 60.0)
            max_retries = cfg.get("max_retries", 3)
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
            self.kwargs = cfg
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("google-generativeai must be installed for Gemini judges.") from exc

            api_key = cfg.get("api_key") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set for Gemini judge.")

            genai.configure(api_key=api_key)
            model_name = self.model_name or cfg.get("model")
            if not model_name:
                model_name = "gemini-2.0-flash-001"
            # Strip 'models/' prefix if accidentally included
            if model_name.startswith("models/"):
                model_name = model_name[7:]
            generation_config = cfg.get("generation_config")
            safety_settings = cfg.get("safety_settings")
            init_kwargs: Dict[str, Any] = {}
            if generation_config:
                init_kwargs["generation_config"] = generation_config
            if safety_settings:
                init_kwargs["safety_settings"] = safety_settings
            self.gemini_client = genai.GenerativeModel(model_name, **init_kwargs)
            self.gemini_kwargs = cfg
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
            
            # Retry logic for connection errors
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name or "gpt-4o-mini",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                    return self._parse_choice(text)
                except (openai.APIConnectionError, openai.APIError) as e:
                    if attempt < max_attempts - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
        elif self.provider == "gemini":
            response = self.gemini_client.generate_content(formatted)
            text = getattr(response, "text", "") or ""
            if not text and getattr(response, "candidates", None):
                try:
                    text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    text = ""
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


QUAL_SAMPLE_LIMIT = 20


def _select_sample_records(decisions: List[Dict[str, Any]], limit: int = QUAL_SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    if len(decisions) <= limit:
        return decisions.copy()
    step = max(1, len(decisions) // limit)
    sampled: List[Dict[str, Any]] = []
    for idx in range(0, len(decisions), step):
        sampled.append(decisions[idx])
        if len(sampled) >= limit:
            break
    return sampled


def evaluate_comparison(
    comparison: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    responses: Dict[str, List[Dict[str, Any]]],
    judges: List[PairwiseJudge],
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
            decision_iter = _progress(
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

        qualitative_samples[judge.name] = _select_sample_records(decisions)

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

    def _build_decision_table(decision_paths: List[Path], max_rows: int, prompt_chars: int):
        if not decision_paths or max_rows <= 0:
            return None
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
        return table if rows_added else None

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
        if build_ablation_bar_figure is None:
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
                    chart_name = bar_cfg.get("name", "eval/ablation_bar")
                    import matplotlib.pyplot as plt

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

    # Decision table logging and attachments
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
    # Deduplicate while preserving order
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
        decision_table = _build_decision_table(decision_paths, max_rows, prompt_chars)
        if decision_table is not None:
            run.log({"eval/decision_table": decision_table}, commit=False)

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


def run_evaluation(
    config: str,
    limit: Optional[int] = None,
    force_generate: bool = False,
    force_judge: bool = False,
) -> None:
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

    # Compute per-model stats for reward hacking sanity checks
    model_level_stats: Dict[str, Dict[str, float]] = {}
    for model_name, records in responses.items():
        total_length = 0
        safety_flags = 0
        refusals = 0
        for record in records:
            text = record.get("response", "")
            total_length += len(text)
            metadata = record.get("metadata") or {}
            if metadata.get("safety_flag"):
                safety_flags += 1
            if metadata.get("refused"):
                refusals += 1
        count = max(1, len(records))
        model_level_stats[model_name] = {
            "count": count,
            "avg_length_chars": total_length / count,
            "safety_rate": safety_flags / count,
            "refusal_rate": refusals / count,
        }

    # Prepare judges
    judges = [PairwiseJudge(entry) for entry in cfg.get("judges", [])]
    if not judges:
        raise ValueError("At least one judge must be specified in the config.")

    # Evaluate comparisons
    metrics_summary: Dict[str, Dict[str, Any]] = {}
    qualitative_samples: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    per_judge_decisions: Dict[str, Dict[str, Any]] = {judge.name: {} for judge in judges}

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for comparison in cfg.get("comparisons", []):
        typer.echo(f"[eval] Evaluating comparison {comparison['name']}")
        comparison_metrics, comparison_sample_records = evaluate_comparison(
            comparison,
            prompts,
            responses,
            judges,
            output_dir,
            force=force_judge,
        )
        qualitative_samples[comparison["name"]] = comparison_sample_records
        for judge_name, judge_metrics in comparison_metrics.items():
            metric_entry = judge_metrics[comparison["name"]]
            per_judge_decisions[judge_name][comparison["name"]] = metric_entry
            metrics_summary.setdefault(comparison["name"], {})[judge_name] = metric_entry

    def _clip_text(value: str, limit: int = 1000) -> str:
        if not isinstance(value, str):
            value = str(value)
        return value if len(value) <= limit else value[:limit]

    qualitative_by_model: Dict[str, List[Dict[str, Any]]] = {model: [] for model in responses.keys()}

    def _add_model_sample(
        model_name: str,
        position: str,
        record: Dict[str, Any],
        opponent_model: str,
        comparison_name: str,
        judge_name: str,
    ) -> None:
        rows = qualitative_by_model.setdefault(model_name, [])
        if len(rows) >= QUAL_SAMPLE_LIMIT:
            return
        response_key = "response_a" if position == "A" else "response_b"
        opponent_key = "response_b" if position == "A" else "response_a"
        prompt_text = _clip_text(record.get("prompt", ""))
        response_text = _clip_text(record.get(response_key, ""))
        opponent_response_text = _clip_text(record.get(opponent_key, ""))
        choice = record.get("choice")
        result = "win" if choice == position else "loss"
        rows.append(
            {
                "comparison": comparison_name,
                "judge": judge_name,
                "prompt_id": record.get("id"),
                "prompt": prompt_text,
                "response": response_text,
                "opponent_model": opponent_model,
                "opponent_response": opponent_response_text,
                "position": position,
                "choice": choice,
                "result": result,
            }
        )

    for comparison_name, judge_records in qualitative_samples.items():
        for judge_name, sample_records in judge_records.items():
            metric_entry = metrics_summary.get(comparison_name, {}).get(judge_name)
            if not metric_entry:
                continue
            model_a = metric_entry["model_a"]
            model_b = metric_entry["model_b"]
            for record in sample_records:
                _add_model_sample(model_a, "A", record, model_b, comparison_name, judge_name)
                _add_model_sample(model_b, "B", record, model_a, comparison_name, judge_name)

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

    model_stats_path = None
    if model_level_stats:
        model_stats_path = metrics_dir / "model_stats.json"
        with model_stats_path.open("w", encoding="utf-8") as f:
            json.dump(model_level_stats, f, indent=2)
        typer.echo(f"[eval] Wrote model-level stats to {model_stats_path}")

    reward_hacking_report = assess_reward_hacking(model_level_stats, cfg.get("reward_hacking"))
    reward_hacking_path = None
    if reward_hacking_report:
        if reward_hacking_report.get("alerts"):
            for alert in reward_hacking_report["alerts"]:
                typer.echo(f"[eval] Reward hacking warning: {alert}")
        else:
            typer.echo("[eval] Reward hacking checks passed for all models.")
        reward_hacking_path = metrics_dir / "reward_hacking.json"
        with reward_hacking_path.open("w", encoding="utf-8") as f:
            json.dump(reward_hacking_report, f, indent=2)
        typer.echo(f"[eval] Wrote reward hacking report to {reward_hacking_path}")

    log_results_to_wandb(
        cfg,
        metrics_summary,
        summary_path,
        csv_path,
        prompt_count=len(prompts),
        config_path=config,
        model_stats=model_level_stats,
        model_stats_path=model_stats_path,
        reward_hacking_report=reward_hacking_report,
        reward_hacking_path=reward_hacking_path,
        qualitative_samples_by_model=qualitative_by_model,
        output_dir=output_dir,
    )

    # Judge agreement
    agreement = compute_judge_agreement(per_judge_decisions)
    if agreement:
        agreement_path = metrics_dir / "judge_agreement.json"
        with agreement_path.open("w", encoding="utf-8") as f:
            json.dump(agreement, f, indent=2)
        typer.echo(f"[eval] Wrote judge agreement report to {agreement_path}")
    else:
        typer.echo("[eval] Agreement metrics skipped (need at least two judges).")


@app.command()
def main(
    config: str = typer.Option(..., help="Path to evaluation YAML config."),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    """Run evaluation with an explicit config."""
    run_evaluation(config, limit, force_generate, force_judge)


@app.command("openai-judge")
def run_openai_judge(
    config: str = typer.Option(
        DEFAULT_OPENAI_ONLY_CONFIG,
        "--config",
        "-c",
        help="Path to an OpenAI-only eval config.",
        show_default=False,
    ),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    """Run evaluation using only the OpenAI judge."""
    typer.echo(f"[eval] Using OpenAI-only judge config: {config}")
    run_evaluation(config, limit, force_generate, force_judge)


@app.command("gemini-judge")
def run_gemini_judge(
    config: str = typer.Option(
        DEFAULT_GEMINI_ONLY_CONFIG,
        "--config",
        "-c",
        help="Path to a Gemini-only eval config.",
        show_default=False,
    ),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    """Run evaluation using only the Gemini judge."""
    typer.echo(f"[eval] Using Gemini-only judge config: {config}")
    run_evaluation(config, limit, force_generate, force_judge)


@app.command("all-judges")
def run_all_judges(
    config: str = typer.Option(
        DEFAULT_ALL_JUDGES_CONFIG,
        "--config",
        "-c",
        help="Path to the combined judge eval config.",
        show_default=False,
    ),
    limit: Optional[int] = typer.Option(None, help="Override prompt limit from config."),
    force_generate: bool = typer.Option(False, help="Force regeneration of model responses."),
    force_judge: bool = typer.Option(False, help="Force re-running judges even if cached decisions exist."),
):
    """Run evaluation with both OpenAI and Gemini judges."""
    typer.echo(f"[eval] Using combined judge config: {config}")
    run_evaluation(config, limit, force_generate, force_judge)


if __name__ == "__main__":
    app()
