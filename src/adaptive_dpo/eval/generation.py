from __future__ import annotations

# Ensure Unsloth patches transformers before we import heavy deps like torch/AutoModel.
try:  # pragma: no cover - optional GPU dependency
    from unsloth import FastLanguageModel as _FastLanguageModel  # type: ignore
except Exception as exc:  # pragma: no cover - best effort on CPU CI
    _UNSLOTH_IMPORT_ERROR: Optional[Exception] = exc  # type: ignore[assignment]
    FastLanguageModel = None
else:
    FastLanguageModel = _FastLanguageModel  # type: ignore
    _UNSLOTH_IMPORT_ERROR = None

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency for tests
    import torch
except Exception:  # pragma: no cover - fall back to mocks
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for tests
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - fall back to mocks
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from adaptive_dpo.modeling import load_qwen25_7b_base
from adaptive_dpo.utils.generate import generate_batch

from .prompts import strip_prompt
from .utils import progress


def _require_unsloth() -> None:
    if FastLanguageModel is None:
        raise RuntimeError(
            "Unsloth is unavailable in this environment. "
            "GPU-backed installations are required when loading LoRA models."
        ) from _UNSLOTH_IMPORT_ERROR


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for HF model loading. Install the CPU build for tests or enable GPU support."
        )
    return torch


def _require_transformers():
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ModuleNotFoundError(
            "transformers is required for HF model loading. Install the library to use 'hf' generation entries."
        )
    return AutoModelForCausalLM, AutoTokenizer


def _resolve_checkpoint_dir(ckpt_path: Path) -> Path:
    """Return a checkpoint directory, falling back to nested beta_* folders if needed."""
    if ckpt_path.exists():
        return ckpt_path

    parent = ckpt_path.parent
    if not parent.exists():
        return ckpt_path

    candidates = [nested for nested in parent.glob(f"*/{ckpt_path.name}") if nested.exists()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous LoRA checkpoint for '{ckpt_path}'. "
            f"Multiple candidates found: {', '.join(str(c) for c in candidates)}. "
            "Please provide the full nested path."
        )
    return ckpt_path


def load_lora_model(ckpt_dir: str, max_seq_length: int = 4096, load_in_4bit: bool = True):
    _require_unsloth()
    validated_path = _resolve_checkpoint_dir(Path(ckpt_dir))
    adapter_path = validated_path / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Expected LoRA adapter weights at '{adapter_path}'. "
            "Verify the checkpoint directory is correct and training completed."
        )

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[operator]
            model_name=str(validated_path),
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
        torch_mod = _require_torch()
        AutoModelCls, AutoTokenizerCls = _require_transformers()
        model_id = entry.get("model")
        if not model_id:
            raise ValueError(f"Model '{name}' of kind 'hf' requires a 'model' identifier.")
        tokenizer = AutoTokenizerCls.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
        dtype = torch_mod.float16 if torch_mod.cuda.is_available() else None
        model = AutoModelCls.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported model kind '{kind}' for model '{name}'.")

    model.eval()
    return model, tokenizer


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(record.get("id"))
            records[key] = record
    return records


def ensure_responses(
    name: str,
    entry: Dict[str, Any],
    prompts: List[Dict[str, Any]],
    generation_cfg: Dict[str, Any],
    output_dir: Path,
    force: bool = False,
    model_loader=load_model_entry,
) -> List[Dict[str, Any]]:
    responses_dir = output_dir / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)
    cache_path = responses_dir / f"{name}.jsonl"

    def _prompt_key(prompt_obj: Dict[str, Any], fallback_idx: int) -> str:
        prompt_id = prompt_obj.get("id", fallback_idx)
        return str(prompt_id)

    records_by_id: Dict[str, Dict[str, Any]] = _load_cache(cache_path) if not force else {}

    missing_prompts: List[Tuple[int, Dict[str, Any]]] = []
    for idx, prompt_obj in enumerate(prompts):
        key = _prompt_key(prompt_obj, idx)
        if key not in records_by_id:
            missing_prompts.append((idx, prompt_obj))

    if records_by_id and not missing_prompts:
        return [records_by_id[_prompt_key(prompt_obj, idx)] for idx, prompt_obj in enumerate(prompts)]

    if not missing_prompts:
        missing_prompts = [(idx, prompt_obj) for idx, prompt_obj in enumerate(prompts)]

    batch_size = int(generation_cfg.get("batch_size", 8))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 256))
    model, tokenizer = model_loader(name, entry)

    prompt_texts = [prompt_obj["prompt"] for _, prompt_obj in missing_prompts]
    generated_texts: List[str] = []
    total_batches = max(1, (len(prompt_texts) + batch_size - 1) // batch_size)
    batch_iter = progress(
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

