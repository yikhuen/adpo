from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_dpo.modeling import load_qwen25_7b_base
from adaptive_dpo.utils.generate import generate_batch
from unsloth import FastLanguageModel

from .prompts import strip_prompt
from .utils import progress


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

