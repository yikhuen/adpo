from typing import List, Optional
import torch

FastLanguageModel = None
_UNSLOTH_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - optional GPU dependency
    from unsloth import FastLanguageModel as _FastLanguageModel  # type: ignore
except Exception as exc:
    _UNSLOTH_IMPORT_ERROR = exc
else:
    FastLanguageModel = _FastLanguageModel  # type: ignore


def _require_unsloth() -> None:
    if FastLanguageModel is None:
        raise RuntimeError(
            "Unsloth generation utilities require a GPU-enabled environment. "
            "Install unsloth with GPU support or mock generate_batch in tests."
        ) from _UNSLOTH_IMPORT_ERROR


def load_base_qwen(max_seq_length: int = 4096, load_in_4bit: bool = True, dtype=None):
    _require_unsloth()
    model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[operator]
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 512):
    _require_unsloth()
    model.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
        )
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts
