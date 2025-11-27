from typing import Optional, Tuple

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
            "Unsloth is unavailable (GPU-only dependency). Install a GPU-compatible stack or "
            "mock load_qwen25_7b* during tests."
        ) from _UNSLOTH_IMPORT_ERROR


def load_qwen25_7b_base(max_seq_length: int = 4096, load_in_4bit: bool = True, dtype=None) -> Tuple[object, object]:
    _require_unsloth()
    model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore[operator]
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


_QWEN_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def configure_qwen25_7b_lora(model) -> object:
    _require_unsloth()
    return FastLanguageModel.get_peft_model(  # type: ignore[call-arg]
        model,
        r=64,
        target_modules=_QWEN_LORA_TARGET_MODULES,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


def load_qwen25_7b(max_seq_length: int = 4096, load_in_4bit: bool = True, dtype=None) -> Tuple[object, object]:
    model, tokenizer = load_qwen25_7b_base(max_seq_length=max_seq_length, load_in_4bit=load_in_4bit, dtype=dtype)
    model = configure_qwen25_7b_lora(model)
    return model, tokenizer
