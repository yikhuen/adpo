from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _apply_chat_prompt(tokenizer, system: Optional[str], messages: List[Dict[str, str]]) -> str:
    prompt_messages: List[Dict[str, str]] = []
    if system:
        prompt_messages.append({"role": "system", "content": system})
    prompt_messages.extend(messages)
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


def _hf_column_names(ds: Dataset) -> List[str]:
    return list(getattr(ds, "column_names", list(ds.features)))


# ---------------------------------------------------------------------------
# UltraFeedback formatter
# ---------------------------------------------------------------------------


def _extract_ultrafeedback_fields(example: Dict[str, Any]) -> Dict[str, str]:
    chosen_msgs = example["chosen"]
    rejected_msgs = example["rejected"]

    system_text = ""
    if chosen_msgs and chosen_msgs[0].get("role") == "system":
        system_text = chosen_msgs[0].get("content", "")

    user_text = ""
    for message in chosen_msgs:
        if message.get("role") == "user":
            user_text = message.get("content", "")
            break

    chosen_text = ""
    for message in chosen_msgs:
        if message.get("role") == "assistant":
            chosen_text = message.get("content", "")
            break

    rejected_text = ""
    for message in rejected_msgs:
        if message.get("role") == "assistant":
            rejected_text = message.get("content", "")
            break

    return {
        "system": system_text,
        "user": user_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def _format_ultrafeedback(example: Dict[str, Any], tokenizer, format_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    parts = _extract_ultrafeedback_fields(example)
    messages = [{"role": "user", "content": parts["user"]}]
    prompt_text = _apply_chat_prompt(tokenizer, parts["system"], messages)
    return {
        "prompt": prompt_text,
        "chosen": parts["chosen"],
        "rejected": parts["rejected"],
    }


# ---------------------------------------------------------------------------
# Anthropic HH formatter
# ---------------------------------------------------------------------------


def _parse_hh_conversation(text: str) -> List[Dict[str, str]]:
    """Return list of chat messages derived from Anthropic HH conversation string."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    chunks = [chunk for chunk in normalized.split("\n\n") if chunk.strip()]
    messages: List[Dict[str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk.lower().startswith("human:"):
            messages.append({"role": "user", "content": chunk.split(":", 1)[1].strip()})
        elif chunk.lower().startswith("assistant:"):
            messages.append({"role": "assistant", "content": chunk.split(":", 1)[1].strip()})
    return messages


def _format_anthropic_hh(example: Dict[str, Any], tokenizer, format_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    format_kwargs = format_kwargs or {}
    system_prompt = format_kwargs.get("system_prompt")

    chosen_msgs = _parse_hh_conversation(example["chosen"])
    rejected_msgs = _parse_hh_conversation(example["rejected"])
    if not chosen_msgs or chosen_msgs[-1]["role"] != "assistant":
        raise ValueError("Anthropic HH example does not end with an assistant reply.")
    if not rejected_msgs or rejected_msgs[-1]["role"] != "assistant":
        raise ValueError("Anthropic HH example (rejected) does not end with an assistant reply.")

    context_messages = chosen_msgs[:-1]
    if not context_messages:
        raise ValueError("Anthropic HH example missing user context.")

    prompt_text = _apply_chat_prompt(tokenizer, system_prompt, context_messages)
    return {
        "prompt": prompt_text,
        "chosen": chosen_msgs[-1]["content"],
        "rejected": rejected_msgs[-1]["content"],
    }


# ---------------------------------------------------------------------------
# Sycophancy formatter
# ---------------------------------------------------------------------------


def _format_sycophancy(example: Dict[str, Any], tokenizer, format_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    format_kwargs = format_kwargs or {}
    question_key = format_kwargs.get("question_key", "question")
    chosen_key = format_kwargs.get("preferred_key", "answer_not_matching_behavior")
    rejected_key = format_kwargs.get("rejected_key", "answer_matching_behavior")
    system_prompt = format_kwargs.get("system_prompt")

    prompt_value = example.get(question_key)
    if prompt_value is None:
        raise KeyError(f"Sycophancy example missing '{question_key}' column.")

    chosen_value = example.get(chosen_key)
    rejected_value = example.get(rejected_key)
    if chosen_value is None or rejected_value is None:
        raise KeyError(
            f"Sycophancy example missing preferred ('{chosen_key}') or rejected ('{rejected_key}') response columns."
        )

    prompt_text = _apply_chat_prompt(
        tokenizer,
        system_prompt,
        [{"role": "user", "content": prompt_value}],
    )
    return {
        "prompt": prompt_text,
        "chosen": chosen_value,
        "rejected": rejected_value,
    }


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------


FORMATTERS = {
    "ultrafeedback": {
        "default_path": "HuggingFaceH4/ultrafeedback_binarized",
        "formatter": _format_ultrafeedback,
    },
    "anthropic_hh": {
        "default_path": "Anthropic/hh-rlhf",
        "formatter": _format_anthropic_hh,
    },
    "sycophancy": {
        "default_path": "nala/sycophancy",
        "formatter": _format_sycophancy,
    },
}


def load_preference_dataset(
    tokenizer,
    cfg: Dict[str, Any],
) -> DatasetDict:
    """Load a preference dataset and normalize to {prompt, chosen, rejected} fields.

    Args:
        tokenizer: Tokenizer with `apply_chat_template`.
        cfg: Dictionary configuration with keys:
            alias: dataset alias, one of FORMATTERS keys.
            path: (optional) Hugging Face dataset path override.
            splits: mapping of output split name -> dataset split string.
            sample_frac: optional fraction of examples to keep.
            sample_size: optional absolute number of examples per split.
            seed: seed used for optional shuffling.
            shuffle: whether to shuffle before sampling.
            format_kwargs: optional dict passed to formatter.

    Returns:
        DatasetDict with keys equal to cfg["splits"] keys. Each split contains
        columns {prompt, chosen, rejected}. Additional metadata columns are removed.
    """

    alias = cfg.get("alias", "ultrafeedback")
    if alias not in FORMATTERS:
        raise ValueError(f"Unsupported dataset alias '{alias}'. Supported aliases: {', '.join(FORMATTERS.keys())}.")

    formatter_entry = FORMATTERS[alias]
    dataset_path = cfg.get("path", formatter_entry["default_path"])
    splits_cfg = cfg.get("splits") or {"train": "train"}
    if not isinstance(splits_cfg, dict):
        raise TypeError("cfg['splits'] must be a mapping of output split name to dataset split string.")

    sample_frac = cfg.get("sample_frac")
    sample_size = cfg.get("sample_size")
    shuffle = bool(cfg.get("shuffle", False))
    seed = int(cfg.get("seed", 42))
    format_kwargs = cfg.get("format_kwargs") or {}

    formatter = formatter_entry["formatter"]
    output = DatasetDict()

    for split_name, split_value in splits_cfg.items():
        dataset = load_dataset(dataset_path, split=split_value)
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if sample_frac and 0.0 < float(sample_frac) < 1.0:
            n_samples = max(1, int(len(dataset) * float(sample_frac)))
            dataset = dataset.select(range(n_samples))
        if sample_size and sample_size > 0:
            n_samples = min(int(sample_size), len(dataset))
            dataset = dataset.select(range(n_samples))

        def _map_fn(example):
            return formatter(example, tokenizer, format_kwargs)

        dataset = dataset.map(
            _map_fn,
            remove_columns=_hf_column_names(dataset),
        )
        output[split_name] = dataset

    return output


def load_ultrafeedback_subset_formatted(
    tokenizer,
    sample_frac: float = 0.005,
    splits: Optional[List[str]] = None,
) -> DatasetDict:
    """Backward-compatible helper for legacy training config."""
    if splits is None:
        splits = ["train_prefs", "test_prefs"]
    split_mapping: Dict[str, str] = {}
    for split in splits:
        if "train" in split:
            split_mapping["train"] = split
        else:
            split_mapping["test"] = split
    cfg = {
        "alias": "ultrafeedback",
        "path": FORMATTERS["ultrafeedback"]["default_path"],
        "splits": split_mapping,
        "sample_frac": sample_frac,
    }
    return load_preference_dataset(tokenizer, cfg)
