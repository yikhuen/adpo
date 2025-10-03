from typing import Dict, List
from datasets import load_dataset, DatasetDict


def _extract_prompt_chosen_rejected(example) -> Dict[str, str]:
    chosen_msgs = example["chosen"]
    rejected_msgs = example["rejected"]
    # System (optional)
    system_text = ""
    if chosen_msgs and chosen_msgs[0].get("role") == "system":
        system_text = chosen_msgs[0].get("content", "")
    # First user
    user_text = ""
    for m in chosen_msgs:
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break
    # First assistant responses
    chosen_text = ""
    for m in chosen_msgs:
        if m.get("role") == "assistant":
            chosen_text = m.get("content", "")
            break
    rejected_text = ""
    for m in rejected_msgs:
        if m.get("role") == "assistant":
            rejected_text = m.get("content", "")
            break
    return {
        "system": system_text,
        "user": user_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def load_ultrafeedback_subset_formatted(tokenizer, sample_frac: float = 0.005, splits: List[str] = None) -> DatasetDict:
    if splits is None:
        splits = ["train_prefs", "test_prefs"]
    raw = DatasetDict()
    for split in splits:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
        if sample_frac and 0.0 < sample_frac < 1.0:
            n = max(1, int(len(ds) * sample_frac))
            ds = ds.select(range(n))
        def _format(ex):
            parts = _extract_prompt_chosen_rejected(ex)
            # Build prompt text with chat template (system + user) and add generation prompt
            messages = []
            if parts["system"]:
                messages.append({"role": "system", "content": parts["system"]})
            messages.append({"role": "user", "content": parts["user"]})
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {
                "prompt": prompt_text,
                "chosen": parts["chosen"],
                "rejected": parts["rejected"],
            }
        ds = ds.map(_format, remove_columns=list(ds.features))
        raw["train" if "train" in split else "test"] = ds
    return raw
