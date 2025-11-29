from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .formatters import FORMATTERS

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

_DATASETS_MISSING_MSG = (
    "The 'datasets' package is required for data loading utilities. "
    "Install the project dependencies (e.g. `pip install -r requirements.txt`) "
    "before calling `adaptive_dpo.data.load_preference_dataset`."
)

__all__ = [
    "load_preference_dataset",
    "load_ultrafeedback_subset_formatted",
]


def _require_datasets() -> Tuple[Any, Callable[..., Any]]:
    """Import datasets lazily so FORMATTERS remain usable without the dependency."""
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise ModuleNotFoundError(_DATASETS_MISSING_MSG) from exc
    return DatasetDict, load_dataset


def _hf_column_names(ds: "Dataset") -> List[str]:
    return list(getattr(ds, "column_names", list(ds.features)))


def load_preference_dataset(
    tokenizer,
    cfg: Dict[str, Any],
) -> "DatasetDict":
    """Load a preference dataset and normalize to {prompt, chosen, rejected} fields."""

    alias = cfg.get("alias", "ultrafeedback")
    if alias not in FORMATTERS:
        raise ValueError(f"Unsupported dataset alias '{alias}'. Supported aliases: {', '.join(FORMATTERS.keys())}.")

    formatter_entry = FORMATTERS[alias]
    # Allow callers to include an explicit `path=None` so we still fall back to the
    # formatter default (the orchestration helpers do this when aliases are passed).
    dataset_path = cfg.get("path")
    if not dataset_path:
        dataset_path = formatter_entry["default_path"]
    splits_cfg = cfg.get("splits") or {"train": "train"}
    if not isinstance(splits_cfg, dict):
        raise TypeError("cfg['splits'] must be a mapping of output split name to dataset split string.")

    sample_frac = cfg.get("sample_frac")
    sample_size = cfg.get("sample_size")
    shuffle = bool(cfg.get("shuffle", False))
    seed = int(cfg.get("seed", 42))
    format_kwargs = cfg.get("format_kwargs") or {}

    DatasetDictCls, load_dataset_fn = _require_datasets()

    formatter = formatter_entry["formatter"]
    output = DatasetDictCls()

    for split_name, split_value in splits_cfg.items():
        try:
            dataset = load_dataset_fn(dataset_path, split=split_value)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_path}' (alias='{alias}', split='{split_value}'): {exc}"
            ) from None
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        if sample_frac and 0.0 < float(sample_frac) < 1.0:
            n_samples = max(1, int(len(dataset) * float(sample_frac)))
            dataset = dataset.select(range(n_samples))
        if sample_size and sample_size > 0:
            n_samples = min(int(sample_size), len(dataset))
            dataset = dataset.select(range(n_samples))

        def _map_fn(batch):
            out = {"prompt": [], "chosen": [], "rejected": []}
            keys = list(batch.keys())
            n_examples = len(batch[keys[0]])

            for i in range(n_examples):
                example = {k: batch[k][i] for k in keys}
                try:
                    formatted = formatter(example, tokenizer, format_kwargs)
                    out["prompt"].append(formatted["prompt"])
                    out["chosen"].append(formatted["chosen"])
                    out["rejected"].append(formatted["rejected"])
                except ValueError:
                    # Skip malformed examples (e.g. empty assistant reply)
                    continue
            return out

        dataset = dataset.map(
            _map_fn,
            batched=True,
            remove_columns=_hf_column_names(dataset),
            desc=f"Formatting {alias}:{split_name}",
        )
        output[split_name] = dataset

    return output


def load_ultrafeedback_subset_formatted(
    tokenizer,
    sample_frac: float = 0.005,
    splits: Optional[List[str]] = None,
) -> "DatasetDict":
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
        "splits": split_mapping,
        "sample_frac": sample_frac,
    }
    return load_preference_dataset(tokenizer, cfg)

