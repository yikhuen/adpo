import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_PATH = REPO_ROOT / "src" / "adaptive_dpo" / "pipelines" / "orchestration.py"
SPEC = importlib.util.spec_from_file_location("adaptive_dpo.pipelines.orchestration", ORCH_PATH)
assert SPEC and SPEC.loader
orch = importlib.util.module_from_spec(SPEC)
sys.modules["adaptive_dpo.pipelines.orchestration"] = orch
SPEC.loader.exec_module(orch)


def test_parse_dataset_spec_alias(monkeypatch, tmp_path):
    called = {}

    def fake_prepare(label, alias, options):
        called["label"] = label
        called["alias"] = alias
        called["options"] = options
        output = tmp_path / f"{label}.jsonl"
        output.write_text("[]", encoding="utf-8")
        return output

    monkeypatch.setattr(orch, "_prepare_dataset_prompts_from_alias", fake_prepare)
    opts = orch.DatasetPromptOptions(size=25, split="test", tokenizer="tok")
    label, path = orch._parse_dataset_spec("uf=alias:ultrafeedback", opts)
    assert label == "uf"
    assert Path(path).name == "uf.jsonl"
    assert called["alias"] == "ultrafeedback"
    assert called["options"].size == 25


def test_parse_dataset_spec_config(monkeypatch, tmp_path):
    called = {}

    def fake_prepare(label, config_path, options):
        called["label"] = label
        called["config"] = config_path
        output = tmp_path / f"{label}_config.jsonl"
        output.write_text("[]", encoding="utf-8")
        return output

    monkeypatch.setattr(orch, "_prepare_dataset_prompts_from_config", fake_prepare)
    opts = orch.DatasetPromptOptions(size=50, split="dev", tokenizer="tok")
    label, path = orch._parse_dataset_spec("hh=config:configs/train/sample.yaml", opts)
    assert label == "hh"
    assert Path(path).name == "hh_config.jsonl"
    assert Path(called["config"]).as_posix().endswith("configs/train/sample.yaml")


def test_parse_dataset_spec_alias_requires_options():
    with pytest.raises(orch.typer.BadParameter):
        orch._parse_dataset_spec("uf=alias:ultrafeedback", None)

