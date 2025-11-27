from pathlib import Path

import pytest

from adaptive_dpo.eval import generation


def test_ensure_responses_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompts = [{"id": 1, "prompt": "Hello"}, {"id": 2, "prompt": "World"}]

    def fake_model_loader(name, entry):
        class Dummy:
            pass

        return Dummy(), Dummy()

    def fake_generate_batch(model, tokenizer, chunk, max_new_tokens):
        return [text + " :: reply" for text in chunk]

    monkeypatch.setattr(generation, "generate_batch", fake_generate_batch)

    records = generation.ensure_responses(
        name="toy",
        entry={"kind": "lora", "checkpoint": "unused"},
        prompts=prompts,
        generation_cfg={"batch_size": 1},
        output_dir=tmp_path,
        model_loader=fake_model_loader,
    )
    assert len(records) == 2
    assert records[0]["response"].endswith("reply")

    # Second call should hit cache and skip recomputation
    records_again = generation.ensure_responses(
        name="toy",
        entry={"kind": "lora", "checkpoint": "unused"},
        prompts=prompts,
        generation_cfg={"batch_size": 1},
        output_dir=tmp_path,
        model_loader=fake_model_loader,
    )
    assert records_again == records

