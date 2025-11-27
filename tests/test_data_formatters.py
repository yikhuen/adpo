from adaptive_dpo.data.formatters import FORMATTERS


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}:{msg['content']}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


def test_ultrafeedback_formatter_roundtrip():
    formatter = FORMATTERS["ultrafeedback"]["formatter"]
    tokenizer = DummyTokenizer()
    example = {
        "chosen": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "good"},
        ],
        "rejected": [
            {"role": "assistant", "content": "bad"},
        ],
    }
    formatted = formatter(example, tokenizer, {})
    assert formatted["prompt"].startswith("system:sys")
    assert formatted["chosen"] == "good"
    assert formatted["rejected"] == "bad"


def test_helpsteer2_formatter_requires_columns():
    formatter = FORMATTERS["helpsteer2"]["formatter"]
    tokenizer = DummyTokenizer()
    example = {"prompt": "p", "chosen": "c", "rejected": "r"}
    formatted = formatter(example, tokenizer, {})
    assert "assistant:" in formatted["prompt"]
    assert formatted["chosen"] == "c"

