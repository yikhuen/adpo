import os
import json
import time
import yaml
import typer
import random
from typing import List, Dict, Tuple

import torch
import openai
from unsloth import FastLanguageModel

app = typer.Typer()

RUBRIC_TEMPLATE = (
    "You are a strict evaluator. Given a prompt and two responses (A and B), "
    "choose which response better follows the instructions, is helpful, correct, concise, and safe. "
    "Reply with only 'A' or 'B'."
)


def wilson_ci(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    adj = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return max(0.0, lower), min(1.0, upper)


def load_base_model():
    model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", max_seq_length=4096, load_in_4bit=True)
    return model, tok


def load_lora_model(ckpt_dir: str):
    base_model, tok = FastLanguageModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct", max_seq_length=4096, load_in_4bit=True)
    try:
        base_model.load_adapter(ckpt_dir)
    except Exception:
        # Fallback: PEFT merges are saved under same dir; if not present, rely on base
        pass
    return base_model, tok


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    model.eval()
    inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def judge_choice(client, model: str, system_prompt: str, prompt: str, a: str, b: str) -> str:
    content = [
        {"role": "system", "content": system_prompt or RUBRIC_TEMPLATE},
        {
            "role": "user",
            "content": f"Prompt:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\nWhich is better? Reply with only A or B.",
        },
    ]
    resp = client.chat.completions.create(model=model, messages=content, temperature=0.0, max_tokens=1)
    text = resp.choices[0].message.content.strip().upper()
    return "A" if ("A" in text and "B" not in text) else ("B" if "B" in text else "A")


@app.command()
def main(config: str = typer.Option(...), ckpt_adaptive: str = typer.Option("outputs"), ckpt_fixed: str = typer.Option("outputs_fixed"), dev: str = typer.Option(...)):
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = openai.OpenAI(api_key=api_key)

    # Load models
    base_model, base_tok = load_base_model()
    ada_model, ada_tok = load_lora_model(ckpt_adaptive)
    fix_model, fix_tok = load_lora_model(ckpt_fixed)

    # Load prompts
    prompts = []
    with open(dev, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])

    def eval_pair(model_a, tok_a, model_b, tok_b, title: str):
        wins = 0
        total = 0
        for p in prompts:
            a = generate(model_a, tok_a, p)
            b = generate(model_b, tok_b, p)
            choice = judge_choice(
                client,
                model=cfg["judge"].get("model", "gpt-4o-mini"),
                system_prompt=cfg["judge"].get("system_prompt", RUBRIC_TEMPLATE),
                prompt=p,
                a=a,
                b=b,
            )
            wins += 1 if choice == "A" else 0
            total += 1
        wr = wins / max(1, total)
        lo, hi = wilson_ci(wins, total)
        print(json.dumps({"pair": title, "wins": wins, "total": total, "win_rate": wr, "ci95": [lo, hi]}, indent=2))

    eval_pair(ada_model, ada_tok, base_model, base_tok, "adaptive_vs_base")
    eval_pair(ada_model, ada_tok, fix_model, fix_tok, "adaptive_vs_fixed")


if __name__ == "__main__":
    app()
