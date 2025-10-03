from typing import List, Optional
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


def load_base_qwen(max_seq_length: int = 4096, load_in_4bit: bool = True, dtype=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 512):
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
