from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import progress

DEFAULT_PROMPT_TEMPLATE = "Prompt:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\nWhich is better? Reply with only A or B."


class PairwiseJudge:
    def __init__(self, cfg: Dict[str, Any]):
        self.name = cfg.get("name") or cfg.get("model")
        self.provider = cfg.get("provider", "openai")
        self.model_name = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 64))
        self.system_prompt = cfg.get("system_prompt")
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

        if self.provider == "openai":
            api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set for OpenAI judge.")
            timeout = cfg.get("timeout", 60.0)
            max_retries = cfg.get("max_retries", 3)
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
            self.kwargs = cfg
        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("google-generativeai must be installed for Gemini judges.") from exc

            api_key = cfg.get("api_key") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set for Gemini judge.")

            genai.configure(api_key=api_key)
            model_name = self.model_name or cfg.get("model") or "gemini-2.0-flash-001"
            if model_name.startswith("models/"):
                model_name = model_name[7:]
            generation_config = cfg.get("generation_config")
            safety_settings = cfg.get("safety_settings")
            init_kwargs: Dict[str, Any] = {}
            if generation_config:
                init_kwargs["generation_config"] = generation_config
            if safety_settings:
                init_kwargs["safety_settings"] = safety_settings
            self.gemini_client = genai.GenerativeModel(model_name, **init_kwargs)
            self.gemini_kwargs = cfg
        elif self.provider == "hf_causal":
            if not self.model_name:
                raise ValueError("hf_causal judge requires 'model' identifier.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False, trust_remote_code=True
            )
            dtype = torch.float16 if torch.cuda.is_available() else None
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=dtype,
            )
            self.model.eval()
        else:
            raise ValueError(f"Unsupported judge provider '{self.provider}'.")

    def _format_prompt(self, prompt: str, response_a: str, response_b: str) -> str:
        return self.prompt_template.format(prompt=prompt, a=response_a, b=response_b)

    def _parse_choice(self, raw: str) -> str:
        text = raw.strip().upper()
        if "A" in text and "B" not in text:
            return "A"
        if "B" in text:
            return "B"
        return "A"

    def judge(self, prompt: str, response_a: str, response_b: str) -> str:
        formatted = self._format_prompt(prompt, response_a, response_b)
        if self.provider == "openai":
            messages: List[Dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            else:
                messages.append({"role": "system", "content": DEFAULT_PROMPT_TEMPLATE})
            messages.append({"role": "user", "content": formatted})

            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model_name or "gpt-4o-mini",
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                    return self._parse_choice(text)
                except (openai.APIConnectionError, openai.APIError):
                    if attempt < max_attempts - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise
        elif self.provider == "gemini":
            response = self.gemini_client.generate_content(formatted)
            text = getattr(response, "text", "") or ""
            if not text and getattr(response, "candidates", None):
                try:
                    text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    text = ""
            return self._parse_choice(text)

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )
            generated = outputs[0][inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_choice(text)

