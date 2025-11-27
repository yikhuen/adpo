from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Protocol

import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT_TEMPLATE = "Prompt:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\nWhich is better? Reply with only A or B."


class _JudgeBackend(Protocol):
    def generate(self, formatted_prompt: str) -> str:
        ...


class _OpenAIBackend:
    def __init__(
        self,
        *,
        client: openai.OpenAI,
        model_name: Optional[str],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        default_system_prompt: str,
        max_attempts: int,
    ):
        self.client = client
        self.model_name = model_name or "gpt-4o-mini"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or default_system_prompt
        self.max_attempts = max(1, max_attempts)

    def generate(self, formatted_prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]
        for attempt in range(self.max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except (openai.APIConnectionError, openai.APIError):
                if attempt < self.max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise


class _GeminiBackend:
    def __init__(self, *, client, temperature: float, max_tokens: int):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, formatted_prompt: str) -> str:
        response = self.client.generate_content(formatted_prompt)
        text = getattr(response, "text", "") or ""
        if not text and getattr(response, "candidates", None):
            try:
                text = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                text = ""
        return text


class _HFCausalBackend:
    def __init__(self, *, model_name: str, temperature: float, max_tokens: int):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
        dtype = torch.float16 if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
        )
        self.model.eval()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, formatted_prompt: str) -> str:
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )
        generated = outputs[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class PairwiseJudge:
    def __init__(self, cfg: Dict[str, Any]):
        self.name = cfg.get("name") or cfg.get("model")
        self.provider = cfg.get("provider", "openai")
        self.model_name = cfg.get("model")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_tokens = int(cfg.get("max_tokens", 64))
        self.system_prompt = cfg.get("system_prompt")
        self.prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self.backend = self._build_backend(cfg)

    def _build_backend(self, cfg: Dict[str, Any]) -> _JudgeBackend:
        if self.provider == "openai":
            api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set for OpenAI judge.")
            timeout = cfg.get("timeout", 60.0)
            max_retries = int(cfg.get("max_retries", 3))
            client = openai.OpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
            return _OpenAIBackend(
                client=client,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system_prompt=self.system_prompt,
                default_system_prompt=DEFAULT_PROMPT_TEMPLATE,
                max_attempts=max_retries,
            )
        if self.provider == "gemini":
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
            client = genai.GenerativeModel(model_name, **init_kwargs)
            return _GeminiBackend(client=client, temperature=self.temperature, max_tokens=self.max_tokens)
        if self.provider == "hf_causal":
            if not self.model_name:
                raise ValueError("hf_causal judge requires 'model' identifier.")
            return _HFCausalBackend(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
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
        raw = self.backend.generate(formatted)
        return self._parse_choice(raw)

