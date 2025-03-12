"""LLM provider abstraction layer.

Supports OpenAI-compatible APIs (including self-hosted vLLM, Ollama, etc.)
and HuggingFace Inference API. The provider interface is minimal by design --
classification prompts are constructed upstream and providers just handle
the transport and response parsing.

In practice, most T&S teams start with OpenAI for prototyping and migrate
to self-hosted models (DeepSeek, Llama) for cost and data sovereignty.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from crisis_lens.config import LLMProviderConfig


class LLMResponse(dict[str, Any]):
    """Thin wrapper around parsed LLM JSON output."""

    @property
    def raw_text(self) -> str:
        return str(self.get("_raw_text", ""))


class LLMProvider(ABC):
    """Abstract base for LLM inference providers."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        ...

    def _parse_json_response(self, text: str) -> LLMResponse:
        """Extract JSON from LLM response, handling markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Drop first and last fence lines
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.strip() == "```" and in_block:
                    break
                if in_block:
                    json_lines.append(line)
            cleaned = "\n".join(json_lines)

        try:
            parsed = json.loads(cleaned)
            result = LLMResponse(parsed)
            result["_raw_text"] = text
            return result
        except json.JSONDecodeError:
            return LLMResponse({"_raw_text": text, "_parse_error": True})


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI-compatible chat completion APIs.

    Works with OpenAI, Azure OpenAI, vLLM, Ollama, and any endpoint
    that implements the /v1/chat/completions interface.
    """

    def __init__(self, config: LLMProviderConfig | None = None):
        config = config or LLMProviderConfig()
        super().__init__(config)
        self.api_key = os.environ.get(config.api_key_env, "")
        self.api_base = config.api_base or "https://api.openai.com/v1"

    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return self._parse_json_response(content)


class HuggingFaceProvider(LLMProvider):
    """Provider for HuggingFace Inference API.

    Designed for models hosted on HF Inference Endpoints or the free
    Inference API. Supports text-generation models with chat templates.
    """

    def __init__(self, config: LLMProviderConfig | None = None):
        config = config or LLMProviderConfig(
            provider="huggingface",
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            api_key_env="HF_API_KEY",
        )
        super().__init__(config)
        self.api_key = os.environ.get(config.api_key_env, "")
        self.api_base = config.api_base or "https://api-inference.huggingface.co/models"

    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.api_base}/{self.config.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "inputs": combined_prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens,
                "return_full_text": False,
            },
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            content = data[0].get("generated_text", "")
        else:
            content = str(data)

        return self._parse_json_response(content)


def create_provider(config: LLMProviderConfig | None = None) -> LLMProvider:
    """Factory for creating the appropriate LLM provider."""
    config = config or LLMProviderConfig()
    providers: dict[str, type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
    }
    provider_cls = providers.get(config.provider)
    if provider_cls is None:
        raise ValueError(
            f"Unknown provider: {config.provider}. Supported: {list(providers.keys())}"
        )
    return provider_cls(config)
