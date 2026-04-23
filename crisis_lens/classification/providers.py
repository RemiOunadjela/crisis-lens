"""LLM provider abstraction layer.

Supports OpenAI-compatible APIs (including self-hosted vLLM, Ollama, etc.)
and HuggingFace Inference API. The provider interface is minimal by design --
classification prompts are constructed upstream and providers just handle
the transport and response parsing.

In practice, most T&S teams start with OpenAI for prototyping and migrate
to self-hosted models (DeepSeek, Llama) for cost and data sovereignty.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any

import httpx

from crisis_lens.config import LLMProviderConfig

# HTTP status codes that warrant a retry (transient server-side errors and rate limits)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


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
    async def _complete_once(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Single attempt at LLM completion -- no retry logic."""
        ...

    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the LLM with exponential backoff retry on transient errors.

        Retries on network timeouts, connection errors, rate limits (429), and
        server errors (5xx). Raises immediately for client errors (4xx except 429).
        """
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._complete_once(system_prompt, user_prompt)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in _RETRYABLE_STATUS_CODES:
                    last_exc = exc
                else:
                    raise

            if attempt < self.config.max_retries:
                delay = min(
                    self.config.retry_base_delay * (2**attempt),
                    self.config.retry_max_delay,
                )
                # Add ±25% jitter to avoid thundering herd on shared endpoints
                delay *= 0.75 + random.random() * 0.5
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

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

    async def _complete_once(self, system_prompt: str, user_prompt: str) -> LLMResponse:
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

    async def _complete_once(self, system_prompt: str, user_prompt: str) -> LLMResponse:
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
