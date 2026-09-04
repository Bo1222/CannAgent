from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

from llm_config import get_env

from .models import LLMResponse


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> LLMResponse: ...


class OpenAICompatibleProvider:
    """Client for DeepSeek and OpenAI Chat Completions APIs."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        timeout: int = 600,
    ):
        if not api_key:
            raise ValueError("LLM API key is empty")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @classmethod
    def from_env(cls, config):
        prefix = "DEEPSEEK" if config.provider == "deepseek" else "OPENAI"
        return cls(
            model=config.model,
            base_url=config.base_url,
            api_key=get_env(f"{prefix}_API_KEY"),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    def generate(self, prompt: str) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert AscendC kernel engineer. Follow the requested JSON schema exactly."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {error.code}: {detail[:2000]}") from error
        elapsed = time.monotonic() - started
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"invalid LLM API response: {str(payload)[:2000]}") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM API returned empty content")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return LLMResponse(
            content=content,
            model=str(payload.get("model", self.model)),
            usage=usage,
            latency_seconds=elapsed,
            request_id=payload.get("id"),
        )


class MockProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1
        module = "_mock_ascendc_ext"
        payload = {
            "analysis": f"mock generation round {self.calls}",
            "files": [
                {
                    "path": "model_new_ascendc.py",
                    "content": (
                        "import torch\nimport torch.nn as nn\n\n"
                        "class ModelNew(nn.Module):\n"
                        "    def forward(self, x, *args):\n"
                        "        return x\n"
                    ),
                },
                {
                    "path": "kernel/pybind11.cpp",
                    "content": f"#include <pybind11/pybind11.h>\nPYBIND11_MODULE({module}, m) {{}}\n",
                },
                {
                    "path": "kernel/mock.cpp",
                    "content": f"// mock AscendC source, round {self.calls}\n",
                },
            ],
            "delete": [],
        }
        content = json.dumps(payload)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(content) // 4)
        return LLMResponse(
            content=content,
            model="mock",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )
