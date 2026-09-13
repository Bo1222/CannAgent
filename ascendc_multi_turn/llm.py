from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol

from llm_config import get_env

from .models import LLMCallConfig, LLMResponse

SYSTEM_PROMPT_ID = "ascendc-json-engineer-v1"
SYSTEM_PROMPT = (
    "You are an expert AscendC kernel engineer. "
    "Return exactly one valid JSON object matching the requested schema."
)


def system_prompt_for(_call_type: str) -> str:
    """Single source of truth for the actual provider system message."""

    return SYSTEM_PROMPT


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, call_config: LLMCallConfig | None = None) -> LLMResponse: ...


class OpenAICompatibleProvider:
    """Client for DeepSeek and OpenAI Chat Completions APIs."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 65536,
        timeout: int = 600,
        provider: str = "deepseek",
    ):
        if not api_key:
            raise ValueError("LLM API key is empty")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = provider

    @classmethod
    def from_env(cls, config):
        prefix = "DEEPSEEK" if config.provider == "deepseek" else "OPENAI"
        return cls(
            model=config.model,
            base_url=config.base_url,
            api_key=get_env(f"{prefix}_API_KEY"),
            temperature=config.temperature,
            max_tokens=(
                getattr(config, "generator_max_tokens", None)
                or getattr(config, "max_tokens", None)
                or 65536
            ),
            timeout=config.timeout,
            provider=config.provider,
        )

    def generate(self, prompt: str, *, call_config: LLMCallConfig | None = None) -> LLMResponse:
        effective = call_config or LLMCallConfig(
            call_type="generator",
            max_tokens=self.max_tokens,
            thinking="disabled",
        )
        request_body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt_for(effective.call_type),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": effective.max_tokens,
            "response_format": {"type": effective.response_format},
        }
        request_options: dict[str, object] = {
            "call_type": effective.call_type,
            "max_tokens": effective.max_tokens,
            "response_format": effective.response_format,
            "thinking_requested": effective.thinking,
            "reasoning_effort_requested": effective.reasoning_effort,
        }
        if self.provider == "deepseek":
            request_body["thinking"] = {"type": effective.thinking}
            if effective.thinking == "enabled" and effective.reasoning_effort:
                request_body["reasoning_effort"] = effective.reasoning_effort
            else:
                request_body["temperature"] = self.temperature
        else:
            # DeepSeek's thinking wire fields are not portable to generic
            # OpenAI-compatible endpoints.
            request_body["temperature"] = self.temperature
            request_options["thinking_requested"] = "not_sent"
            request_options["reasoning_effort_requested"] = None
        body = json.dumps(request_body).encode("utf-8")
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
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"invalid LLM API response: {str(payload)[:2000]}") from error
        if not isinstance(content, str) or not isinstance(reasoning_content, str):
            raise RuntimeError(f"invalid LLM message content: {str(message)[:2000]}")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return LLMResponse(
            content=content,
            model=str(payload.get("model", self.model)),
            usage=usage,
            latency_seconds=elapsed,
            request_id=payload.get("id"),
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None,
            reasoning_content=reasoning_content,
            system_fingerprint=payload.get("system_fingerprint"),
            requested_model=self.model,
            request_options=request_options,
        )


class MockProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, *, call_config: LLMCallConfig | None = None) -> LLMResponse:
        self.calls += 1
        if call_config and call_config.call_type in {"planner", "diagnose"}:
            initial = "Return exactly one complete baseline item" in prompt
            content = json.dumps(
                {
                    "diagnosis": (
                        "mock direct AscendC baseline plan"
                        if initial
                        else "mock evidence-driven plan"
                    ),
                    "items": [
                        {
                            "id": "bootstrap-1" if initial else f"p{index}",
                            "kind": "correctness" if initial else "performance",
                            "hypothesis": f"mock hypothesis {index}",
                            "change": (
                                "implement one complete direct AscendC baseline"
                                if initial
                                else f"apply mock change {index}"
                            ),
                            "expected_signal": (
                                "valid compiled and benchmarked candidate"
                                if initial
                                else "higher valid score"
                            ),
                        }
                        for index in range(1, 2)
                    ],
                }
            )
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
                requested_model="mock",
                request_options={
                    "call_type": call_config.call_type,
                    "max_tokens": call_config.max_tokens,
                    "thinking_requested": call_config.thinking,
                    "reasoning_effort_requested": call_config.reasoning_effort,
                },
            )
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
            requested_model="mock",
            request_options={
                "call_type": call_config.call_type if call_config else "generator",
                "max_tokens": call_config.max_tokens if call_config else None,
                "thinking_requested": call_config.thinking if call_config else "disabled",
                "reasoning_effort_requested": call_config.reasoning_effort if call_config else None,
            },
        )
