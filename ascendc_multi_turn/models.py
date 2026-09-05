from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from llm_config import get_env


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    request_id: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileBundle:
    files: dict[str, str]
    delete: list[str] = field(default_factory=list)
    analysis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    compiled: bool
    correctness: bool
    score: float | None = None
    compile_output: str = ""
    verify_output: str = ""
    performance: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    op_file: str
    output_dir: str
    model: str = ""
    base_url: str = ""
    provider: str = field(default_factory=lambda: get_env("LLM_PROVIDER", "deepseek"))
    max_rounds: int = 3
    temperature: float = field(default_factory=lambda: float(get_env("ASCENDC_LLM_TEMPERATURE", "0.2")))
    max_tokens: int = field(default_factory=lambda: int(get_env("ASCENDC_LLM_MAX_TOKENS", "8192")))
    timeout: int = field(default_factory=lambda: int(get_env("ASCENDC_LLM_TIMEOUT", "600")))
    device: int = 0
    soc_version: str = "Ascend910B3"
    cann_version: str = "auto"
    evaluator: str = "local"
    resume: bool = False
    mock: bool = False

    def __post_init__(self) -> None:
        self.provider = self.provider.lower()
        if self.provider not in {"deepseek", "openai"}:
            raise ValueError("provider must be 'deepseek' or 'openai'")
        prefix = "DEEPSEEK" if self.provider == "deepseek" else "OPENAI"
        if not self.model:
            self.model = get_env(f"{prefix}_MODEL", "deepseek-chat" if prefix == "DEEPSEEK" else "gpt-4.1")
        if not self.base_url:
            self.base_url = get_env(
                f"{prefix}_BASE_URL",
                "https://api.deepseek.com" if prefix == "DEEPSEEK" else "https://api.openai.com/v1",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
