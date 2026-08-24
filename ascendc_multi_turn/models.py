from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    request_id: str | None = None

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
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_rounds: int = 3
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 600
    device: int = 0
    soc_version: str = "Ascend910B3"
    evaluator: str = "local"
    resume: bool = False
    mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
