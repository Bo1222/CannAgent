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
    reasoning_content: str = ""
    system_fingerprint: str | None = None
    requested_model: str | None = None
    request_options: dict[str, Any] = field(default_factory=dict)

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
    failure_stage: str | None = None
    failure_code: str | None = None
    error_excerpt: str = ""
    details_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMCallConfig:
    call_type: str
    max_tokens: int
    thinking: str = "disabled"
    reasoning_effort: str | None = None
    response_format: str = "json_object"

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("thinking must be 'enabled' or 'disabled'")
        if self.reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, max, or None")


@dataclass
class RunConfig:
    op_file: str
    output_dir: str
    model: str = ""
    base_url: str = ""
    provider: str = field(default_factory=lambda: get_env("LLM_PROVIDER", "deepseek"))
    max_rounds: int = 3
    temperature: float = field(default_factory=lambda: float(get_env("ASCENDC_LLM_TEMPERATURE", "0.2")))
    # max_tokens is the legacy blanket override. New callers should use the
    # per-call fields below so routing cannot consume the code-generation
    # output budget.
    max_tokens: int | None = None
    timeout: int = field(default_factory=lambda: int(get_env("ASCENDC_LLM_TIMEOUT", "600")))
    device: int = 0
    soc_version: str = "Ascend910B3"
    cann_version: str = "auto"
    evaluator: str = "local"
    resume: bool = False
    mock: bool = False
    router_max_tokens: int | None = None
    generator_max_tokens: int | None = None
    repair_max_tokens: int | None = None
    router_thinking: str = field(default_factory=lambda: get_env("ASCENDC_ROUTER_THINKING", "disabled"))
    generator_thinking: str = field(default_factory=lambda: get_env("ASCENDC_GENERATOR_THINKING", "enabled"))
    repair_thinking: str = field(default_factory=lambda: get_env("ASCENDC_REPAIR_THINKING", "enabled"))
    generator_reasoning_effort: str = field(
        default_factory=lambda: get_env("ASCENDC_GENERATOR_REASONING_EFFORT", "high")
    )
    repair_reasoning_effort: str = field(
        default_factory=lambda: get_env("ASCENDC_REPAIR_REASONING_EFFORT", "max")
    )
    llm_transient_retries: int = field(
        default_factory=lambda: int(get_env("ASCENDC_LLM_TRANSIENT_RETRIES", "3"))
    )
    legacy_max_tokens_active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.provider = self.provider.lower()
        if self.provider not in {"deepseek", "openai"}:
            raise ValueError("provider must be 'deepseek' or 'openai'")
        prefix = "DEEPSEEK" if self.provider == "deepseek" else "OPENAI"
        if not self.model:
            self.model = get_env(
                f"{prefix}_MODEL",
                "deepseek-v4-flash" if prefix == "DEEPSEEK" else "gpt-4.1",
            )
        if not self.base_url:
            self.base_url = get_env(
                f"{prefix}_BASE_URL",
                "https://api.deepseek.com" if prefix == "DEEPSEEK" else "https://api.openai.com/v1",
            )
        legacy_cli_limit = self.max_tokens
        legacy_env = get_env("ASCENDC_LLM_MAX_TOKENS", "").strip()
        legacy_env_limit = int(legacy_env) if legacy_env else None
        specific_overrides = (
            self.router_max_tokens is not None or bool(get_env("ASCENDC_ROUTER_MAX_TOKENS", "").strip()),
            self.generator_max_tokens is not None or bool(get_env("ASCENDC_GENERATOR_MAX_TOKENS", "").strip()),
            self.repair_max_tokens is not None or bool(get_env("ASCENDC_REPAIR_MAX_TOKENS", "").strip()),
        )
        self.legacy_max_tokens_active = legacy_cli_limit is not None or bool(
            legacy_env_limit is not None and not all(specific_overrides)
        )

        def resolve_limit(value: int | None, env_name: str, default: int) -> int:
            if value is not None:
                resolved = value
            elif legacy_cli_limit is not None:
                resolved = legacy_cli_limit
            else:
                env_value = get_env(env_name, "").strip()
                if env_value:
                    resolved = int(env_value)
                elif legacy_env_limit is not None:
                    resolved = legacy_env_limit
                else:
                    resolved = default
            if resolved < 1:
                raise ValueError(f"{env_name} must be at least 1")
            return resolved

        self.router_max_tokens = resolve_limit(
            self.router_max_tokens, "ASCENDC_ROUTER_MAX_TOKENS", 4096
        )
        self.generator_max_tokens = resolve_limit(
            self.generator_max_tokens, "ASCENDC_GENERATOR_MAX_TOKENS", 65536
        )
        self.repair_max_tokens = resolve_limit(
            self.repair_max_tokens, "ASCENDC_REPAIR_MAX_TOKENS", 65536
        )
        if self.llm_transient_retries < 0:
            raise ValueError("llm_transient_retries must be non-negative")
        for name in ("router_thinking", "generator_thinking", "repair_thinking"):
            value = getattr(self, name).lower()
            if value not in {"enabled", "disabled"}:
                raise ValueError(f"{name} must be 'enabled' or 'disabled'")
            setattr(self, name, value)
        for name in ("generator_reasoning_effort", "repair_reasoning_effort"):
            value = getattr(self, name).lower()
            if value not in {"low", "high", "max"}:
                raise ValueError(f"{name} must be low, high, or max")
            setattr(self, name, value)

    def call_config(self, call_type: str) -> LLMCallConfig:
        if call_type == "knowledge_router":
            return LLMCallConfig(
                call_type=call_type,
                max_tokens=int(self.router_max_tokens),
                thinking=self.router_thinking,
            )
        if call_type in {"compile_repair"}:
            return LLMCallConfig(
                call_type=call_type,
                max_tokens=int(self.repair_max_tokens),
                thinking=self.repair_thinking,
                reasoning_effort=self.repair_reasoning_effort,
            )
        return LLMCallConfig(
            call_type=call_type,
            max_tokens=int(self.generator_max_tokens),
            thinking=self.generator_thinking,
            reasoning_effort=self.generator_reasoning_effort,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
