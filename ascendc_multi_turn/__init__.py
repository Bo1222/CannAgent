"""Provider-neutral multi-turn AscendC kernel generation."""

from .models import EvalResult, FileBundle, LLMResponse, RunConfig


def __getattr__(name: str):
    if name == "MultiTurnRunner":
        from .runner import MultiTurnRunner

        return MultiTurnRunner
    raise AttributeError(name)

__all__ = [
    "EvalResult",
    "FileBundle",
    "LLMResponse",
    "MultiTurnRunner",
    "RunConfig",
]
