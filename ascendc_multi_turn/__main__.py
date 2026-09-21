from __future__ import annotations

import argparse
import json
import sys

from llm_config import get_env

from .devkit import DevkitError, resolve_devkit, resolve_runtime_version
from .evaluator import LocalAscendEvaluator, MockEvaluator
from .llm import MockProvider, OpenAICompatibleProvider
from .models import RunConfig
from .progress import ProgressReporter
from .runner import MultiTurnRunner


def parser() -> argparse.ArgumentParser:
    provider = get_env("LLM_PROVIDER", "deepseek").lower()
    if provider not in {"deepseek", "openai"}:
        raise RuntimeError("LLM_PROVIDER must be 'deepseek' or 'openai'")
    result = argparse.ArgumentParser(description="Generate and optimize AscendC operators through DeepSeek or OpenAI")
    result.add_argument("--op-name", required=True, help="safe C/C++ operator identifier used by the fixed project template")
    result.add_argument("--op-file", required=True, help="reference model.py and authoritative Model.forward ABI")
    result.add_argument("--op-json", default="", help="JSON/JSONL cases; defaults to the same stem as --op-file")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--provider", choices=("deepseek", "openai"), default=provider)
    result.add_argument("--model", default=None, help="override the selected provider's model")
    result.add_argument("--base-url", default=None, help="override the selected provider's base URL")
    result.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="candidate evaluations after a correct, benchmarked baseline exists",
    )
    result.add_argument(
        "--max-bootstrap-rounds",
        type=int,
        default=8,
        help="candidate attempts allowed while establishing a correct benchmarked baseline",
    )
    result.add_argument(
        "--max-total-rounds",
        type=int,
        default=None,
        help="optional cap on all completed candidate evaluations across bootstrap and optimization",
    )
    result.add_argument("--temperature", type=float, default=float(get_env("ASCENDC_LLM_TEMPERATURE", "0.2")))
    result.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="deprecated blanket output-token override for every LLM call",
    )
    result.add_argument("--router-max-tokens", type=int, default=None)
    result.add_argument("--generator-max-tokens", type=int, default=None)
    result.add_argument("--planner-max-tokens", type=int, default=None)
    result.add_argument("--repair-max-tokens", type=int, default=None)
    result.add_argument(
        "--router-thinking", choices=("enabled", "disabled"),
        default=get_env("ASCENDC_ROUTER_THINKING", "disabled"),
    )
    result.add_argument(
        "--generator-thinking", choices=("enabled", "disabled"),
        default=get_env("ASCENDC_GENERATOR_THINKING", "disabled"),
    )
    result.add_argument(
        "--reasoning-log",
        choices=("full", "metadata"),
        default=get_env("ASCENDC_REASONING_LOG_MODE", "full"),
        help="persist full reasoning text locally or retain metadata only",
    )
    result.add_argument(
        "--planner-thinking", choices=("enabled", "disabled"),
        default=get_env("ASCENDC_PLANNER_THINKING", "disabled"),
    )
    result.add_argument(
        "--repair-thinking", choices=("enabled", "disabled"),
        default=get_env("ASCENDC_REPAIR_THINKING", "enabled"),
    )
    result.add_argument(
        "--generator-reasoning-effort", choices=("low", "high", "max"),
        default=get_env("ASCENDC_GENERATOR_REASONING_EFFORT", "high"),
    )
    result.add_argument(
        "--planner-reasoning-effort", choices=("low", "high", "max"),
        default=get_env("ASCENDC_PLANNER_REASONING_EFFORT", "low"),
    )
    result.add_argument(
        "--repair-reasoning-effort", choices=("low", "high", "max"),
        default=get_env("ASCENDC_REPAIR_REASONING_EFFORT", "max"),
    )
    result.add_argument(
        "--llm-transient-retries", type=int,
        default=int(get_env("ASCENDC_LLM_TRANSIENT_RETRIES", "3")),
    )
    result.add_argument("--timeout", type=int, default=int(get_env("ASCENDC_LLM_TIMEOUT", "600")))
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--soc-version", default="Ascend910B3")
    result.add_argument("--cann-version", default="auto", help="installed CANN version or 'auto'")
    result.add_argument(
        "--asc-devkit-dir",
        default="",
        help="healthy Asc DevKit 9.1.0 checkout; otherwise use the managed cache",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--quiet", action="store_true", help="suppress progress and failure details on stderr")
    result.add_argument("--mock", action="store_true", help="exercise orchestration without API calls, compilation, or NPU")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be at least 1")
    if args.max_bootstrap_rounds < 1:
        raise SystemExit("--max-bootstrap-rounds must be at least 1")
    if args.max_total_rounds is not None and args.max_total_rounds < 1:
        raise SystemExit("--max-total-rounds must be at least 1")
    prefix = "DEEPSEEK" if args.provider == "deepseek" else "OPENAI"
    model = args.model or get_env(
        f"{prefix}_MODEL", "deepseek-flash" if prefix == "DEEPSEEK" else "gpt-4.1"
    )
    base_url = args.base_url or get_env(
        f"{prefix}_BASE_URL",
        "https://api.deepseek.com" if prefix == "DEEPSEEK" else "https://api.openai.com/v1",
    )
    config = RunConfig(
        op_name=args.op_name,
        op_file=args.op_file,
        op_json=args.op_json,
        output_dir=args.output_dir,
        provider=args.provider,
        model=model,
        base_url=base_url,
        max_rounds=args.max_rounds,
        max_bootstrap_rounds=args.max_bootstrap_rounds,
        max_total_rounds=args.max_total_rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        router_max_tokens=args.router_max_tokens,
        generator_max_tokens=args.generator_max_tokens,
        planner_max_tokens=args.planner_max_tokens,
        repair_max_tokens=args.repair_max_tokens,
        router_thinking=args.router_thinking,
        generator_thinking=args.generator_thinking,
        planner_thinking=args.planner_thinking,
        repair_thinking=args.repair_thinking,
        generator_reasoning_effort=args.generator_reasoning_effort,
        planner_reasoning_effort=args.planner_reasoning_effort,
        repair_reasoning_effort=args.repair_reasoning_effort,
        llm_transient_retries=args.llm_transient_retries,
        timeout=args.timeout,
        device=args.device,
        soc_version=args.soc_version,
        cann_version=args.cann_version,
        asc_devkit_dir=args.asc_devkit_dir,
        reasoning_log_mode=args.reasoning_log,
        evaluator="mock" if args.mock else "local",
        resume=args.resume,
        mock=args.mock,
    )
    config.deprecated_repair_options_active = any(
        option in sys.argv[1:]
        for option in (
            "--repair-max-tokens",
            "--repair-thinking",
            "--repair-reasoning-effort",
        )
    )
    if not args.mock:
        prefix = "DEEPSEEK" if config.provider == "deepseek" else "OPENAI"
        try:
            get_env(f"{prefix}_API_KEY")
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not config.model or not config.base_url:
            raise SystemExit(f"{config.provider} model and base URL must be configured in .env or CLI arguments")
    progress = ProgressReporter(enabled=not args.quiet)
    provider = MockProvider() if args.mock else OpenAICompatibleProvider.from_env(config)
    if not args.mock:
        try:
            config.asc_devkit_dir = str(resolve_devkit(config.asc_devkit_dir or None))
            runtime = resolve_runtime_version(config.cann_version)
            if runtime.status != "exact":
                raise DevkitError(runtime.warning)
        except DevkitError as error:
            raise SystemExit(str(error)) from error
    evaluator = (
        MockEvaluator()
        if args.mock
        else LocalAscendEvaluator(
            device=args.device,
            soc_version=args.soc_version,
            timeout=args.timeout,
            progress=progress,
        )
    )
    summary = MultiTurnRunner(config, provider, evaluator, progress=progress).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["success"] and summary.get("completed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
