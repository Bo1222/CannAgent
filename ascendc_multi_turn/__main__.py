from __future__ import annotations

import argparse
import json

from llm_config import get_env

from .evaluator import LocalAscendEvaluator, MockEvaluator
from .llm import MockProvider, OpenAICompatibleProvider
from .models import RunConfig
from .runner import MultiTurnRunner


def parser() -> argparse.ArgumentParser:
    provider = get_env("LLM_PROVIDER", "deepseek").lower()
    if provider not in {"deepseek", "openai"}:
        raise RuntimeError("LLM_PROVIDER must be 'deepseek' or 'openai'")
    result = argparse.ArgumentParser(description="Generate and optimize AscendC operators through DeepSeek or OpenAI")
    result.add_argument("--op-file", required=True, help="reference model.py; a same-stem .json file is copied when present")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--provider", choices=("deepseek", "openai"), default=provider)
    result.add_argument("--model", default=None, help="override the selected provider's model")
    result.add_argument("--base-url", default=None, help="override the selected provider's base URL")
    result.add_argument("--max-rounds", type=int, default=3)
    result.add_argument("--temperature", type=float, default=float(get_env("ASCENDC_LLM_TEMPERATURE", "0.2")))
    result.add_argument("--max-tokens", type=int, default=int(get_env("ASCENDC_LLM_MAX_TOKENS", "8192")))
    result.add_argument("--timeout", type=int, default=int(get_env("ASCENDC_LLM_TIMEOUT", "600")))
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--soc-version", default="Ascend910B3")
    result.add_argument("--cann-version", default="auto", help="installed CANN version or 'auto'")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--mock", action="store_true", help="exercise orchestration without API calls, compilation, or NPU")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be at least 1")
    prefix = "DEEPSEEK" if args.provider == "deepseek" else "OPENAI"
    model = args.model or get_env(f"{prefix}_MODEL", "deepseek-chat" if prefix == "DEEPSEEK" else "gpt-4.1")
    base_url = args.base_url or get_env(
        f"{prefix}_BASE_URL",
        "https://api.deepseek.com" if prefix == "DEEPSEEK" else "https://api.openai.com/v1",
    )
    config = RunConfig(
        op_file=args.op_file,
        output_dir=args.output_dir,
        provider=args.provider,
        model=model,
        base_url=base_url,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        device=args.device,
        soc_version=args.soc_version,
        cann_version=args.cann_version,
        evaluator="mock" if args.mock else "local",
        resume=args.resume,
        mock=args.mock,
    )
    if not args.mock:
        prefix = "DEEPSEEK" if config.provider == "deepseek" else "OPENAI"
        try:
            get_env(f"{prefix}_API_KEY")
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not config.model or not config.base_url:
            raise SystemExit(f"{config.provider} model and base URL must be configured in .env or CLI arguments")
    provider = MockProvider() if args.mock else OpenAICompatibleProvider.from_env(config)
    evaluator = MockEvaluator() if args.mock else LocalAscendEvaluator(device=args.device, soc_version=args.soc_version, timeout=args.timeout)
    summary = MultiTurnRunner(config, provider, evaluator).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
