from __future__ import annotations

import argparse
import json

from .evaluator import LocalAscendEvaluator, MockEvaluator
from .llm import MockProvider, OpenAICompatibleProvider
from .models import RunConfig
from .runner import MultiTurnRunner


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Generate and optimize AscendC operators through a direct LLM API")
    result.add_argument("--op-file", required=True, help="reference model.py; a same-stem .json file is copied when present")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--model", default="deepseek-chat")
    result.add_argument("--base-url", default="https://api.deepseek.com")
    result.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    result.add_argument("--max-rounds", type=int, default=3)
    result.add_argument("--temperature", type=float, default=0.2)
    result.add_argument("--max-tokens", type=int, default=8192)
    result.add_argument("--timeout", type=int, default=600)
    result.add_argument("--device", type=int, default=0)
    result.add_argument("--soc-version", default="Ascend910B3")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--mock", action="store_true", help="exercise orchestration without API calls, compilation, or NPU")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be at least 1")
    config = RunConfig(
        op_file=args.op_file,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        device=args.device,
        soc_version=args.soc_version,
        evaluator="mock" if args.mock else "local",
        resume=args.resume,
        mock=args.mock,
    )
    provider = MockProvider() if args.mock else OpenAICompatibleProvider.from_env(config)
    evaluator = MockEvaluator() if args.mock else LocalAscendEvaluator(device=args.device, soc_version=args.soc_version, timeout=args.timeout)
    summary = MultiTurnRunner(config, provider, evaluator).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
