from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from llm_config import get_env

from .bundle import parse_file_bundle, validate_initial_bundle
from .llm import OpenAICompatibleProvider
from .logging import TrajectoryLogger
from .models import LLMCallConfig, LLMResponse


def _write_private_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return sorted(
        str(item["id"])
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


def _response_summary(
    response: LLMResponse,
    *,
    content_path: Path,
    reasoning_path: Path | None,
    root: Path,
) -> dict[str, Any]:
    usage = response.usage if isinstance(response.usage, dict) else {}
    return {
        "requested_model": response.requested_model,
        "returned_model": response.model,
        "model_route_mismatch": bool(
            response.requested_model and response.requested_model != response.model
        ),
        "request_id": response.request_id,
        "system_fingerprint": response.system_fingerprint,
        "finish_reason": response.finish_reason,
        "latency_seconds": response.latency_seconds,
        "usage": usage,
        "content_chars": len(response.content),
        "reasoning_chars": len(response.reasoning_content),
        "content_path": content_path.relative_to(root).as_posix(),
        "reasoning_path": reasoning_path.relative_to(root).as_posix() if reasoning_path else None,
        "content_sha256": hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
        "reasoning_sha256": (
            hashlib.sha256(response.reasoning_content.encode("utf-8")).hexdigest()
            if response.reasoning_content
            else None
        ),
    }


def run_diagnostics(
    provider: Any,
    *,
    output_dir: Path,
    real_prompt: str,
    model: str,
    reasoning_probe_max_tokens: int = 8192,
    generation_max_tokens: int = 65536,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"diagnostic output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrajectoryLogger(
        output_dir,
        {
            "kind": "llm_diagnostic",
            "requested_model": model,
            "reasoning_probe_max_tokens": reasoning_probe_max_tokens,
            "generation_max_tokens": generation_max_tokens,
        },
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "requested_model": model,
        "model_catalog": {},
        "steps": [],
        "success": False,
    }

    try:
        catalog = provider.list_models()
        ids = _model_ids(catalog)
        _write_private_json(output_dir / "models.json", catalog)
        manifest["model_catalog"] = {
            "status": "ok",
            "model_ids": ids,
            "requested_model_present": model in ids,
            "path": "models.json",
        }
    except Exception as error:  # noqa: BLE001 - diagnostics must continue through independent probes
        manifest["model_catalog"] = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }

    minimal_prompt = (
        '只返回一个 JSON 对象：{"ok":true,"purpose":"deepseek api diagnostic"}。'
        "不要使用 Markdown，不要增加其他字段。"
    )
    probes = [
        (
            "minimal_nonthinking",
            minimal_prompt,
            LLMCallConfig("diagnostic", 512, thinking="disabled"),
            "json",
        ),
        (
            "minimal_thinking_low",
            minimal_prompt,
            LLMCallConfig("diagnostic", 4096, thinking="enabled", reasoning_effort="low"),
            "json",
        ),
        (
            "real_prompt_thinking_high",
            real_prompt,
            LLMCallConfig(
                "generator",
                reasoning_probe_max_tokens,
                thinking="enabled",
                reasoning_effort="high",
            ),
            "observation",
        ),
        (
            "real_prompt_nonthinking",
            real_prompt,
            LLMCallConfig("generator", generation_max_tokens, thinking="disabled"),
            "bundle",
        ),
    ]
    for index, (label, prompt, call_config, validation) in enumerate(probes, 1):
        step: dict[str, Any] = {
            "label": label,
            "request_options": call_config.__dict__,
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "validation": validation,
            "valid": False,
        }
        try:
            response = provider.generate(prompt, call_config=call_config)
            response_path = output_dir / label / "response.txt"
            content_path, reasoning_path = logger.persist_response_artifacts(
                response_path,
                content=response.content,
                reasoning_content=response.reasoning_content,
                reasoning_log_mode="full",
            )
            logger.save_call(
                index,
                response.to_dict(),
                call_type=call_config.call_type,
                evaluation_round=index,
                prompt_metadata={
                    "diagnostic_label": label,
                    "prompt_chars": len(prompt),
                    "prompt_sha256": step["prompt_sha256"],
                },
                response_content_path=content_path,
                reasoning_content_path=reasoning_path,
            )
            step.update(
                _response_summary(
                    response,
                    content_path=content_path,
                    reasoning_path=reasoning_path,
                    root=output_dir,
                )
            )
            if validation == "json":
                payload = json.loads(response.content)
                step["valid"] = isinstance(payload, dict) and response.finish_reason != "length"
            elif validation == "bundle":
                bundle = parse_file_bundle(response.content)
                validate_initial_bundle(bundle)
                step["valid"] = response.finish_reason != "length"
                step["bundle_file_count"] = len(bundle.files)
            else:
                step["valid"] = bool(response.reasoning_content or response.content)
        except Exception as error:  # noqa: BLE001 - preserve every independent diagnostic result
            step["error"] = f"{type(error).__name__}: {error}"
        manifest["steps"].append(step)
        _write_private_json(output_dir / "manifest.json", manifest)

    catalog_ok = bool(manifest["model_catalog"].get("requested_model_present"))
    required_labels = {"minimal_nonthinking", "minimal_thinking_low", "real_prompt_nonthinking"}
    required_ok = all(
        item.get("valid")
        for item in manifest["steps"]
        if item.get("label") in required_labels
    ) and required_labels.issubset({item.get("label") for item in manifest["steps"]})
    finite_latencies = all(
        math.isfinite(float(item["latency_seconds"])) and float(item["latency_seconds"]) >= 0
        for item in manifest["steps"]
        if "latency_seconds" in item
    )
    manifest["success"] = bool(catalog_ok and required_ok and finite_latencies)
    _write_private_json(output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Diagnose DeepSeek model routing, thinking output, and real Generator behavior"
    )
    result.add_argument("--output-dir", required=True)
    result.add_argument("--prompt-file", required=True)
    result.add_argument("--model", default="deepseek-flash")
    result.add_argument("--base-url", default=None)
    result.add_argument("--timeout", type=int, default=1800)
    result.add_argument("--reasoning-probe-max-tokens", type=int, default=8192)
    result.add_argument("--generation-max-tokens", type=int, default=65536)
    return result


def main() -> int:
    args = parser().parse_args()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    if not prompt_path.is_file():
        raise SystemExit(f"prompt file does not exist: {prompt_path}")
    provider = OpenAICompatibleProvider(
        model=args.model,
        base_url=args.base_url or get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=get_env("DEEPSEEK_API_KEY"),
        timeout=args.timeout,
        provider="deepseek",
    )
    manifest = run_diagnostics(
        provider,
        output_dir=Path(args.output_dir),
        real_prompt=prompt_path.read_text(encoding="utf-8"),
        model=args.model,
        reasoning_probe_max_tokens=args.reasoning_probe_max_tokens,
        generation_max_tokens=args.generation_max_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
