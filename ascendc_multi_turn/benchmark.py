from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import inspect
import json
import math
import shutil
import statistics
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCHEMA_VERSION = 2
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 20
DEFAULT_PROFILE_TOP_K = 3
PROFILE_SKIP_FIRST = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 5


def _persist_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _find_model_class(module: Any, preferred_name: str):
    preferred = getattr(module, preferred_name, None)
    if inspect.isclass(preferred) and issubclass(preferred, nn.Module):
        return preferred
    for value in vars(module).values():
        if inspect.isclass(value) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise AttributeError(f"No nn.Module subclass found in {module.__file__}")


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _summarize_input(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, list):
        return [_summarize_input(item) for item in value[:16]]
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_summarize_input(item) for item in value[:16]]}
    if isinstance(value, dict):
        return {
            str(key): _summarize_input(item)
            for key, item in list(value.items())[:16]
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": type(value).__name__}


def _get_input_groups(module: Any) -> list[list[Any]]:
    if hasattr(module, "get_input_groups"):
        groups = module.get_input_groups()
        if not isinstance(groups, list) or not groups:
            raise ValueError("get_input_groups() must return a non-empty list")
        return groups
    if hasattr(module, "get_inputs"):
        inputs = module.get_inputs()
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("get_inputs() must return a non-empty list")
        return [inputs]
    raise AttributeError(f"Neither get_input_groups() nor get_inputs() found in {module.__file__}")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("latency samples must not be empty")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_statistics(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms or any(not math.isfinite(value) or value <= 0 for value in samples_ms):
        raise ValueError("latency samples must be finite positive numbers")
    mean = statistics.fmean(samples_ms)
    stddev = statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "sample_count": len(samples_ms),
        "mean_ms": mean,
        "median_ms": statistics.median(samples_ms),
        "p90_ms": _percentile(samples_ms, 0.9),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "stddev_ms": stddev,
        "coefficient_of_variation": stddev / mean if mean else 0.0,
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _require_npu() -> torch.device:
    import torch_npu  # noqa: F401

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable; benchmark refuses CPU/CUDA fallback")
    return torch.device("npu")


def _peak_memory_mb() -> float:
    getter = getattr(torch.npu, "max_memory_allocated", None)
    return round(float(getter()) / (1024 * 1024), 3) if callable(getter) else 0.0


def _reset_peak_memory() -> None:
    reset = getattr(torch.npu, "reset_peak_memory_stats", None)
    if callable(reset):
        reset()


def _measure_with_events(
    model: nn.Module,
    inputs: list[Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, float | int], float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        torch.npu.synchronize()
        _reset_peak_memory()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
        for index in range(repeats):
            starts[index].record()
            model(*inputs)
            ends[index].record()
        torch.npu.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    return latency_statistics(samples), _peak_memory_mb()


def _find_file(root: Path, name: str) -> Path | None:
    return next((path for path in root.rglob(name) if path.is_file()), None)


def parse_operator_details(path: Path, *, active_count: int) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"Name", "Device Self Duration(us)"}
        if not required.issubset(fields):
            raise ValueError(f"operator_details.csv lacks columns: {sorted(required - fields)}")
        for row in reader:
            name = str(row.get("Name") or "").strip()
            if not name:
                continue
            try:
                duration = float(row["Device Self Duration(us)"])
            except (TypeError, ValueError):
                continue
            try:
                count = float(row.get("Count") or 0)
            except (TypeError, ValueError):
                count = 0.0
            current = totals.setdefault(name, {"duration_us": 0.0, "count": 0.0})
            current["duration_us"] += duration
            current["count"] += count
    operators = [
        {
            "name": name,
            "mean_device_self_us": values["duration_us"] / active_count,
            "observed_count": int(values["count"]),
        }
        for name, values in totals.items()
    ]
    return sorted(operators, key=lambda item: (-item["mean_device_self_us"], item["name"]))[:12]


def _profile_step_count(*, skip_first: int, warmup: int, active: int) -> int:
    return skip_first + warmup + active


def _profile_model(model: nn.Module, inputs: list[Any], *, label: str) -> dict[str, Any]:
    import torch_npu

    profile_root = Path(tempfile.mkdtemp(prefix=f"cannagent-{label}-"))
    try:
        experimental = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=None,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )
        with (
            torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0,
                    warmup=PROFILE_WARMUP,
                    active=PROFILE_ACTIVE,
                    repeat=1,
                    skip_first=PROFILE_SKIP_FIRST,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(profile_root)
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental,
            ) as profiler,
            torch.inference_mode(),
        ):
            for _ in range(
                _profile_step_count(
                    skip_first=PROFILE_SKIP_FIRST,
                    warmup=PROFILE_WARMUP,
                    active=PROFILE_ACTIVE,
                )
            ):
                model(*inputs)
                profiler.step()
                torch.npu.synchronize()
        details = _find_file(profile_root, "operator_details.csv")
        if details is None:
            return {"status": "unavailable", "reason": "operator_details.csv was not produced"}
        return {
            "status": "ok",
            "active_count": PROFILE_ACTIVE,
            "operators": parse_operator_details(details, active_count=PROFILE_ACTIVE),
        }
    except Exception as error:  # noqa: BLE001 - profiler diagnostics must not invalidate timing
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    finally:
        shutil.rmtree(profile_root, ignore_errors=True)


def _base_report(task_dir: Path, *, warmup: int, repeats: int, seed: int, profile_top_k: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": "benchmark did not complete",
        "op": task_dir.name,
        "task_dir": str(task_dir),
        "device": "npu",
        "measurement": {
            "method": "torch.npu.Event",
            "latency_statistic": "median_ms",
            "score_aggregation": "geometric_mean",
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
        },
        "reference": {"model_path": str(task_dir / "model.py"), "case_results": []},
        "ascendc": {"model_path": str(task_dir / "model_new_ascendc.py"), "case_results": []},
        "per_case_speedup": [],
        "overall_speedup": None,
        "score": None,
        "bottlenecks": [],
        "diagnostics": {"status": "not_started", "profile_top_k": profile_top_k},
    }


def run_benchmark(
    task_dir: Path | str,
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
    profile_top_k: int = DEFAULT_PROFILE_TOP_K,
    output_path: Path | None = None,
    profiler: Callable[[nn.Module, list[Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1 or profile_top_k < 0:
        raise ValueError("warmup/profile-top-k must be non-negative and repeats must be positive")
    root = Path(task_dir).expanduser().resolve()
    report = _base_report(root, warmup=warmup, repeats=repeats, seed=seed, profile_top_k=profile_top_k)
    try:
        reference_path = root / "model.py"
        candidate_path = root / "model_new_ascendc.py"
        build_path = root / "build"
        for required in (reference_path, candidate_path, build_path):
            if not required.exists():
                raise FileNotFoundError(f"required benchmark path is missing: {required}")
        for path in (root, build_path):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        device = _require_npu()
        reference_module = _load_module(reference_path, f"benchmark_{root.name}_reference")
        candidate_module = _load_module(candidate_path, f"benchmark_{root.name}_candidate")
        reference_class = _find_model_class(reference_module, "Model")
        candidate_class = _find_model_class(candidate_module, "ModelNew")
        init_inputs = (
            candidate_module.get_init_inputs()
            if hasattr(candidate_module, "get_init_inputs")
            else getattr(reference_module, "get_init_inputs", list)()
        )
        input_groups = _get_input_groups(reference_module)

        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        reference_model = reference_class(*_clone(init_inputs)).to(device).eval()
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        candidate_model = candidate_class(*_clone(init_inputs)).to(device).eval()

        for index, raw_inputs in enumerate(input_groups):
            order = ("reference", "ascendc") if index % 2 == 0 else ("ascendc", "reference")
            measurements: dict[str, tuple[dict[str, float | int], float]] = {}
            for implementation in order:
                model = reference_model if implementation == "reference" else candidate_model
                model_inputs = _move_to_device(_clone(raw_inputs), device)
                measurements[implementation] = _measure_with_events(
                    model,
                    model_inputs,
                    warmup=warmup,
                    repeats=repeats,
                )
            reference_stats, reference_memory = measurements["reference"]
            candidate_stats, candidate_memory = measurements["ascendc"]
            reference_ms = float(reference_stats["median_ms"])
            candidate_ms = float(candidate_stats["median_ms"])
            speedup = reference_ms / candidate_ms
            input_summary = _summarize_input(raw_inputs)
            report["reference"]["case_results"].append(
                {
                    "index": index,
                    "latency": reference_stats,
                    "peak_memory_mb": reference_memory,
                }
            )
            report["ascendc"]["case_results"].append(
                {
                    "index": index,
                    "latency": candidate_stats,
                    "peak_memory_mb": candidate_memory,
                }
            )
            report["per_case_speedup"].append(
                {
                    "index": index,
                    "inputs": input_summary,
                    "measurement_order": list(order),
                    "reference_ms": reference_ms,
                    "ascendc_ms": candidate_ms,
                    "speedup": speedup,
                    "reference_cv": reference_stats["coefficient_of_variation"],
                    "ascendc_cv": candidate_stats["coefficient_of_variation"],
                }
            )

        score = geometric_mean([float(item["speedup"]) for item in report["per_case_speedup"]])
        report["score"] = score
        report["overall_speedup"] = score
        report["status"] = "ok"
        report["error"] = ""
        ranked = sorted(report["per_case_speedup"], key=lambda item: (item["speedup"], item["index"]))
        report["bottlenecks"] = [dict(item) for item in ranked[:profile_top_k]]
        report["diagnostics"]["status"] = "timing_complete"
        _persist_report(output_path, report)

        profile_fn = profiler or (lambda model, inputs: _profile_model(model, inputs, label=root.name))
        unavailable = 0
        for bottleneck in report["bottlenecks"]:
            index = int(bottleneck["index"])
            raw_inputs = input_groups[index]
            profiles: dict[str, Any] = {}
            for implementation, model in (
                ("reference", reference_model),
                ("ascendc", candidate_model),
            ):
                try:
                    model_inputs = _move_to_device(_clone(raw_inputs), device)
                    profile = profile_fn(model, model_inputs)
                    if not isinstance(profile, dict):
                        raise TypeError("profiler must return a JSON object")
                except Exception as error:  # noqa: BLE001 - profiling is supplementary
                    profile = {
                        "status": "unavailable",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                profiles[implementation] = profile
                unavailable += int(profile.get("status") != "ok")
            bottleneck["profiling"] = profiles
            _persist_report(output_path, report)
        report["diagnostics"]["status"] = "complete" if unavailable == 0 else "partial"
        report["diagnostics"]["unavailable_profiles"] = unavailable
        _persist_report(output_path, report)
        return report
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        _persist_report(output_path, report)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="AscendC multi-turn internal NPU benchmark")
    result.add_argument("--task-dir", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    result.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--profile-top-k", type=int, default=DEFAULT_PROFILE_TOP_K)
    return result


def main() -> int:
    args = parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    try:
        report = run_benchmark(
            args.task_dir,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
            profile_top_k=args.profile_top_k,
            output_path=output,
        )
    except Exception:  # noqa: BLE001 - CLI converts every benchmark failure to exit code 1
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "score": report["score"],
                "cases": len(report["per_case_speedup"]),
                "diagnostics": report["diagnostics"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
