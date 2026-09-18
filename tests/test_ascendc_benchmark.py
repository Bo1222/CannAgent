from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from ascendc_multi_turn import benchmark
from ascendc_multi_turn.benchmark import (
    _profile_step_count,
    geometric_mean,
    latency_statistics,
    parse_operator_details,
)
from ascendc_multi_turn.diagnostics import compact_evaluation
from ascendc_multi_turn.evaluator import benchmark_command, validate_performance_report
from ascendc_multi_turn.models import EvalResult


def _report() -> dict:
    cases = [
        {
            "index": 0,
            "reference_ms": 2.0,
            "ascendc_ms": 1.0,
            "speedup": 2.0,
            "reference_cv": 0.01,
            "ascendc_cv": 0.02,
        },
        {
            "index": 1,
            "reference_ms": 1.0,
            "ascendc_ms": 2.0,
            "speedup": 0.5,
            "reference_cv": 0.03,
            "ascendc_cv": 0.04,
        },
    ]
    score = geometric_mean([item["speedup"] for item in cases])
    return {
        "schema_version": 2,
        "status": "ok",
        "score": score,
        "overall_speedup": score,
        "measurement": {"method": "torch.npu.Event", "repeats": 50},
        "per_case_speedup": cases,
        "bottlenecks": [
            {
                **cases[1],
                "profiling": {
                    "ascendc": {
                        "status": "ok",
                        "operators": [
                            {"name": f"op_{index}", "mean_device_self_us": index}
                            for index in range(8)
                        ],
                    }
                },
            }
        ],
        "diagnostics": {"status": "complete"},
    }


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_latency_statistics_and_geometric_score(self) -> None:
        stats = latency_statistics([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["sample_count"], 4)
        self.assertEqual(stats["median_ms"], 2.5)
        self.assertAlmostEqual(stats["p90_ms"], 3.7)
        self.assertAlmostEqual(geometric_mean([2.0, 0.5]), 1.0)

    def test_statistics_reject_non_positive_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite positive"):
            latency_statistics([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "finite positive"):
            geometric_mean([1.0, math.inf])

    def test_profiler_schedule_counts_skip_warmup_and_active_steps(self) -> None:
        self.assertEqual(_profile_step_count(skip_first=1, warmup=2, active=5), 8)

    def test_operator_csv_uses_all_rows_and_standard_library_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "operator_details.csv"
            path.write_text(
                "Name,Device Self Duration(us),Count\n"
                "kernel_a,20,5\n"
                "kernel_a,10,5\n"
                "kernel_b,50,10\n",
                encoding="utf-8",
            )
            operators = parse_operator_details(path, active_count=5)
            self.assertEqual([item["name"] for item in operators], ["kernel_b", "kernel_a"])
            self.assertEqual(operators[0]["mean_device_self_us"], 10.0)
            self.assertEqual(operators[1]["observed_count"], 10)

    def test_internal_runner_builds_versioned_report_without_skill_script(self) -> None:
        class Reference(nn.Module):
            def forward(self, value):
                return value

        class Candidate(nn.Module):
            def forward(self, value):
                return value

        reference_module = SimpleNamespace(
            __file__="model.py",
            Model=Reference,
            get_input_groups=lambda: [[1.0]],
            get_init_inputs=list,
        )
        candidate_module = SimpleNamespace(__file__="model_new_ascendc.py", ModelNew=Candidate)
        reference_stats = latency_statistics([2.0, 2.0])
        candidate_stats = latency_statistics([1.0, 1.0])

        def measure(model, inputs, *, warmup, repeats):
            del inputs, warmup, repeats
            return (reference_stats, 1.0) if isinstance(model, Reference) else (candidate_stats, 1.0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("# reference\n", encoding="utf-8")
            (root / "model_new_ascendc.py").write_text("# candidate\n", encoding="utf-8")
            (root / "build").mkdir()
            output = root / "performance.json"
            fake_npu = SimpleNamespace(manual_seed=lambda seed: None)
            with (
                mock.patch.object(benchmark, "_require_npu", return_value=torch.device("cpu")),
                mock.patch.object(
                    benchmark,
                    "_load_module",
                    side_effect=[reference_module, candidate_module],
                ),
                mock.patch.object(benchmark, "_measure_with_events", side_effect=measure),
                mock.patch.object(benchmark.torch, "npu", fake_npu),
                mock.patch.object(benchmark.torch, "manual_seed", return_value=None),
            ):
                report = benchmark.run_benchmark(
                    root,
                    warmup=1,
                    repeats=2,
                    profile_top_k=1,
                    output_path=output,
                    profiler=lambda model, inputs: (_ for _ in ()).throw(
                        RuntimeError("profiler unavailable")
                    ),
                )
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["score"], 2.0)
            self.assertEqual(report["diagnostics"]["status"], "partial")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["score"], 2.0)


class BenchmarkBoundaryTests(unittest.TestCase):
    def test_evaluator_uses_package_module_not_repository_skills(self) -> None:
        command = benchmark_command(
            python="python",
            task_dir=Path("/tmp/operator"),
            output_path=Path("/tmp/performance.json"),
        )
        self.assertEqual(command[1:3], ["-m", "ascendc_multi_turn.benchmark"])
        self.assertNotIn("skills", " ".join(command))

    def test_all_multiturn_python_sources_are_free_of_root_skills_paths(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ascendc_multi_turn"
        forbidden = ('REPO_ROOT / "skills"', "REPO_ROOT / 'skills'", "skills/ascendc/")
        violations = []
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                violations.append(path.name)
        self.assertEqual(violations, [])

    def test_report_validation_requires_exact_full_case_set(self) -> None:
        report = _report()
        self.assertAlmostEqual(
            validate_performance_report(report, expected_case_indices={0, 1}),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "differs from full correctness"):
            validate_performance_report(report, expected_case_indices={0, 1, 2})

    def test_report_validation_rejects_non_event_or_inconsistent_score(self) -> None:
        report = _report()
        report["measurement"]["method"] = "time.perf_counter"
        with self.assertRaisesRegex(ValueError, "torch.npu.Event"):
            validate_performance_report(report)
        report = _report()
        report["score"] = 2.0
        with self.assertRaisesRegex(ValueError, "geometric mean"):
            validate_performance_report(report)

    def test_compact_evaluation_exposes_bounded_optimization_evidence(self) -> None:
        report = _report()
        compact = compact_evaluation(EvalResult(True, True, score=1.0, performance=report))
        performance = compact["performance"]
        self.assertEqual(performance["measurement"]["method"], "torch.npu.Event")
        self.assertEqual(len(performance["bottlenecks"]), 1)
        self.assertEqual(
            len(performance["bottlenecks"][0]["profiling"]["ascendc"]["operators"]),
            5,
        )

    def test_report_is_json_serializable(self) -> None:
        json.dumps(_report())


if __name__ == "__main__":
    unittest.main()
