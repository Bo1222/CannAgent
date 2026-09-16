from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ascendc_multi_turn.diagnostics import observe_evaluation_stages
from ascendc_multi_turn.logging import TrajectoryLogger
from ascendc_multi_turn.models import EvalResult, LLMResponse
from ascendc_multi_turn.runner import MultiTurnRunner


class KnowledgeObservabilityTests(unittest.TestCase):
    def test_round_records_input_and_result_failure_routes_separately(self) -> None:
        selection = SimpleNamespace(
            to_dict=lambda: {
                "task_facts": {
                    "input_route": {
                        "primary": "host_integration_debug",
                        "reason": "host_kernel_launch_abi_evidence",
                        "ownership": "Host/Kernel boundary",
                        "debug_category": "launch_abi",
                    }
                }
            }
        )
        result = EvalResult(
            compiled=True,
            correctness=False,
            error="profile=shape mismatch [128,128] + [128,1]",
            failure_stage="correctness",
            verify_output="Comparison case[17]: passed=False",
        )
        record = MultiTurnRunner._round_record(
            attempt_id=10,
            evaluation_round=10,
            budget_phase="bootstrap",
            budget_round=10,
            decision="FAIL",
            result=result,
            selection=selection,
            response=LLMResponse(content="", model="fixture"),
            generation_attempts=1,
            candidate_path="round_10/candidate.json",
            plan_item=None,
            fingerprint=None,
            frontier={},
            incident_id="fixture",
            confirmed_experience_id=None,
            stage_observation={},
            stage_timings=[],
        )

        self.assertEqual(record["input_route"]["primary"], "host_integration_debug")
        self.assertEqual(record["result_failure_route"]["primary"], "kernel_design")
        self.assertEqual(record["result_failure_route"]["ownership"], "Kernel")

    def test_binding_failure_is_distinct_from_execution_and_correctness(self) -> None:
        stages = observe_evaluation_stages(
            EvalResult(
                compiled=True,
                correctness=False,
                error="Input must be a CUDA tensor",
                failure_stage="correctness",
                verify_output="RuntimeError: Input must be a CUDA tensor",
            )
        )

        self.assertEqual(stages["B_compile"]["status"], "pass")
        self.assertEqual(stages["C_load"]["status"], "fail")
        self.assertEqual(stages["D_execute"]["status"], "not_reached")
        self.assertEqual(stages["E_correct"]["status"], "not_reached")

    def test_optimization_correctness_loss_is_recorded_as_failure(self) -> None:
        stages = observe_evaluation_stages(
            EvalResult(
                compiled=True,
                correctness=False,
                error="comparison mismatch",
                failure_stage="correctness",
                verify_output="Comparison\ncase[0]: passed=False",
            ),
            workflow_phase="optimization",
        )

        self.assertEqual(stages["G_optimization"]["status"], "pass")
        self.assertEqual(stages["H_optimized_correct"]["status"], "fail")

    def test_metrics_are_stage_normalized_and_censor_unreached_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            logger = TrajectoryLogger(Path(temporary), {"test": True})
            logger.save_call(
                1,
                {
                    "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
                    "latency_seconds": 2.0,
                },
                evaluation_round=1,
                prompt_metadata={
                    "knowledge_reference": {"chars": 400, "estimated_tokens": 100},
                    "source_code": {"chars": 800, "estimated_tokens": 200},
                    "compiler_evaluator_evidence": {"chars": 40, "estimated_tokens": 10},
                },
            )
            logger.save_round(
                {
                    "evaluation_round": 1,
                    "stage_observation": {
                        "A_source_valid": {"status": "pass"},
                        "B_compile": {"status": "fail"},
                    },
                    "stage_timings": [
                        {"label": "round_01 · AscendC build", "elapsed_seconds": 3.0}
                    ],
                },
                best_round=None,
            )
            metrics = logger.normalized_metrics()

        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["tokens_to_first"]["source_valid"], 100)
        self.assertIsNone(metrics["tokens_to_first"]["compile"])
        self.assertEqual(metrics["prompt_component_totals"]["knowledge_reference"]["chars"], 400)
        self.assertEqual(metrics["evaluation_latency_by_stage_seconds"]["compile"], 3.0)
        self.assertEqual(metrics["transition_rates"]["source_valid_to_compile"]["rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
