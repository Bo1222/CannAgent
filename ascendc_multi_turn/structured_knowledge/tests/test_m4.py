from __future__ import annotations

import unittest
from types import SimpleNamespace

from ascendc_multi_turn.structured_knowledge.api_call_resolver import ApiCallResolver
from ascendc_multi_turn.structured_knowledge.api_constraint_validator import (
    ApiConstraintValidator,
)

SOURCE = """
AscendC::LocalTensor<float> dst;
AscendC::GlobalTensor<float> src;
AscendC::DataCopyExtParams params;
AscendC::DataCopyPadExtParams<float> pad;
uint32_t elementCount = 47;
params.blockLen = elementCount;
AscendC::DataCopyPad(dst, src, params, pad);
"""


def _knowledge():
    return SimpleNamespace(
        symbols={"DataCopyPad": "api_DataCopyPad"},
        facts=[
            {
                "fact_id": "fact_block_len_bytes",
                "subject": "DataCopyPad.blockLen",
                "predicate": "parameter_semantics",
                "value": {"parameter": "blockLen", "unit": "byte"},
                "applicability": {
                    "api": "DataCopyPad",
                    "parameter_structure": "DataCopyExtParams",
                    "overload": "gm_to_local",
                },
                "provenance": {"document_id": "pad", "evidence_text": "blockLen unit is byte"},
            },
            {
                "fact_id": "fact_other_api",
                "subject": "DataCopy.blockLen",
                "predicate": "parameter_semantics",
                "value": {"parameter": "blockLen", "unit": "data_block"},
                "applicability": {"api": "DataCopy", "parameter_structure": "DataCopyParams"},
                "provenance": {"document_id": "copy", "evidence_text": "blockLen unit is block"},
            },
        ],
        project_contracts=[],
    )


class ApiConstraintValidatorTests(unittest.TestCase):
    def test_call_resolver_binds_overload_structure_and_official_fact(self) -> None:
        calls = ApiCallResolver(_knowledge()).resolve_text(
            SOURCE,
            source_path="kernel/copy.cpp",
        )

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call.api, "DataCopyPad")
        self.assertEqual(call.resolved_overload, "gm_to_local")
        self.assertIn("DataCopyExtParams", call.parameter_structures)
        self.assertEqual(call.parameter_semantics["blockLen"]["unit"], "byte")
        self.assertEqual(call.source_fact_ids, ["fact_block_len_bytes"])
        self.assertNotIn("fact_other_api", call.source_fact_ids)

    def test_wrong_parameter_unit_is_rejected_before_compile(self) -> None:
        calls, issues = ApiConstraintValidator(_knowledge()).validate_text(
            SOURCE, source_path="kernel/copy.cpp"
        )

        self.assertEqual(calls[0].status, "resolved")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "parameter_unit_mismatch")
        self.assertEqual(issues[0].fact_id, "fact_block_len_bytes")

    def test_project_contracts_use_generic_required_and_forbidden_rules(self) -> None:
        knowledge = _knowledge()
        knowledge.project_contracts = [
            {
                "contract_id": "contract_stream",
                "subject": "current stream must be forwarded",
                "constraint": r"required_regex:GetCurrentNPUStream",
            },
            {
                "contract_id": "contract_launch",
                "subject": "ACL launch macro is forbidden",
                "constraint": r"forbidden_regex:ACLRT_LAUNCH_KERNEL",
            },
        ]
        _, missing = ApiConstraintValidator(knowledge).validate_text(
            "void host() {}",
            source_path="host.cpp",
        )
        _, forbidden = ApiConstraintValidator(knowledge).validate_text(
            "GetCurrentNPUStream(); ACLRT_LAUNCH_KERNEL(foo);", source_path="host.cpp"
        )
        self.assertEqual(missing[0].code, "project_contract_missing")
        self.assertEqual(forbidden[0].code, "project_contract_forbidden")


if __name__ == "__main__":
    unittest.main()
