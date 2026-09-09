from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.source_validation import validate_source_tree


class AscendCSourceValidationTests(unittest.TestCase):
    def _issues(self, source: str, pybind: str = ""):
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary)
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir()
            (kernel_dir / "kernel.cpp").write_text(source, encoding="utf-8")
            if pybind:
                (kernel_dir / "pybind11.cpp").write_text(pybind, encoding="utf-8")
            return validate_source_tree(task_dir)

    def test_rejects_duplicate_file_constants_and_tpipe_queue_methods(self) -> None:
        issues = self._issues(
            """namespace Demo {
constexpr float TANH_C1 = 1.0f;
constexpr float TANH_C1 = 2.0f;
class Kernel {
  AscendC::TPipe pipe_;
  void Run() { pipe_.EnQue(1); }
};
}
"""
        )

        self.assertEqual(
            {issue.code for issue in issues},
            {"duplicate_file_symbol", "invalid_tpipe_owner"},
        )

    def test_allows_tque_methods_overloads_and_nested_locals(self) -> None:
        issues = self._issues(
            """namespace Demo {
constexpr float TANH_C1 = 1.0f;
void Compute(int x) {}
void Compute(float x) {}
class Kernel {
  AscendC::TPipe pipe_;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> queue_;
  void Run() { const int value = 1; queue_.EnQue(value); }
};
}
"""
        )

        self.assertEqual(issues, [])

    def test_accepts_repository_host_wrapper_contract(self) -> None:
        issues = self._issues(
            '''extern "C" __global__ __aicore__ void gelu_custom(GM_ADDR x, GM_ADDR y) {}
extern "C" void gelu_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y)
{
    gelu_custom<<<blockDim, nullptr, stream>>>(x, y);
}
''',
            '''#include <pybind11/pybind11.h>
extern "C" void gelu_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y);
void run(void *stream, uint8_t *x, uint8_t *y) { gelu_do(1, stream, x, y); }
PYBIND11_MODULE(_gelu, m) {}
''',
        )

        self.assertEqual(issues, [])

    def test_rejects_direct_acl_launch_and_missing_header(self) -> None:
        issues = self._issues(
            "// kernel\n",
            '''#include <acl/acl_rt_launch.h>
#include "/invented/cann/header.h"
void run() { ACLRT_LAUNCH_KERNEL(gelu, 1, nullptr); }
PYBIND11_MODULE(_gelu, m) {}
''',
        )

        self.assertTrue(
            {"unsupported_launch_header", "absolute_include", "unsupported_launch_macro"}
            .issubset({issue.code for issue in issues})
        )

    def test_rejects_missing_or_mismatched_wrapper_definition(self) -> None:
        issues = self._issues(
            '''extern "C" void gelu_do(uint32_t blockDim, void *stream, uint8_t *x)
{
    // no launch
}
''',
            '''extern "C" void gelu_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y);
void run(void *s, uint8_t *x, uint8_t *y) { gelu_do(1, s, x, y); }
PYBIND11_MODULE(_gelu, m) {}
''',
        )

        self.assertIn("host_wrapper_signature_mismatch", {issue.code for issue in issues})
        self.assertIn("missing_kernel_launch", {issue.code for issue in issues})

    def test_repository_examples_satisfy_the_host_launch_contract(self) -> None:
        repository = Path(__file__).resolve().parent.parent
        checked = 0
        for task_dir in sorted((repository / "archive_tasks").iterdir()):
            if not (task_dir / "kernel" / "pybind11.cpp").is_file():
                continue
            checked += 1
            self.assertEqual(validate_source_tree(task_dir), [], task_dir.name)
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
