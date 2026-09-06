from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.source_validation import validate_source_tree


class AscendCSourceValidationTests(unittest.TestCase):
    def _issues(self, source: str):
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary)
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir()
            (kernel_dir / "kernel.cpp").write_text(source, encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
