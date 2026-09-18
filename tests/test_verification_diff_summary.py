"""浮点失配定位的回归测试。

`utils/verification_ascendc.py` 以子进程方式被 evaluator 调用，不是可导入的包，
所以这里按路径加载模块。

背景：精度失败时，整数路径一直报告 `first_mismatch(index=..., ref=..., cand=...)`，
浮点路径却只有聚合统计。模型因此只知道“错了”，不知道“错在哪”，
无法区分整体偏移、尾部残留还是某个维度错位。本文件锁定浮点路径的定位输出。
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

_MODULE_PATH = Path(__file__).resolve().parent.parent / "utils" / "verification_ascendc.py"
_SPEC = importlib.util.spec_from_file_location("verification_ascendc_under_test", _MODULE_PATH)
verification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verification)


class FloatDiffLocationTests(unittest.TestCase):
    def test_float_mismatch_reports_index_and_values(self) -> None:
        """浮点失配必须给出可定位的索引与 ref/cand 取值。"""

        reference = torch.arange(128, dtype=torch.float32) + 1.0
        candidate = reference.clone()
        candidate[:84] = candidate[:84] + 4.19

        summary = verification._tensor_diff_summary(reference, candidate)

        self.assertIn("first_mismatch(index=(0,)", summary)
        self.assertIn("ref=1", summary)
        self.assertIn("cand=5.19", summary)
        # 84/128 命中，与 MERE 同判据的相对误差比例。
        self.assertIn("rel_mismatch_ratio=0.6562", summary)

    def test_matching_float_tensors_report_no_mismatch_location(self) -> None:
        reference = torch.arange(32, dtype=torch.float32)
        summary = verification._tensor_diff_summary(reference, reference.clone())

        self.assertIn("passed=True", summary)
        self.assertNotIn("first_mismatch", summary)

    def test_scalar_float_mismatch_uses_empty_index(self) -> None:
        summary = verification._tensor_diff_summary(torch.tensor(3.0), torch.tensor(3.5))

        self.assertIn("first_mismatch(index=(), ref=3, cand=3.5)", summary)

    def test_float_mismatch_is_located_inside_a_multidimensional_tensor(self) -> None:
        reference = torch.zeros(4, 8, dtype=torch.float32)
        candidate = reference.clone()
        candidate[2, 3] = 10.0

        summary = verification._tensor_diff_summary(reference, candidate)

        self.assertIn("first_mismatch(index=(2, 3)", summary)

    def test_infinite_inputs_do_not_crash_the_location_pass(self) -> None:
        """同号 inf 相减会产生 nan，定位过程必须能处理。"""

        reference = torch.tensor([1.0, float("inf"), 2.0, 3.0])
        candidate = torch.tensor([1.0, float("inf"), 2.5, 3.0])

        summary = verification._tensor_diff_summary(reference, candidate)

        self.assertIn("first_mismatch(index=(2,)", summary)

    def test_integer_path_keeps_reporting_first_mismatch(self) -> None:
        """整数路径的既有输出不应被浮点改动影响。"""

        reference = torch.zeros(8, dtype=torch.int32)
        candidate = reference.clone()
        candidate[5] = 3

        summary = verification._tensor_diff_summary(reference, candidate)

        self.assertIn("unequal_elements=1", summary)
        self.assertIn("first_mismatch(index=(5,), ref=0, cand=3)", summary)


if __name__ == "__main__":
    unittest.main()
