from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_PATH = REPO_ROOT / "skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py"
SPEC = importlib.util.spec_from_file_location("validate_ascendc_impl", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load validator from {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _wrapper(forward_body: str, *, extra_imports: str = "") -> str:
    return f"""import _gelu_ext as _ext
{extra_imports}
class ModelNew:
    def forward(self, x):
{forward_body}
"""


class AscendCValidatorTests(unittest.TestCase):
    def test_allows_forbidden_tensor_names_on_confirmed_extension(self) -> None:
        for name in ("gelu", "relu", "matmul", "sum"):
            with self.subTest(name=name):
                result = VALIDATOR.validate(_wrapper(f"        return _ext.{name}(x)"))
                self.assertTrue(result["valid"], result)

    def test_rejects_tensor_method_with_same_name_as_extension_export(self) -> None:
        result = VALIDATOR.validate(
            _wrapper("        y = _ext.gelu(x)\n        return y.gelu()")
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 3)
        self.assertEqual(result["checks"]["no_forbidden_torch_ops"]["violations"][0]["call"], "y.gelu()")

    def test_rejects_functional_fallback_after_extension_call(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        y = _ext.gelu(x)\n        return F.gelu(y)",
                extra_imports="import torch.nn.functional as F",
            )
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 3)
        self.assertEqual(result["checks"]["no_forbidden_torch_ops"]["violations"][0]["call"], "F.gelu")

    def test_cli_marks_unreached_checks_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model_new_ascendc.py"
            source.write_text(
                _wrapper("        y = _ext.gelu(x)\n        return y.gelu()"),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(VALIDATOR_PATH), str(source)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("[FAIL] no_forbidden_torch_ops", completed.stdout)
        self.assertIn("[SKIP] no_scalar_for_loops", completed.stdout)


if __name__ == "__main__":
    unittest.main()
