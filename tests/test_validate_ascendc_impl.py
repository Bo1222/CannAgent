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


def _wrapper(
    forward_body: str,
    *,
    extra_imports: str = "",
    module_name: str = "_gelu_ext",
    init_body: str = "        pass",
    helper_body: str = "",
    forward_args: str = "x",
) -> str:
    helper = f"\n{helper_body}\n" if helper_body else ""
    return f"""import {module_name} as _ext
{extra_imports}
class ModelNew:
    def __init__(self):
{init_body}
{helper}
    def forward(self, {forward_args}):
{forward_body}
"""


class AscendCValidatorTests(unittest.TestCase):
    def test_allows_forbidden_tensor_names_on_confirmed_extension(self) -> None:
        for name in ("gelu", "relu", "matmul", "sum"):
            with self.subTest(name=name):
                result = VALIDATOR.validate(
                    _wrapper(f"        return _ext.{name}(x)"),
                    expected_extension_modules={"_gelu_ext"},
                )
                self.assertTrue(result["valid"], result)

    def test_rejects_tensor_method_with_same_name_as_extension_export(self) -> None:
        result = VALIDATOR.validate(
            _wrapper("        y = _ext.gelu(x)\n        return y.gelu()"),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 3)
        self.assertEqual(result["checks"]["no_forbidden_torch_ops"]["violations"][0]["call"], "y.gelu()")

    def test_rejects_functional_fallback_after_extension_call(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        y = _ext.gelu(x)\n        return F.gelu(y)",
                extra_imports="import torch.nn.functional as F",
            ),
            expected_extension_modules={"_gelu_ext"},
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
            kernel = source.parent / "kernel"
            kernel.mkdir()
            (kernel / "pybind11.cpp").write_text(
                "PYBIND11_MODULE(_gelu_ext, m) {}\n",
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

    def test_accepts_arbitrary_module_name_confirmed_by_pybind(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        return _ext.gelu(x)",
                module_name="_ascendc_gelu_impl",
            ),
            expected_extension_modules={"_ascendc_gelu_impl"},
        )
        self.assertTrue(result["valid"], result)

    def test_rejects_import_that_does_not_match_pybind_module(self) -> None:
        result = VALIDATOR.validate(
            _wrapper("        return _ext.gelu(x)"),
            expected_extension_modules={"_different_module"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 1)
        self.assertIn("_different_module", result["checks"]["ascendc_ext_imported"]["error"])

    def test_resolves_torch_alias_and_direct_function_import(self) -> None:
        cases = [
            ("import torch as backend", "        y = _ext.gelu(x)\n        return backend.sum(y)"),
            (
                "from torch.nn.functional import gelu as torch_gelu",
                "        y = _ext.gelu(x)\n        return torch_gelu(y)",
            ),
        ]
        for extra_imports, body in cases:
            with self.subTest(extra_imports=extra_imports):
                result = VALIDATOR.validate(
                    _wrapper(body, extra_imports=extra_imports),
                    expected_extension_modules={"_gelu_ext"},
                )
                self.assertFalse(result["valid"])
                self.assertEqual(result["regression_type"], 3)

    def test_rejects_tensor_arithmetic_operator(self) -> None:
        result = VALIDATOR.validate(
            _wrapper("        y = _ext.gelu(x)\n        return y + x"),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["checks"]["no_forbidden_torch_ops"]["violations"][0]["source"], "Tensor operator")

    def test_allows_non_torch_module_with_torch_like_method_name(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        y = _ext.gelu(x)\n        return custom.gelu(y)",
                extra_imports="import custom_backend as custom",
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertTrue(result["valid"], result)

    def test_rejects_torch_fallback_hidden_in_local_helper(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        return self._helper(x)",
                extra_imports="import torch.nn.functional as F",
                helper_body="    def _helper(self, value):\n        _ext.gelu(value)\n        return F.gelu(value)",
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 3)

    def test_rejects_torch_module_attribute_but_allows_layout_methods(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        y = _ext.gelu(x.reshape(-1).contiguous())\n        return self.layer(y)",
                extra_imports="import torch.nn as nn",
                init_body="        self.layer = nn.Linear(4, 4)",
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 3)
        self.assertEqual(result["checks"]["no_forbidden_torch_ops"]["violations"][0]["source"], "torch.nn.Module")

    def test_allows_tensor_metadata_layout_and_buffer_allocation(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        y = x.reshape(-1).contiguous().new_empty(x.shape)\n"
                "        return _ext.gelu(x, y)"
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertTrue(result["valid"], result)

    def test_rejects_per_element_extension_launch_loop(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        for i in range(x.shape[0]):\n"
                "            _ext.gelu(x[i])\n"
                "        return x"
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["regression_type"], 4)

    def test_allows_shape_metadata_loop(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        dims = []\n"
                "        for i in range(x.ndim):\n"
                "            dims.append(x.shape[i])\n"
                "        return _ext.gelu(x)"
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertTrue(result["valid"], result)

    def test_treats_literal_default_arguments_as_scalars(self) -> None:
        result = VALIDATOR.validate(
            _wrapper(
                "        if approximate == 'tanh' and dim >= 0:\n"
                "            return _ext.gelu(x, approximate, dim)\n"
                "        return _ext.gelu(x, approximate, dim)",
                forward_args="x, approximate='none', dim=0",
            ),
            expected_extension_modules={"_gelu_ext"},
        )
        self.assertTrue(result["valid"], result)


if __name__ == "__main__":
    unittest.main()
