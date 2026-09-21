from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.bundle import parse_file_bundle
from ascendc_multi_turn.project_generator import (
    ProjectGenerationError,
    create_project,
)

MODEL_SOURCE = b"""import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return torch.add(x, y, alpha=alpha)
"""
JSONL_SOURCE = b'{"inputs":[{"name":"x","type":"tensor"}]}\n{"inputs":[]}\n'


def _inputs(root: Path) -> tuple[Path, Path]:
    model = root / "3_Add.py"
    cases = root / "3_Add.json"
    model.write_bytes(MODEL_SOURCE)
    cases.write_bytes(JSONL_SOURCE)
    return model, cases


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class AscendCProjectGeneratorTests(unittest.TestCase):
    def test_exact_tree_dynamic_abi_and_raw_input_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            output = root / "add_alpha"
            spec = create_project(
                op_name="add_alpha", op_file=model, op_json=cases, output=output
            )

            expected_files = {
                "3_Add.json",
                "CMakeLists.txt",
                "model.py",
                "model_new_ascendc.py",
                "op_extension/add_alpha_torch.cpp",
                "op_extension/ops.h",
                "op_extension/register.cpp",
                "op_host/add_alpha.asc",
                "op_host/data_utils.h",
                "op_kernel/add_alpha_kernel.asc",
                "op_kernel/add_alpha_tiling.h",
            }
            actual_files = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            }
            actual_directories = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_dir()
            }
            self.assertEqual(actual_files, expected_files)
            self.assertEqual(actual_directories, {"op_extension", "op_host", "op_kernel", "scripts"})
            self.assertEqual(list((output / "scripts").iterdir()), [])
            self.assertEqual((output / "model.py").read_bytes(), MODEL_SOURCE)
            self.assertEqual((output / "3_Add.json").read_bytes(), JSONL_SOURCE)
            self.assertEqual(spec.abi.schema("add_alpha"), "add_alpha(Tensor x, Tensor y, float alpha=1.0) -> Tensor")

            registration = (output / "op_extension/register.cpp").read_text(encoding="utf-8")
            declarations = (output / "op_extension/ops.h").read_text(encoding="utf-8")
            cmake = (output / "CMakeLists.txt").read_text(encoding="utf-8")
            wrapper = (output / "model_new_ascendc.py").read_text(encoding="utf-8")
            self.assertIn('m.def("add_alpha(Tensor x, Tensor y, float alpha=1.0) -> Tensor")', registration)
            self.assertIn("at::Tensor add_alpha_torch(const at::Tensor& x, const at::Tensor& y, double alpha)", declarations)
            self.assertIn("project(add_alpha LANGUAGES ASC CXX)", cmake)
            self.assertIn(
                "def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:",
                wrapper,
            )
            self.assertNotIn("@OP_", "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file()))

    def test_invalid_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            for name in ("3_add", "../add", "add-alpha", "add/alpha", ""):
                with self.subTest(name=name), self.assertRaises(ProjectGenerationError):
                    create_project(op_name=name, op_file=model, op_json=cases, output=root / "out")

    def test_invalid_json_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            cases.write_text('{"inputs":}\n', encoding="utf-8")
            output = root / "out"
            with self.assertRaisesRegex(ProjectGenerationError, "neither JSON nor JSONL"):
                create_project(op_name="add", op_file=model, op_json=cases, output=output)
            self.assertFalse(output.exists())

    def test_python_is_the_only_abi_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            cases.write_text(
                '{"inputs":[{"name":"unrelated","type":"attr","value":"x"}]}\n',
                encoding="utf-8",
            )
            output = root / "out"
            spec = create_project(op_name="add", op_file=model, op_json=cases, output=output)
            self.assertEqual(spec.abi.schema("add"), "add(Tensor x, Tensor y, float alpha=1.0) -> Tensor")

    def test_standard_json_and_dynamic_optional_tuple_abi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            model.write_text(
                "import torch\nclass Model:\n"
                "    def forward(self, x: torch.Tensor, weight: torch.Tensor = None, *, enabled: bool = True) -> tuple[torch.Tensor, torch.Tensor]:\n"
                "        return x, x\n",
                encoding="utf-8",
            )
            cases.write_text('{"inputs": []}\n', encoding="utf-8")
            output = root / "out"
            spec = create_project(op_name="pair", op_file=model, op_json=cases, output=output)
            self.assertEqual(
                spec.abi.schema("pair"),
                "pair(Tensor x, Tensor? weight=None, *, bool enabled=True) -> (Tensor, Tensor)",
            )
            self.assertIn(
                "std::tuple<at::Tensor, at::Tensor> pair_torch(",
                (output / "op_extension/ops.h").read_text(encoding="utf-8"),
            )

    def test_ambiguous_python_abi_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            model.write_text(
                "class Model:\n    def forward(self, x):\n        return x\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProjectGenerationError, "type annotation"):
                create_project(op_name="add", op_file=model, op_json=cases, output=root / "out")

    def test_nonempty_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            output = root / "out"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ProjectGenerationError, "not empty"):
                create_project(op_name="add", op_file=model, op_json=cases, output=output)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_forbidden_artifacts_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            output = root / "out"
            create_project(op_name="add", op_file=model, op_json=cases, output=output)
            relative = [path.relative_to(output).as_posix() for path in output.rglob("*")]
            forbidden_exact = {"build", "README", "build.sh", "CMakePresets.json", "main.cpp"}
            self.assertFalse(any(path in forbidden_exact for path in relative))
            self.assertFalse(
                any(token in path.lower() for path in relative for token in ("benchmark", "manifest", "aclnn"))
            )

    def test_repeated_generation_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            first, second = root / "first", root / "second"
            create_project(op_name="add_alpha", op_file=model, op_json=cases, output=first)
            create_project(op_name="add_alpha", op_file=model, op_json=cases, output=second)
            self.assertEqual(_file_hashes(first), _file_hashes(second))

    def test_render_failure_preserves_existing_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, cases = _inputs(root)
            output = root / "out"
            output.mkdir()
            with self.assertRaises(ProjectGenerationError):
                create_project(
                    op_name="add",
                    op_file=model,
                    op_json=cases,
                    output=output,
                    template_root=root / "missing-templates",
                )
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])
            self.assertFalse(any(path.name.startswith(".out.tmp-") for path in root.iterdir()))

    def test_llm_bundle_rejects_protected_template_paths(self) -> None:
        payload = '{"files":[{"path":"CMakeLists.txt","content":"changed"}]}'
        with self.assertRaisesRegex(ValueError, "protected"):
            parse_file_bundle(payload, allowed_paths={"model_new_ascendc.py"})


if __name__ == "__main__":
    unittest.main()
