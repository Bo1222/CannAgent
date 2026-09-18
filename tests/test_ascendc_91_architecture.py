from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.__main__ import parser
from ascendc_multi_turn.bundle import (
    FileBundle,
    capture_bundle,
    parse_file_bundle,
    validate_initial_bundle,
)
from ascendc_multi_turn.devkit import (
    ASC_DEVKIT_COMMIT,
    ASC_DEVKIT_REF,
    ASC_DEVKIT_REPOSITORY,
    ASC_DEVKIT_VERSION,
    initialize_devkit,
    inspect_devkit,
    resolve_devkit,
)
from ascendc_multi_turn.devkit_retrieval import DevkitRetriever
from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import RunConfig
from ascendc_multi_turn.runner import MultiTurnRunner
from ascendc_multi_turn.skill_adapter import CANNBOT_VENDOR_MANIFEST_PATH, SkillAdapter
from ascendc_multi_turn.source_validation import validate_source_tree


def _devkit(root: Path) -> Path:
    for directory in ("docs/api/math", "examples", "include", "impl"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "version.cmake").write_text(
        'set(ASCEND_CANN_PACKAGE_VERSION "9.1.0")\n', encoding="utf-8"
    )
    (root / ".cannagent-devkit.json").write_text(
        json.dumps({"commit": ASC_DEVKIT_COMMIT}), encoding="utf-8"
    )
    return root


def _project_files() -> dict[str, str]:
    return {
        "model_new_ascendc.py": (
            "import torch\n"
            "torch.ops.load_library('build/libadd.so')\n"
            "class ModelNew:\n"
            "    def forward(self, x): return torch.ops.cannagent.add(x)\n"
        ),
        "op_kernel/add_tiling.h": "#pragma once\n",
        "op_kernel/add_kernel.asc": "// AscendC kernel\n",
        "op_host/add.asc": "// host launch\n",
        "op_host/data_utils.h": "#pragma once\n",
        "op_extension/add_torch.cpp": "// torch op\n",
        "op_extension/register.cpp": (
            "TORCH_LIBRARY(cannagent, m) {}\n"
            "TORCH_LIBRARY_IMPL(cannagent, PrivateUse1, m) {}\n"
            "TORCH_LIBRARY_IMPL(cannagent, Meta, m) {}\n"
        ),
        "op_extension/ops.h": "#pragma once\n",
        "scripts/golden.py": "# golden\n",
        "scripts/test_torch.py": "# test\n",
        "CMakeLists.txt": (
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(add LANGUAGES ASC CXX)\n"
            "find_package(ASC REQUIRED)\n"
            "add_library(add SHARED op_kernel/add_kernel.asc op_host/add.asc "
            "op_extension/add_torch.cpp op_extension/register.cpp)\n"
        ),
    }


class DevkitTests(unittest.TestCase):
    def test_supply_chain_is_pinned_to_gitcode_91_commit(self) -> None:
        self.assertEqual(ASC_DEVKIT_REPOSITORY, "https://gitcode.com/cann/asc-devkit.git")
        self.assertEqual(ASC_DEVKIT_REF, "9.1.0")
        self.assertEqual(ASC_DEVKIT_COMMIT, "c785b5f76f23c9dc0ddb553174463d0497e59288")

    def test_managed_clone_uses_pinned_gitcode_branch_and_commit(self) -> None:
        commands: list[list[str]] = []

        def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if command[1] == "clone":
                _devkit(Path(command[-1]))
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as temporary:
            status = initialize_devkit(cache_root=Path(temporary), command_runner=run)

        self.assertTrue(status.healthy, status.errors)
        self.assertEqual(
            commands[0][:-1],
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--single-branch",
                "--branch",
                ASC_DEVKIT_REF,
                ASC_DEVKIT_REPOSITORY,
            ],
        )
        self.assertEqual(commands[1][-3:], ["checkout", "--detach", ASC_DEVKIT_COMMIT])

    def test_health_check_requires_exact_91_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _devkit(Path(temporary))
            status = inspect_devkit(root)
            self.assertTrue(status.healthy, status.errors)
            self.assertEqual(status.version, ASC_DEVKIT_VERSION)
            self.assertEqual(resolve_devkit(root), root.resolve())

    def test_retrieval_priority_metadata_and_image_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _devkit(Path(temporary))
            (root / "docs/api/math/Add.md").write_text("# Add\nAscendC Add API", encoding="utf-8")
            (root / "examples/add.asc").write_text("Add(dst, x, y);", encoding="utf-8")
            (root / "docs/api/math/Add.png").write_bytes(b"image")
            bundle = DevkitRetriever(root).retrieve(operator="add", symbols=["Add"], query_text="Add")
            self.assertEqual(bundle.evidence[0].source_kind, "api_doc")
            self.assertTrue(all(not item.source_path.endswith(".png") for item in bundle.evidence))
            self.assertTrue(all(item.commit == ASC_DEVKIT_COMMIT for item in bundle.evidence))


class ArchitectureContractTests(unittest.TestCase):
    def test_legacy_knowledge_flags_are_absent(self) -> None:
        help_text = parser().format_help()
        for option in ("--knowledge-mode", "--knowledge-source", "--knowledge-store", "--skill-adapter"):
            self.assertNotIn(option, help_text)
        self.assertIn("--asc-devkit-dir", help_text)

    def test_bundle_requires_cannbot_project_and_rejects_legacy_layout(self) -> None:
        validate_initial_bundle(FileBundle(files=_project_files()))
        with self.assertRaisesRegex(ValueError, "CANNBot AscendC project"):
            parse_file_bundle(json.dumps({"files": [{"path": "kernel/pybind11.cpp", "content": "x"}]}))

    def test_source_contract_requires_dispatcher_and_rejects_do_abi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in _project_files().items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            self.assertEqual(validate_source_tree(root), [])
            (root / "op_host/add.asc").write_text('extern "C" void add_do() {}\n', encoding="utf-8")
            self.assertIn("legacy_do_abi", {item.code for item in validate_source_tree(root)})

    def test_pinned_cannbot_subset_is_hash_validated(self) -> None:
        payload = json.loads(CANNBOT_VENDOR_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["upstream_commit"], "a08c49706e35a400d7c77e0875bc7c72a3a79012")
        self.assertEqual(len(payload["skills"]), 10)
        self.assertTrue(SkillAdapter().registry)

    def test_mock_workflow_emits_new_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            op = root / "add.py"
            op.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "output"
            config = RunConfig(
                op_file=str(op), output_dir=str(output), mock=True,
                max_bootstrap_rounds=1, max_rounds=1, max_total_rounds=1,
            )
            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()
            self.assertTrue(summary["success"])
            bundle = capture_bundle(output)
            self.assertIn("op_extension/register.cpp", bundle.files)
            self.assertNotIn("kernel/pybind11.cpp", bundle.files)


if __name__ == "__main__":
    unittest.main()
