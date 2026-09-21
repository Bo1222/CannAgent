from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .diagnostics import diagnostics_for_result
from .models import EvalResult, FileBundle

_PATH = re.compile(
    r"(?:model_new_ascendc\.py|"
    r"(?:op_kernel|op_host|op_extension)/[A-Za-z0-9_./-]+\.(?:asc|cpp|cc|cxx|h|hpp))"
)


@dataclass(frozen=True)
class RepairPolicy:
    stage: str
    allowed_paths: tuple[str, ...]
    protected_snippets: dict[str, list[str]] = field(default_factory=dict)
    protected_region_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["protected_snippets"] = {
            path: [snippet.strip() for snippet in snippets]
            for path, snippets in self.protected_snippets.items()
        }
        return payload


def _stage(previous: EvalResult | None, workflow_phase: str) -> str:
    failure = (previous.failure_stage or "") if previous else ""
    if failure == "performance":
        return "performance_tuning"
    if workflow_phase == "optimization":
        return "optimization"
    if failure in {
        "response_format",
        "bundle_validation",
        "ascendc_source_validation",
        "api_constraint_validation",
        "static_validation",
        "ascendc_build",
        "compile",
    }:
        return "compile_repair"
    if failure in {"runtime", "acl_runtime"}:
        return "runtime_repair"
    if failure == "correctness":
        evidence = previous.failure_evidence or {}
        if evidence.get("runtime_code") or evidence.get("device_exception"):
            return "runtime_repair"
        return "correctness_repair"
    return "bootstrap_generation"


def _anchor_lines(source: str, markers: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for line in source.splitlines(keepends=True):
        if any(marker in line for marker in markers):
            result.append(line)
    return result


def _is_editable_logic_path(path: str) -> bool:
    return bool(
        path == "model_new_ascendc.py"
        or (path.startswith("op_kernel/") and path.endswith(("_kernel.asc", "_tiling.h")))
        or (path.startswith("op_host/") and path.endswith(".asc"))
        or (path.startswith("op_extension/") and path.endswith("_torch.cpp"))
    )


def build_repair_policy(
    current: FileBundle | None,
    previous: EvalResult | None,
    active_item: dict | None,
    *,
    workflow_phase: str,
) -> RepairPolicy:
    current = current or FileBundle(files={})
    stage = _stage(previous, workflow_phase)
    plan_text = " ".join(
        str((active_item or {}).get(key, "")) for key in ("hypothesis", "change")
    )
    explicit_targets = {
        str(path).strip()
        for path in (active_item or {}).get("target_files", [])
        if str(path).strip()
    }
    mentioned = explicit_targets or {match.group(0) for match in _PATH.finditer(plan_text)}
    diagnostic_paths = {
        item.source_file
        for item in diagnostics_for_result(previous)
        if item.source_file
    }
    requested = mentioned | diagnostic_paths
    allowed = tuple(
        sorted(
            path
            for path in requested
            if path in current.files and _is_editable_logic_path(path)
        )
    )
    if not allowed:
        allowed = tuple(
            sorted(
                path
                for path in current.files
                if _is_editable_logic_path(path)
            )
        )

    protected: dict[str, list[str]] = {}
    protected_ids: list[str] = []
    if stage == "runtime_repair":
        markers = (
            "#include",
            "TORCH_LIBRARY",
            "PrivateUse1",
            "Meta",
            "Cast(",
            "Muls(",
            "Add(",
        )
        protected_ids.extend(("host_abi", "binding", "arithmetic_dtype_path"))
    elif stage == "correctness_repair":
        markers = (
            "#include",
            "TORCH_LIBRARY",
            "PrivateUse1",
            "Meta",
            "torch.ops",
        )
        protected_ids.extend(("build_and_host_abi", "kernel_launch"))
    elif stage in {"performance_tuning", "optimization"}:
        markers = (
            "TORCH_LIBRARY",
            "PrivateUse1",
            "Meta",
            "torch.ops",
        )
        protected_ids.extend(("public_interface", "host_abi"))
    else:
        markers = ()
    if markers:
        for path, source in current.files.items():
            anchors = _anchor_lines(source, markers)
            if anchors:
                protected[path] = anchors
    return RepairPolicy(stage, allowed, protected, tuple(protected_ids))
