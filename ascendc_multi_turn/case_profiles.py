from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .models import EvaluationProfile

_DTYPE_PRIORITY = {"float32": 0, "float16": 1, "bfloat16": 2}


@dataclass(frozen=True)
class _CaseFacts:
    index: int
    dtype: str
    rank: int
    numel: int
    same_shapes: bool
    broadcast: bool
    tail: bool
    default_attrs: bool


def _product(shape: list[int]) -> int:
    return math.prod(int(value) for value in shape) if shape else 1


def _facts(index: int, case: dict[str, Any]) -> _CaseFacts:
    inputs = case.get("inputs", [])
    tensors = [item for item in inputs if item.get("type") == "tensor"]
    shapes = [list(item.get("shape", [])) for item in tensors]
    dtype = str(tensors[0].get("dtype", "unknown")) if tensors else "unknown"
    rank = max((len(shape) for shape in shapes), default=0)
    numel = max((_product(shape) for shape in shapes), default=1)
    same_shapes = bool(shapes) and all(shape == shapes[0] for shape in shapes[1:])
    broadcast = len(shapes) > 1 and not same_shapes
    tail = numel % 32 != 0
    attrs = [item for item in inputs if item.get("type") == "attr"]
    default_attrs = all(item.get("value") in (None, 0, 1, 1.0, False) for item in attrs)
    return _CaseFacts(index, dtype, rank, numel, same_shapes, broadcast, tail, default_attrs)


def parse_cases(cases_text: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in cases_text.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("each benchmark case must be a JSON object")
        cases.append(value)
    return cases


def build_case_profiles(cases_text: str) -> list[EvaluationProfile]:
    if cases_text.strip() == "(no JSON case file)":
        return [EvaluationProfile("full", tuple()), EvaluationProfile("benchmark", tuple(), True)]
    cases = parse_cases(cases_text)
    if not cases:
        return [EvaluationProfile("full", tuple()), EvaluationProfile("benchmark", tuple(), True)]
    facts = [_facts(index, case) for index, case in enumerate(cases)]
    canonical = min(
        facts,
        key=lambda item: (
            _DTYPE_PRIORITY.get(item.dtype, 99),
            not item.same_shapes,
            item.broadcast,
            item.rank,
            item.numel < 32,
            item.numel > 1_048_576,
            abs(item.numel - 128),
            not item.default_attrs,
            item.index,
        ),
    )
    smoke = (canonical.index,)

    shape_candidates = [item for item in facts if item.dtype == canonical.dtype]
    shape_indices = set(smoke)
    signatures: set[tuple[int, bool, bool]] = set()
    for item in sorted(shape_candidates, key=lambda value: (value.numel, value.index)):
        signature = (item.rank, item.broadcast, item.tail)
        if signature not in signatures:
            signatures.add(signature)
            shape_indices.add(item.index)

    dtype_indices = set(shape_indices)
    for dtype in sorted({item.dtype for item in facts}, key=lambda name: _DTYPE_PRIORITY.get(name, 99)):
        representative = min(
            (item for item in facts if item.dtype == dtype),
            key=lambda item: (not item.same_shapes, item.broadcast, item.rank, item.numel, item.index),
        )
        dtype_indices.add(representative.index)

    all_indices = tuple(range(len(cases)))
    profiles = [
        EvaluationProfile("smoke", smoke, features=("smoke",)),
        EvaluationProfile(
            "shape",
            tuple(sorted(shape_indices)),
            features=("broadcast", "tail", "shape"),
        ),
        EvaluationProfile("dtype", tuple(sorted(dtype_indices)), features=("dtype",)),
        EvaluationProfile("full", all_indices, features=("full",)),
        EvaluationProfile("benchmark", all_indices, True),
    ]
    deduplicated: list[EvaluationProfile] = []
    for profile in profiles:
        if (
            deduplicated
            and profile.case_indices == deduplicated[-1].case_indices
            and not profile.run_performance
            and profile.name != "full"
        ):
            continue
        deduplicated.append(profile)
    return deduplicated
