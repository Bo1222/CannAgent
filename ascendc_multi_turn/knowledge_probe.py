from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentFingerprint:
    fingerprint_id: str
    cann_version: str
    toolkit_root: str | None
    torch_version: str | None
    torch_npu_version: str | None
    compiler: str | None
    compiler_version: str | None
    ascend_compiler: str | None
    ascend_compiler_version: str | None
    include_roots: tuple[str, ...]
    soc_version: str
    build_contract_sha256: str | None
    probe_contract_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    source: str
    command: tuple[str, ...]
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    success: bool
    return_code: int
    command: tuple[str, ...]
    source_sha256: str
    output_sha256: str
    fingerprint_id: str
    fact_ids: tuple[str, ...]
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler_identity() -> tuple[str | None, str | None]:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        return None, None
    try:
        completed = subprocess.run(
            [compiler, "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=5, check=False,
        )
        version = (completed.stdout or "").splitlines()[0].strip() or None
    except (OSError, subprocess.TimeoutExpired):
        version = None
    return str(Path(compiler).resolve()), version


def _ascend_compiler_identity(toolkit_root: Path | None) -> tuple[str | None, str | None]:
    if toolkit_root is None:
        return None, None
    candidates = (
        toolkit_root / "aarch64-linux/ccec_compiler/bin/bisheng",
        toolkit_root / "compiler/ccec_compiler/bin/bisheng",
    )
    compiler = next((path for path in candidates if path.is_file()), None)
    if compiler is None:
        return None, None
    try:
        completed = subprocess.run(
            [str(compiler), "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=5, check=False,
        )
        version = (completed.stdout or "").splitlines()[0].strip() or None
    except (OSError, subprocess.TimeoutExpired):
        version = None
    return str(compiler.resolve()), version


def torch_npu_include_roots() -> list[Path]:
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or not spec.submodule_search_locations:
        return []
    roots: list[Path] = []
    for location in spec.submodule_search_locations:
        include = Path(location) / "include"
        if include.is_dir():
            roots.append(include.resolve())
    return roots


def collect_environment_fingerprint(
    *, runtime_version: str,
    soc_version: str,
    toolkit_root: Path | None,
    include_roots: list[Path],
    project_root: Path | None = None,
) -> EnvironmentFingerprint:
    compiler, compiler_version = _compiler_identity()
    ascend_compiler, ascend_compiler_version = _ascend_compiler_identity(toolkit_root)
    build_script = project_root / "utils/build_ascendc.py" if project_root else None
    payload = {
        "cann_version": runtime_version,
        "toolkit_root": str(toolkit_root.resolve()) if toolkit_root else None,
        "torch_version": _distribution_version("torch"),
        "torch_npu_version": _distribution_version("torch-npu", "torch_npu"),
        "compiler": compiler,
        "compiler_version": compiler_version,
        "ascend_compiler": ascend_compiler,
        "ascend_compiler_version": ascend_compiler_version,
        "include_roots": sorted(str(path.resolve()) for path in include_roots),
        "soc_version": soc_version,
        "build_contract_sha256": _sha256_file(build_script),
        "probe_contract_sha256": _sha256_file(Path(__file__)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return EnvironmentFingerprint(
        fingerprint_id=hashlib.sha256(encoded).hexdigest()[:20],
        include_roots=tuple(payload["include_roots"]),
        **{key: value for key, value in payload.items() if key != "include_roots"},
    )


def run_compile_probe(
    spec: ProbeSpec,
    *,
    output_dir: Path,
    fingerprint: EnvironmentFingerprint,
    timeout: int = 120,
) -> ProbeResult:
    """Run one explicitly supplied probe and persist enough evidence to audit it."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"cannagent-{spec.probe_id}-") as temporary:
        work = Path(temporary)
        source_path = work / "probe.cpp"
        output_path = work / "probe.out"
        source_path.write_text(spec.source, encoding="utf-8")
        command = tuple(
            item.replace("{source}", str(source_path)).replace("{output}", str(output_path))
            for item in spec.command
        )
        try:
            completed = subprocess.run(
                list(command), cwd=work, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            return_code = completed.returncode
            output = completed.stdout or ""
        except subprocess.TimeoutExpired as error:
            return_code = 124
            raw = error.stdout or ""
            output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            output += f"\nTimed out after {timeout}s"
        except OSError as error:
            return_code = 127
            output = f"Failed to start probe command: {error}\n"
    log_path = output_dir / f"{spec.probe_id}.log"
    log_path.write_text(output, encoding="utf-8")
    result = ProbeResult(
        probe_id=spec.probe_id,
        success=return_code == 0,
        return_code=return_code,
        command=command,
        source_sha256=hashlib.sha256(spec.source.encode("utf-8")).hexdigest(),
        output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        fingerprint_id=fingerprint.fingerprint_id,
        fact_ids=spec.fact_ids if return_code == 0 else (),
        log_path=str(log_path.resolve()),
    )
    (output_dir / f"{spec.probe_id}.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_verified_probe_facts(
    manifest_path: Path | None,
    *,
    fingerprint_id: str,
) -> dict[str, dict[str, Any]]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("fingerprint_id") != fingerprint_id:
        return {}
    verified: dict[str, dict[str, Any]] = {}
    for probe in payload.get("probes", []):
        if not isinstance(probe, dict) or not probe.get("success"):
            continue
        for fact_id in probe.get("fact_ids", []):
            verified[str(fact_id)] = {
                "probe_id": str(probe.get("probe_id", "")),
                "probe_status": "passed",
                "fingerprint_id": fingerprint_id,
            }
    return verified


def write_probe_manifest(
    path: Path,
    *,
    fingerprint: EnvironmentFingerprint,
    results: list[ProbeResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint_id": fingerprint.fingerprint_id,
                "environment_fingerprint": fingerprint.to_dict(),
                "probes": [item.to_dict() for item in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def torch_include_roots_without_import() -> tuple[list[Path], str]:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return [], "1"
    package = Path(next(iter(spec.submodule_search_locations))).resolve()
    include = package / "include"
    roots = [include, include / "torch/csrc/api/include"]
    abi = "1"
    config = package / "share/cmake/Torch/TorchConfig.cmake"
    if config.is_file():
        match = __import__("re").search(
            r"-D_GLIBCXX_USE_CXX11_ABI=(\d)",
            config.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            abi = match.group(1)
    return [path for path in roots if path.is_dir()], abi


def _host_probe_spec(*, include_roots: list[Path]) -> ProbeSpec:
    compiler = shutil.which(os.environ.get("CXX", "c++")) or "c++"
    torch_roots, abi = torch_include_roots_without_import()
    python_include = Path(sysconfig.get_paths()["include"])
    command = [compiler, "-std=c++17", "-fsyntax-only", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"]
    roots = list(dict.fromkeys([*torch_roots, *include_roots, python_include]))
    for root in roots:
        command.extend(["-I", str(root)])
    command.append("{source}")
    return ProbeSpec(
        probe_id="host_tensor_stream",
        source=(
            "#include <ATen/ATen.h>\n"
            '#include "torch_npu/csrc/core/npu/DeviceUtils.h"\n'
            '#include "torch_npu/csrc/core/npu/NPUStream.h"\n'
            "void probe_host_contract(const at::Tensor& input) {\n"
            "  bool is_npu = torch_npu::utils::is_npu(input);\n"
            "  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);\n"
            "  (void)is_npu;\n"
            "  (void)stream;\n"
            "}\n"
        ),
        command=tuple(command),
        fact_ids=(
            "runtime:is_npu",
            "runtime:getCurrentNPUStream",
            "runtime:aclrtStream",
        ),
    )


_KERNEL_API_PROBE_SOURCE = """#include "kernel_operator.h"
#include "math/tanh.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void cannagent_api_probe(GM_ADDR input, GM_ADDR output)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> halfBuf;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(floatBuf, 4096);
    pipe.InitBuffer(halfBuf, 2048);
    pipe.InitBuffer(workBuf, 8192);
    LocalTensor<float> values = floatBuf.Get<float>();
    LocalTensor<half> halves = halfBuf.Get<half>();
    LocalTensor<uint8_t> work = workBuf.Get<uint8_t>();
    Add(values, values, values, 256);
    Mul(values, values, values, 256);
    Muls(values, values, 0.5f, 256);
    Cast(values, halves, RoundMode::CAST_NONE, 256);
    Cast(halves, values, RoundMode::CAST_RINT, 256);
    Sqrt(values, values, 256);
    ReduceSum(values, values, values, 256);
    Tanh(values, values, work, 256);
    GlobalTensor<float> inputGm;
    GlobalTensor<float> outputGm;
    inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input), 256);
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), 256);
    outputGm.SetValue(0, inputGm.GetValue(0));
}
"""


def run_kernel_api_probe(
    *,
    output_dir: Path,
    fingerprint: EnvironmentFingerprint,
    toolkit_root: Path,
    soc_version: str,
    timeout: int = 180,
) -> ProbeResult:
    """Compile a minimal AscendC library containing the previously failing APIs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    probe_id = "kernel_api_contracts"
    with tempfile.TemporaryDirectory(prefix="cannagent-kernel-api-") as temporary:
        work = Path(temporary)
        source_path = work / "api_probe.cpp"
        source_path.write_text(_KERNEL_API_PROBE_SOURCE, encoding="utf-8")
        cmake_path = work / "CMakeLists.txt"
        cmake_path.write_text(
            f"""cmake_minimum_required(VERSION 3.16)
project(CannAgentApiProbe CXX)
set(SOC_VERSION "{soc_version}" CACHE STRING "SoC")
set(ASCEND_CANN_PACKAGE_PATH "{toolkit_root}" CACHE PATH "CANN")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
if(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)
else()
  message(FATAL_ERROR "ascendc_kernel_cmake not found")
endif()
include(${{ASCENDC_CMAKE_DIR}}/ascendc.cmake)
ascendc_library(cannagent_api_probe STATIC api_probe.cpp)
""",
            encoding="utf-8",
        )
        build = work / "build"
        configure = (
            "cmake", "-S", str(work), "-B", str(build),
            f"-DSOC_VERSION={soc_version}",
            f"-DASCEND_CANN_PACKAGE_PATH={toolkit_root}",
            "-DCMAKE_BUILD_TYPE=Release",
        )
        compile_command = ("cmake", "--build", str(build), "-j2")
        outputs: list[str] = []
        return_code = 127
        try:
            configured = subprocess.run(
                list(configure), cwd=work, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            outputs.append(configured.stdout or "")
            return_code = configured.returncode
            if return_code == 0:
                compiled = subprocess.run(
                    list(compile_command), cwd=work, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=timeout, check=False,
                )
                outputs.append(compiled.stdout or "")
                return_code = compiled.returncode
        except subprocess.TimeoutExpired as error:
            return_code = 124
            raw = error.stdout or ""
            outputs.append(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
            outputs.append(f"Timed out after {timeout}s")
        except OSError as error:
            outputs.append(f"Failed to run AscendC probe: {error}")
        output = "\n".join(outputs)

    log_path = output_dir / f"{probe_id}.log"
    log_path.write_text(output, encoding="utf-8")
    facts = (
        "runtime:Add", "runtime:Mul", "runtime:Muls", "runtime:Cast",
        "runtime:Sqrt", "runtime:ReduceSum", "runtime:Tanh", "runtime:GlobalTensor",
    )
    result = ProbeResult(
        probe_id=probe_id,
        success=return_code == 0,
        return_code=return_code,
        command=(*configure, "THEN", *compile_command),
        source_sha256=hashlib.sha256(_KERNEL_API_PROBE_SOURCE.encode("utf-8")).hexdigest(),
        output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        fingerprint_id=fingerprint.fingerprint_id,
        fact_ids=facts if return_code == 0 else (),
        log_path=str(log_path.resolve()),
    )
    (output_dir / f"{probe_id}.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run environment-bound CannAgent knowledge compile probes."
    )
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--soc-version", default="unknown")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    # Imported lazily to keep ordinary fact loading free of a module cycle.
    from .runtime_knowledge import _cann_root, _include_roots

    toolkit_root = _cann_root(args.runtime_version)
    if toolkit_root is None:
        parser.error(f"CANN {args.runtime_version} installation was not found")
    torch_roots, _ = torch_include_roots_without_import()
    fingerprint_roots = [
        *_include_roots(toolkit_root), *torch_npu_include_roots(), *torch_roots
    ]
    fingerprint = collect_environment_fingerprint(
        runtime_version=args.runtime_version,
        soc_version=args.soc_version,
        toolkit_root=toolkit_root,
        include_roots=fingerprint_roots,
        project_root=args.project_root.resolve(),
    )
    compiler_roots = [
        toolkit_root / "aarch64-linux/include",
        toolkit_root / "include",
        toolkit_root / "aarch64-linux/asc/include",
        *fingerprint_roots,
    ]
    spec = _host_probe_spec(
        include_roots=list(dict.fromkeys(path.resolve() for path in compiler_roots if path.is_dir())),
    )
    print(f"fingerprint={fingerprint.fingerprint_id}", flush=True)
    print("probe=host_tensor_stream", flush=True)
    print("command=" + " ".join(spec.command), flush=True)
    result = run_compile_probe(
        spec,
        output_dir=args.output_dir.resolve(),
        fingerprint=fingerprint,
        timeout=args.timeout,
    )
    print("probe=kernel_api_contracts", flush=True)
    print("command=cmake configure + AscendC build", flush=True)
    kernel_result = run_kernel_api_probe(
        output_dir=args.output_dir.resolve(),
        fingerprint=fingerprint,
        toolkit_root=toolkit_root,
        soc_version=args.soc_version,
        timeout=max(args.timeout, 180),
    )
    manifest = args.output_dir.resolve() / "manifest.json"
    write_probe_manifest(
        manifest, fingerprint=fingerprint, results=[result, kernel_result]
    )
    print(
        json.dumps(
            {
                "results": [result.to_dict(), kernel_result.to_dict()],
                "manifest": str(manifest),
            },
            indent=2,
        ),
        flush=True,
    )
    if not result.success:
        print(Path(result.log_path).read_text(encoding="utf-8", errors="replace"), flush=True)
    if not kernel_result.success:
        print(Path(kernel_result.log_path).read_text(encoding="utf-8", errors="replace"), flush=True)
    return 0 if result.success and kernel_result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
