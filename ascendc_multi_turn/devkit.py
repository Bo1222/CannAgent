from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

ASC_DEVKIT_VERSION = "9.1.0"
ASC_DEVKIT_REF = "9.1.0"
ASC_DEVKIT_COMMIT = "c785b5f76f23c9dc0ddb553174463d0497e59288"
ASC_DEVKIT_REPOSITORY = "https://gitcode.com/cann/asc-devkit.git"


class DevkitError(RuntimeError):
    pass


@dataclass(frozen=True)
class DevkitStatus:
    path: str
    healthy: bool
    version: str | None
    commit: str | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeVersion:
    runtime_version: str
    knowledge_version: str = ASC_DEVKIT_VERSION
    status: str = "exact"
    warning: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def detect_cann_version(explicit: str = "auto", *, mock: bool = False) -> str:
    if explicit and explicit.lower() != "auto":
        match = re.search(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)", explicit)
        if not match:
            raise DevkitError(f"invalid CANN version: {explicit!r}")
        parts = match.group(1).split(".")
        return ".".join(parts + ["0"] * (3 - len(parts)))
    if mock:
        return ASC_DEVKIT_VERSION
    candidates: list[Path] = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH"):
        if os.environ.get(name):
            candidates.append(Path(os.environ[name]).expanduser())
    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/ascend-toolkit"),
            Path("/usr/local/Ascend"),
        ]
    )
    for root in candidates:
        for relative in ("compiler/version.info", "opp/version.info", "version.info"):
            path = root / relative
            if not path.is_file():
                continue
            match = re.search(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)", path.read_text(encoding="utf-8", errors="replace"))
            if match:
                parts = match.group(1).split(".")
                return ".".join(parts + ["0"] * (3 - len(parts)))
    raise DevkitError("cannot detect installed CANN version")


def resolve_runtime_version(explicit: str = "auto", *, mock: bool = False) -> RuntimeVersion:
    runtime = detect_cann_version(explicit, mock=mock)
    status = "exact" if runtime == ASC_DEVKIT_VERSION else "mismatch"
    warning = "" if status == "exact" else f"installed CANN {runtime} does not match required {ASC_DEVKIT_VERSION}"
    return RuntimeVersion(runtime_version=runtime, status=status, warning=warning)


def default_cache_root() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return base / "cannagent" / "asc-devkit"


def managed_devkit_path(cache_root: Path | None = None) -> Path:
    return (cache_root or default_cache_root()) / ASC_DEVKIT_COMMIT


def _declared_version(path: Path) -> str | None:
    version_file = path / "version.cmake"
    if not version_file.is_file():
        return None
    text = version_file.read_text(encoding="utf-8", errors="replace")
    import re

    match = re.search(r"(?:ASCEND_CANN_PACKAGE_VERSION|ASC_DEVKIT_VERSION)\s+[\"']?([0-9.]+)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"\b(?:9\.1\.0|9\.1)\b", text)
    return numbers[0] if numbers else None


def inspect_devkit(path: Path | str) -> DevkitStatus:
    root = Path(path).expanduser().resolve()
    errors: list[str] = []
    for relative in ("version.cmake", "docs/api", "examples", "include", "impl"):
        if not (root / relative).exists():
            errors.append(f"missing:{relative}")
    version = _declared_version(root)
    if version != ASC_DEVKIT_VERSION:
        errors.append(f"version:{version or 'unknown'}!=${ASC_DEVKIT_VERSION}".replace("$", ""))
    commit: str | None = None
    if (root / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip()
            if commit != ASC_DEVKIT_COMMIT:
                errors.append(f"commit:{commit}!=${ASC_DEVKIT_COMMIT}".replace("$", ""))
        else:
            errors.append("commit:unreadable")
    else:
        marker = root / ".cannagent-devkit.json"
        if marker.is_file():
            try:
                commit = str(json.loads(marker.read_text(encoding="utf-8")).get("commit") or "") or None
            except (OSError, json.JSONDecodeError):
                errors.append("marker:invalid")
            if commit != ASC_DEVKIT_COMMIT:
                errors.append(f"commit:{commit or 'unknown'}!=${ASC_DEVKIT_COMMIT}".replace("$", ""))
        else:
            errors.append("commit:unverifiable")
    return DevkitStatus(str(root), not errors, version, commit, tuple(errors))


def resolve_devkit(explicit: Path | str | None = None) -> Path:
    candidate = explicit or os.environ.get("ASC_DEVKIT_DIR", "").strip() or managed_devkit_path()
    status = inspect_devkit(candidate)
    if not status.healthy:
        raise DevkitError(
            f"Asc DevKit v{ASC_DEVKIT_VERSION} is unavailable or unhealthy at {status.path}: "
            + ", ".join(status.errors)
            + "; run `python -m ascendc_multi_turn.devkit init`"
        )
    return Path(status.path)


def initialize_devkit(
    *,
    cache_root: Path | None = None,
    repository: str = ASC_DEVKIT_REPOSITORY,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> DevkitStatus:
    destination = managed_devkit_path(cache_root)
    existing = inspect_devkit(destination)
    if existing.healthy:
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{ASC_DEVKIT_COMMIT}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise DevkitError(f"DevKit initialization is already running: {lock}") from error
    os.close(descriptor)
    temporary = Path(tempfile.mkdtemp(prefix="asc-devkit-", dir=destination.parent))
    try:
        result = command_runner(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--single-branch",
                "--branch",
                ASC_DEVKIT_REF,
                repository,
                str(temporary),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise DevkitError(f"failed to clone Asc DevKit: {result.stderr.strip()}")
        result = command_runner(
            ["git", "-C", str(temporary), "checkout", "--detach", ASC_DEVKIT_COMMIT],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise DevkitError(f"failed to checkout {ASC_DEVKIT_COMMIT}: {result.stderr.strip()}")
        status = inspect_devkit(temporary)
        if not status.healthy:
            raise DevkitError("downloaded DevKit failed health checks: " + ", ".join(status.errors))
        if destination.exists():
            quarantine = destination.with_name(f"{destination.name}.invalid-{int(time.time())}")
            os.replace(destination, quarantine)
        os.replace(temporary, destination)
        temporary = Path()
        return inspect_devkit(destination)
    finally:
        if temporary and temporary.exists() and temporary != Path("."):
            shutil.rmtree(temporary, ignore_errors=True)
        lock.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage the pinned Asc DevKit used by CannAgent")
    result.add_argument("command", choices=("init", "status"))
    result.add_argument("--cache-root", type=Path, default=None)
    result.add_argument("--asc-devkit-dir", type=Path, default=None)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "init":
        status = initialize_devkit(cache_root=args.cache_root)
    else:
        target = args.asc_devkit_dir or os.environ.get("ASC_DEVKIT_DIR") or managed_devkit_path(args.cache_root)
        status = inspect_devkit(target)
    print(json.dumps(status.to_dict(), ensure_ascii=False, indent=2))
    return 0 if status.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
