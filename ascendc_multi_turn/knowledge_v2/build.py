from __future__ import annotations

import argparse
import json
from pathlib import Path

from .snapshot import build_snapshot


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build an immutable AscendC semantic knowledge snapshot")
    result.add_argument("--source", required=True)
    result.add_argument("--platform", choices=("cann",), default="cann")
    result.add_argument("--version", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    destination = build_snapshot(
        source=Path(args.source), output=Path(args.output), version=args.version
    )
    print(json.dumps({"snapshot": str(destination), "snapshot_id": destination.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
