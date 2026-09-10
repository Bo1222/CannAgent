from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..knowledge_paths import DEFAULT_KNOWLEDGE_SOURCE, DEFAULT_KNOWLEDGE_STORE
from .knowledge_build import build_knowledge


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compile and publish versioned structured AscendC knowledge"
    )
    result.add_argument("--source", default=str(DEFAULT_KNOWLEDGE_SOURCE))
    result.add_argument("--platform", choices=("cann",), default="cann")
    result.add_argument("--version", required=True)
    result.add_argument("--output", default=str(DEFAULT_KNOWLEDGE_STORE))
    return result


def main() -> int:
    args = parser().parse_args()
    destination = build_knowledge(
        source=Path(args.source), output=Path(args.output), version=args.version
    )
    print(
        json.dumps(
            {
                "knowledge_build": str(destination),
                "knowledge_build_id": destination.name,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
