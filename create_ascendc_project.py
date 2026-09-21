from __future__ import annotations

import argparse

from ascendc_multi_turn.project_generator import ProjectGenerationError, create_project


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Create a template-based AscendC direct-call project")
    result.add_argument("--op-name", required=True, help="safe C/C++ operator identifier")
    result.add_argument("--op-file", required=True, help="Python reference containing Model.forward")
    result.add_argument("--op-json", required=True, help="JSON or JSONL cases copied without modification")
    result.add_argument("--output", required=True, help="empty or absent output directory")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        create_project(
            op_name=args.op_name,
            op_file=args.op_file,
            op_json=args.op_json,
            output=args.output,
        )
    except ProjectGenerationError as error:
        parser().error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
