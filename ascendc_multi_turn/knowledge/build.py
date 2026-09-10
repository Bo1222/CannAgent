"""Public CLI for compiling and publishing structured AscendC knowledge."""

from ..structured_knowledge.build import main, parser


__all__ = ["main", "parser"]


if __name__ == "__main__":
    raise SystemExit(main())
