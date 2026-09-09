"""Stable public CLI for the semantic knowledge snapshot compiler."""

from ..knowledge_v2.build import main, parser


__all__ = ["main", "parser"]


if __name__ == "__main__":
    raise SystemExit(main())
