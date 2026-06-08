"""Thin CLI for verifying ``run_search`` end-to-end against live services.

Usage::

    uv run python -m src.search.cli "家长对价格的犹豫和老师的应对话术"

Not meant to be the long-term user-facing interface — the harness
product owner will use the TUI (see ``docs/PRODUCT.md``). This module
exists so ``run_search`` can be smoke-tested quickly.
"""

from __future__ import annotations

import logging
import sys

from .config import load_default_config
from .search import run_search


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0].strip():
        print('Usage: python -m src.search.cli "<query>"', file=sys.stderr)
        return 2

    query = args[0]
    config = load_default_config()

    table = run_search(query, config)

    _render(table)
    return 0


def _render(table) -> None:
    print()
    print(f"Query: {table.query!r}")
    print(
        f"Config: RRFRanker k={table.config.rrf_k} "
        f"per_path_limit={table.config.per_path_limit} "
        f"top_k={table.config.top_k}"
    )
    print("-" * 78)
    if not table.rows:
        print("(no results)")
        return
    for row in table.rows:
        paths = ",".join(row.source_paths) if row.source_paths else "-"
        content = row.chunk_content.replace("\n", " ")
        if len(content) > 120:
            content = content[:120] + "..."
        scores = (
            f"dense={row.scores['dense']:.4f} "
            f"sparse={row.scores['sparse']:.4f}"
        )
        print(f"{row.rank:>2}. {row.chunk_id}  paths=[{paths}]  {scores}")
        print(f"     {content}")
    print("-" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
