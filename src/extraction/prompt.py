"""Prompt loading + message assembly for the span extractor.

The Markdown source at ``harness/prompts/EXTRACTION.md`` carries the
system block and the user-template block as triple-backtick code blocks
(first = system, second = user). This module extracts both, fills the
user template with the query, the line-numbered chunk, and optional
prior FIT spans, and returns the message list for the OpenAI chat API.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.paths import prompt_path

from .errors import ExtractionError

_PRIOR_SPAN_CAP = 5


def _load_blocks(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ExtractionError(f"Prompt file not found: {path}") from exc

    blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if len(blocks) < 2:
        raise ExtractionError(
            f"Expected >= 2 code blocks in {path}, found {len(blocks)}"
        )
    return blocks[0].strip(), blocks[1].strip()


def load_extraction_prompts(repo_root: Path | None = None) -> tuple[str, str]:
    """Return ``(system_block, user_template)`` from EXTRACTION.md."""
    path = (
        repo_root / "harness" / "prompts" / "EXTRACTION.md"
        if repo_root is not None
        else prompt_path("EXTRACTION.md")
    )
    return _load_blocks(path)


def _number_chunk(chunk_content: str) -> str:
    lines = chunk_content.split("\n")
    return "\n".join(f"[{i}] {line}" for i, line in enumerate(lines))


def _render_prior_fit_spans(spans: list[str]) -> str:
    if not spans:
        return ""
    capped = spans[-_PRIOR_SPAN_CAP:]
    rendered = "\n".join(f"- {s}" for s in capped)
    return f"Prior FIT spans from this session (examples of the concept):\n{rendered}\n"


def build_messages(
    query: str,
    prior_fit_spans: list[str],
    chunk_content: str,
    *,
    system_block: str | None = None,
    user_template: str | None = None,
) -> list[dict[str, str]]:
    """Assemble the 2-message request body for one chunk's extraction."""
    if system_block is None or user_template is None:
        loaded_system, loaded_user = load_extraction_prompts()
        system_block = system_block or loaded_system
        user_template = user_template or loaded_user

    user = user_template.format(
        query=query,
        prior_fit_spans_block=_render_prior_fit_spans(prior_fit_spans),
        numbered_chunk=_number_chunk(chunk_content),
    )
    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user},
    ]
