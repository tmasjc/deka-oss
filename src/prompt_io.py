"""Shared two-fence markdown prompt parser.

The harness's prompt files (``harness/prompts/*.md``) follow a uniform
shape: explanatory markdown around triple-backtick fenced code blocks
that contain the actual prompt text. The same regex (``r"```\\n(.*?)```"``,
DOTALL) is used in :mod:`src.reflection.prompt`, :mod:`src.extraction.prompt`,
and now refine — this module exists so refine doesn't grow a third
copy.

Reflection and extraction keep their own loaders today; migrating them
is a pure refactor and out of scope for the Phase 3 cut.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_FENCE_RE = re.compile(r"```\n(.*?)```", re.DOTALL)


class PromptParseError(ValueError):
    """Raised when a prompt markdown file does not match the expected
    fence structure: missing block, missing header, or malformed nesting.

    Carries the offending path so callers can surface it without
    re-discovering it from the stack.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


def load_fenced_blocks(path: Path) -> list[str]:
    """Return every triple-backtick fenced block in ``path`` (stripped).

    Raises :class:`PromptParseError` if the file does not exist or
    contains no fenced blocks.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptParseError(f"Prompt file not found: {path}", path=path) from exc

    blocks = [m.strip() for m in _FENCE_RE.findall(text)]
    if not blocks:
        raise PromptParseError(
            f"No triple-backtick fenced blocks found in {path}", path=path
        )
    return blocks


def load_named_fence_sections(
    path: Path, expected_headers: tuple[str, ...]
) -> tuple[str, ...]:
    """Parse a markdown file with N named ``##`` sections, each
    containing exactly one fenced block.

    Returns one block per header, in declared order. Headers must
    appear in the file in the order given, each followed by at least
    one fenced block before the next header. Extra blocks inside a
    section are tolerated — the *first* block under each header is
    returned.
    """
    if len(expected_headers) < 1:
        raise ValueError("expected_headers must have at least 1 entry")

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptParseError(f"Prompt file not found: {path}", path=path) from exc

    offsets: list[int] = []
    cursor = 0
    for i, header in enumerate(expected_headers):
        idx = _find_header(text, header, start=cursor)
        if idx < 0:
            if i == 0:
                msg = f"{path}: missing header '## {header}'"
            else:
                prev = expected_headers[i - 1]
                msg = f"{path}: missing header '## {header}' (after '## {prev}')"
            raise PromptParseError(msg, path=path)
        offsets.append(idx)
        cursor = idx + 1

    blocks: list[str] = []
    for i, header in enumerate(expected_headers):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        block = _first_fence(text[start:end])
        if block is None:
            raise PromptParseError(
                f"{path}: no fenced block under '## {header}'", path=path
            )
        blocks.append(block)
    return tuple(blocks)


def load_two_fence_sections(
    path: Path, expected_headers: tuple[str, str]
) -> tuple[str, str]:
    """Two-section convenience wrapper around
    :func:`load_named_fence_sections`. Kept for back-compat with
    callers that expect exactly two blocks.
    """
    if len(expected_headers) != 2:
        raise ValueError("expected_headers must have exactly 2 entries")
    blocks = load_named_fence_sections(path, expected_headers)
    return blocks[0], blocks[1]


def _find_header(text: str, header: str, *, start: int = 0) -> int:
    """Find ``## <header>`` at the start of a line, after ``start``.

    Returns the offset of the ``##`` or -1 if not present. The match
    is anchored to a line start so a ``## System`` mention inside a
    fenced block (where it would render as plain text under, say, a
    ``###`` heading documentation example) does not false-positive.
    """
    needle = f"## {header}"
    idx = start
    while True:
        found = text.find(needle, idx)
        if found < 0:
            return -1
        if found == 0 or text[found - 1] == "\n":
            return found
        idx = found + 1


def _first_fence(section: str) -> str | None:
    match = _FENCE_RE.search(section)
    return match.group(1).strip() if match else None


def prompt_sha256(text: str) -> str:
    """Canonical SHA-256 digest of a prompt's text.

    Used to stamp ``prompt_sha256`` and ``meta_prompt_sha256`` on every
    Phase 3 artifact so re-reads can verify the prompt has not drifted
    between the time of the run and the time of audit.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
