"""Session-scoped JSONL log for span-extraction results.

One file per session: ``runs/{session_id}.span_cache.jsonl``. Each line
stores only the result and the derived span text — no key context — so the
file is a compact human-readable audit trail. The in-memory dict (keyed by
hash) provides O(1) within-session cache lookups.

No cross-session reload: the file is append-only and written by one process.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import SpanExtractionResult


@dataclass(frozen=True)
class CacheKey:
    """Canonical cache key for a span-extraction call."""

    model_id: str
    prompt_version: str
    query: str
    chunk_content: str

    def sha256(self) -> str:
        payload = json.dumps(
            [self.model_id, self.prompt_version, self.query, self.chunk_content],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _derive_span_text(chunk_content: str, indices: list[int]) -> str:
    if not indices:
        return ""
    lines = chunk_content.split("\n")
    return "\n".join(lines[i] for i in indices if i < len(lines))


class SpanCache:
    """In-memory cache backed by a session-scoped append-only JSONL file.

    Pass the full path to the JSONL file (e.g.
    ``runs/{session_id}.span_cache.jsonl``). The parent directory is created
    on first use.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, SpanExtractionResult] = {}

    @property
    def path(self) -> Path:
        return self._path

    def get(self, key: CacheKey) -> SpanExtractionResult | None:
        return self._store.get(key.sha256())

    def put(self, key: CacheKey, result: SpanExtractionResult) -> None:
        span_text = _derive_span_text(key.chunk_content, list(result.span_line_indices))
        entry = {
            "span_line_indices": list(result.span_line_indices),
            "reason": result.reason,
            "span_text": span_text,
        }
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False))
            fp.write("\n")
            fp.flush()
        self._store[key.sha256()] = result
