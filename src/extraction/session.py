"""Per-turn extractor closure that carries session-accumulated FIT spans.

The extractor prompt is grounded in up to N prior FIT spans from the
same session so the concept definition stabilizes across turns. The
callable built here is handed to :func:`src.search.search.run_search`
as ``extract_span=``; each call inside ``run_search`` pays at most one
LLM round-trip (cache hits are free). The callable also exposes
``model``, ``prompt_version``, ``latency_ms_total``, and ``cache_hits``
attributes so ``run_search`` can fold them into the observability
block without coupling to the extractor type.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from .errors import ExtractionError
from .extractor import SpanExtractor

if TYPE_CHECKING:
    from src.session.state import SessionState

log = logging.getLogger(__name__)

_PRIOR_SPAN_CAP = 5


def gather_prior_fit_spans(
    state: "SessionState", cap: int = _PRIOR_SPAN_CAP
) -> list[str]:
    """Return the last ``cap`` FIT-rated span_texts seen in the session.

    Pulls from both fused rows and per-path candidates so re-activation
    of a path doesn't forfeit its spans as grounding. Empty ``span_text``
    entries are skipped — they carry no signal.
    """
    out: list[str] = []
    for turn in state.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "FIT" and row.span_text:
                out.append(row.span_text)
        for candidates in turn.evidence_table.per_path_candidates.values():
            for cand in candidates:
                if cand.rating == "FIT" and cand.span_text:
                    out.append(cand.span_text)
    return out[-cap:]


class _TurnExtractSpan:
    """Callable closure with readable per-turn stats.

    Delta-accounting against the extractor's running totals: the
    ``latency_ms_total`` and ``cache_hits`` properties measure only
    what happened during this turn's extraction loop, not across all
    prior turns. ``run_search`` reads the properties after the loop
    completes.
    """

    def __init__(
        self,
        extractor: SpanExtractor,
        state: "SessionState",
        query: str,
        on_progress: "Callable[[int], None] | None" = None,
    ) -> None:
        self._extractor = extractor
        self._prior = gather_prior_fit_spans(state)
        self._query = query
        self._latency_start = extractor.latency_ms_total
        self._cache_hits_start = extractor.cache_hits
        self._on_progress = on_progress
        self._failures = 0
        self._processed = 0

    def __call__(self, chunk_content: str) -> tuple[list[int], str] | None:
        """Return the span for a chunk, or ``None`` to drop it from results.

        - ``(indices, text)`` on success (either a real span or a legitimate
          empty-span verdict from the model).
        - ``None`` when extraction errored. The caller (``run_search``) drops
          rows that come back ``None`` so the rater never sees an
          un-evaluated chunk; a toast surfaces how many were dropped.
        """
        try:
            result = self._extractor.extract(
                query=self._query,
                chunk_content=chunk_content,
                prior_fit_spans=self._prior,
            )
        except ExtractionError as exc:
            # One bad chunk must not kill the turn, but we also don't
            # want to render a partially-evaluated chunk as if it had a
            # valid no-span verdict. Return None so run_search drops it.
            preview = chunk_content.replace("\n", " / ")[:80]
            log.warning(
                "Span extraction failed on chunk (first 80 chars: %r): %s",
                preview,
                exc,
            )
            self._failures += 1
            self._processed += 1
            if self._on_progress is not None:
                self._on_progress(self._processed)
            return None
        indices = list(result.span_line_indices)
        text = self._extractor.derive_span_text(chunk_content, indices)
        self._processed += 1
        if self._on_progress is not None:
            self._on_progress(self._processed)
        return indices, text

    @property
    def model(self) -> str:
        return self._extractor.model

    @property
    def prompt_version(self) -> str:
        return self._extractor.prompt_version

    @property
    def latency_ms_total(self) -> float:
        return round(self._extractor.latency_ms_total - self._latency_start, 2)

    @property
    def cache_hits(self) -> int:
        return self._extractor.cache_hits - self._cache_hits_start

    @property
    def failures(self) -> int:
        """Count of chunks where extraction errored and fell back to empty span."""
        return self._failures


def build_extract_span(
    extractor: SpanExtractor,
    state: "SessionState",
    query: str,
    on_progress: Callable[[int], None] | None = None,
) -> _TurnExtractSpan:
    """Bind ``extractor`` + session grounding into a per-turn callable.

    ``on_progress`` is called with the running processed-count after each
    extraction attempt (success or failure). Use it to drive a progress
    indicator while the closure runs on a worker thread.
    """
    return _TurnExtractSpan(extractor, state, query, on_progress=on_progress)
