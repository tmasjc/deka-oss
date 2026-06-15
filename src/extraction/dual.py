"""Dual-vendor span extraction — issue #54.

Wraps two :class:`SpanExtractor` instances and runs them in parallel
on a 2-worker :class:`~concurrent.futures.ThreadPoolExecutor` per
chunk. The two vendors' ``span_line_indices`` are merged by sorted-
unique union (trading ~2× LLM cost for higher recall on the most
leverage-bearing step in the pipeline). When only one vendor
succeeds, its result is used alone — only when both fail does the
chunk drop via the existing ``_TurnExtractSpan`` handler at
``session.py:91``.

Exposes the same duck-typed surface ``_TurnExtractSpan`` consumes —
``extract()``, ``derive_span_text()``, and the ``model`` /
``prompt_version`` / ``cache_hits`` / ``latency_ms_total`` /
``call_count`` / ``reset_stats`` properties. Stats aggregate across
the two vendors.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

from .errors import ExtractionError
from .extractor import SpanExtractor
from .schema import SpanExtractionResult

log = logging.getLogger(__name__)


class DualSpanExtractor:
    """Two-vendor wrapper around :class:`SpanExtractor`."""

    def __init__(
        self,
        primary: SpanExtractor,
        secondary: SpanExtractor,
    ) -> None:
        if primary.prompt_version != secondary.prompt_version:
            raise ExtractionError(
                f"Dual extractors must share prompt_version (got "
                f"{primary.prompt_version!r} and "
                f"{secondary.prompt_version!r})"
            )
        if primary.model == secondary.model:
            raise ExtractionError(
                f"Dual extractors must use distinct models (both = "
                f"{primary.model!r}) — same cache key would collide"
            )
        self._primary = primary
        self._secondary = secondary
        self._pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="span-extract-dual"
        )

    @property
    def model(self) -> str:
        return f"{self._primary.model}+{self._secondary.model}"

    @property
    def prompt_version(self) -> str:
        return self._primary.prompt_version

    @property
    def cache_hits(self) -> int:
        return self._primary.cache_hits + self._secondary.cache_hits

    @property
    def call_count(self) -> int:
        return self._primary.call_count + self._secondary.call_count

    @property
    def latency_ms_total(self) -> float:
        return round(
            self._primary.latency_ms_total + self._secondary.latency_ms_total,
            2,
        )

    def reset_stats(self) -> None:
        self._primary.reset_stats()
        self._secondary.reset_stats()

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        # Each sub-extractor owns its own OpenAI httpx pool (built
        # independently in ``SpanExtractor.__init__``); they do not
        # alias, so close both. Best-effort so one failure can't leak
        # the other.
        for extractor in (self._primary, self._secondary):
            close = getattr(extractor, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("DualSpanExtractor sub-extractor close raised: %s", exc)

    def __del__(self) -> None:
        # Best-effort cleanup if the owner forgot to call ``close()``.
        try:
            self._pool.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass

    def extract(
        self,
        *,
        query: str,
        chunk_content: str,
        prior_fit_spans: list[str],
    ) -> SpanExtractionResult:
        fut_a = self._pool.submit(
            self._primary.extract,
            query=query,
            chunk_content=chunk_content,
            prior_fit_spans=prior_fit_spans,
        )
        fut_b = self._pool.submit(
            self._secondary.extract,
            query=query,
            chunk_content=chunk_content,
            prior_fit_spans=prior_fit_spans,
        )
        result_a, err_a = _resolve(fut_a, vendor="primary")
        result_b, err_b = _resolve(fut_b, vendor="secondary")

        if result_a is None and result_b is None:
            raise ExtractionError(
                f"Both vendors failed: primary={err_a}; secondary={err_b}"
            )
        if result_a is None:
            log.warning(
                "Dual extract: primary (%s) failed (%s); using secondary alone.",
                self._primary.model,
                err_a,
            )
            return result_b  # type: ignore[return-value]
        if result_b is None:
            log.warning(
                "Dual extract: secondary (%s) failed (%s); using primary alone.",
                self._secondary.model,
                err_b,
            )
            return result_a

        union = sorted(
            set(result_a.span_line_indices) | set(result_b.span_line_indices)
        )
        merged_reason = f"[primary] {result_a.reason} || [secondary] {result_b.reason}"
        # ``model_construct`` bypasses the Pydantic length-cap
        # validator (≤3) so a legitimate 4–6 line union isn't
        # rejected. Each vendor's individual output still went
        # through the full validator + bounds-check in extract(),
        # so the other invariants (non-negative, sorted, unique,
        # within chunk bounds) hold by construction on the union.
        return SpanExtractionResult.model_construct(
            span_line_indices=union, reason=merged_reason
        )

    @staticmethod
    def derive_span_text(chunk_content: str, indices: list[int]) -> str:
        return SpanExtractor.derive_span_text(chunk_content, indices)


def _resolve(
    fut: Future, *, vendor: str
) -> tuple[SpanExtractionResult | None, ExtractionError | None]:
    """Wait on a vendor future. Treat ``ExtractionError`` as a
    survivable per-vendor failure; let unexpected exceptions
    propagate (they're bugs, not provider errors)."""
    try:
        return fut.result(), None
    except ExtractionError as exc:
        log.debug("Dual extract: %s vendor raised ExtractionError: %s", vendor, exc)
        return None, exc
