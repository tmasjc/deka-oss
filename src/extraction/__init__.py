"""LLM-driven concept-span extraction for Phase 1 rating.

Exposes :class:`SpanExtractor`, which extracts 0-3 lines from
a chunk that express a query's concept. Results attach to
``EvidenceRow`` / ``CandidateRow`` before the TUI renders them so the
human rater sees the span highlighted inside the full chunk.
"""

from .cache import CacheKey, SpanCache
from .dual import DualSpanExtractor
from .errors import CacheError, ExtractionError
from .extractor import SpanExtractor
from .schema import SpanExtractionResult
from .session import build_extract_span, gather_prior_fit_spans

__all__ = [
    "CacheError",
    "CacheKey",
    "DualSpanExtractor",
    "ExtractionError",
    "SpanCache",
    "SpanExtractor",
    "SpanExtractionResult",
    "build_extract_span",
    "gather_prior_fit_spans",
]
