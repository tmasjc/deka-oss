"""Phase 3 — Boundary refinement via rubric prompt.

Phase 3 reads a converged Phase 1 session plus its Phase 2 sidecar and
produces a per-query, plug-and-play **rubric prompt** the operator can
ship downstream. The rubric prompt names the linguistic checks that
separate FIT chunks from NOT_FIT chunks, in a markdown shape any
LLM-with-JSON-mode endpoint can apply chunk-by-chunk.

Public surface kept narrow until callers settle.
"""

from .errors import (
    RefineConfigError,
    RefineError,
    RefineJudgeError,
    RefineLoadError,
    RefineParseError,
    RefineValidationError,
)

__all__ = [
    "RefineError",
    "RefineConfigError",
    "RefineLoadError",
    "RefineParseError",
    "RefineValidationError",
    "RefineJudgeError",
]
