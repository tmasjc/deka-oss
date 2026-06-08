"""Anchor-phase exceptions."""

from __future__ import annotations


class AnchorError(Exception):
    """Base for anchor-phase failures."""


class AnchorLoadError(AnchorError):
    """Session log missing, malformed, or carries no FIT-rated chunks.

    Also raised when the span/chunk embedding prerequisites can't be
    satisfied: empty span_text on a FIT row, missing details sidecar
    fields, embed-model drift between Phase 1 and Phase 2, or a FIT
    pk absent from the target Milvus collection.
    """


class AnchorRetrievalError(AnchorError):
    """Milvus k-NN search failed during the anchor pass."""


class AnchorValidationError(AnchorError):
    """Phase 2 precondition or gate refused the run — no sidecar written.

    Raised when the session has not converged, the LOO recovery falls
    below the hard-fail floor, the anchor-quality gate drops the cohort
    below the structural LOO floor, or the anchor-frequency gate is
    configured for more anchors than survived quality gating.
    """
