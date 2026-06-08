"""Refine-phase exceptions.

Mirrors :mod:`src.anchor.errors`. Every Phase 3 failure mode raises a
subclass of :class:`RefineError` so callers can funnel config / load /
parse / validation / judge problems through a single error surface.
"""

from __future__ import annotations


class RefineError(Exception):
    """Base for Phase 3 (refine) failures."""


class RefineConfigError(RefineError):
    """Config section missing, malformed, or carries invalid values.

    Also raised when a hard prerequisite is missing — e.g. Postgres is
    disabled in the unified config but the judge needs it to fetch
    chunk content.
    """


class RefineLoadError(RefineError):
    """Phase 1 progress log or Phase 2 sidecar missing, malformed, or
    inconsistent (e.g. session not converged, no FITs, no Phase 2 jsonl).
    """


class RefineParseError(RefineError):
    """A two-fence markdown prompt (meta-prompt or rubric prompt) failed
    structural validation: missing fence header, malformed
    ``failed_check`` enum, missing ``{numbered_chunk}`` placeholder, or
    unparseable check block.
    """


class RefineValidationError(RefineError):
    """Phase 3 precondition or gate refused the run — no sidecar written.

    Covers convergence-not-met, sample-budget misconfig, and rubric
    metadata round-trip failure.
    """


class RefineJudgeError(RefineError):
    """Async judge call exhausted retries or returned schema-invalid
    output that did not parse into ``JudgeVerdict`` after the configured
    retry budget. Per-chunk errors persist as ``verdict="ERROR"`` rows;
    this exception surfaces only catastrophic failures (e.g. the
    judge endpoint is unreachable for the entire run).
    """
