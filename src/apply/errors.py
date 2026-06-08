"""Apply-phase exceptions.

Mirrors :mod:`src.refine.errors`. Every Phase 4 failure mode raises a
subclass of :class:`ApplyError` so callers funnel config / load /
guardrail / train / predict problems through one error surface.
"""

from __future__ import annotations


class ApplyError(Exception):
    """Base for Phase 4 (apply) failures."""


class ApplyConfigError(ApplyError):
    """Config section missing, malformed, or carries invalid values."""


class ApplyLoadError(ApplyError):
    """Phase 2 or Phase 3 sidecar missing or malformed; classifier file
    missing or malformed."""


class ApplyGuardrailError(ApplyError):
    """Rubric drift detected: persisted classifier's ``rubric_version`` or
    ``prompt_sha256`` does not match the session's current rubric.

    Same lock discipline as Phase 1's locked config. Refusing to apply
    on drift is by design — rubric drift silently invalidates the
    classifier's calibration.
    """


class ApplyPrecisionError(ApplyError):
    """Eval precision fell below the configured ``apply.min_precision``
    bar and the caller did not supply an explicit override flag.

    Headless callers see this as a hard exit; the web UI surfaces it
    as a warning gate that requires an explicit override click.
    """


class ApplyValidationError(ApplyError):
    """Precondition or gate refused the run — no sidecar written."""
