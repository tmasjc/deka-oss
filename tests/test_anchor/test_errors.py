"""Error-class smoke tests for src.anchor.errors."""

from __future__ import annotations

from src.anchor.errors import (
    AnchorError,
    AnchorLoadError,
    AnchorRetrievalError,
    AnchorValidationError,
)


def test_retrieval_error_is_anchor_error():
    assert issubclass(AnchorRetrievalError, AnchorError)


def test_load_error_is_anchor_error():
    assert issubclass(AnchorLoadError, AnchorError)


def test_validation_error_is_anchor_error():
    assert issubclass(AnchorValidationError, AnchorError)


def test_saturation_error_is_gone():
    """Phase 2 retired AnchorSaturationError — budget exhaustion is a
    warning, not an exception. Importing it must fail."""
    import src.anchor.errors as errors_mod

    assert not hasattr(errors_mod, "AnchorSaturationError")
