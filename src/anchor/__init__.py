"""FIT-span-anchored retrieval (Phase 2).

Public surface:

* :func:`run_anchor` — orchestrator used by the TUI and CLI.
* :class:`AnchorResult` — the full result envelope.
* Individual stage functions for tests / advanced callers.
"""

from .errors import (
    AnchorError,
    AnchorLoadError,
    AnchorRetrievalError,
    AnchorValidationError,
)

__all__ = [
    "AnchorError",
    "AnchorLoadError",
    "AnchorRetrievalError",
    "AnchorValidationError",
]
