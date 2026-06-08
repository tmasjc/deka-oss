"""Attach a per-session FileHandler to the root logger.

The handler captures every existing ``log.info/warning/debug`` call
across the codebase (search, reflection, TUI) at DEBUG level, without
touching the existing stderr handler.
"""

from __future__ import annotations

import logging
from pathlib import Path

_ATTACHED: dict[str, logging.FileHandler] = {}

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_run_logging(session_id: str, runs_dir: Path) -> Path:
    """Attach a DEBUG-level FileHandler for this session. Returns the
    log file path. Idempotent per session_id."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{session_id}.log"

    if session_id in _ATTACHED:
        return log_path

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.set_name(f"deka-session-{session_id}")

    root = logging.getLogger()
    if root.level > logging.DEBUG or root.level == logging.NOTSET:
        root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    _ATTACHED[session_id] = handler
    return log_path


def flush_run_logging(session_id: str) -> None:
    """Flush the FileHandler for this session so buffered lines hit disk.

    Used before reading/zipping the log file mid-session. No-op if the
    session has no attached handler."""
    handler = _ATTACHED.get(session_id)
    if handler is not None:
        handler.flush()


def teardown_run_logging(session_id: str) -> None:
    """Detach and close the FileHandler previously attached for this
    session. No-op if the session was never attached or already torn
    down."""
    handler = _ATTACHED.pop(session_id, None)
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    handler.close()
