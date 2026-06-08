"""Unit tests for the per-session FileHandler attach/detach helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from src.logging import setup_run_logging, teardown_run_logging
from src.logging.run_logger import _ATTACHED


def _session_handlers(session_id: str) -> list[logging.Handler]:
    name = f"deka-session-{session_id}"
    return [h for h in logging.getLogger().handlers if h.get_name() == name]


def test_setup_attaches_handler_and_writes_to_file(tmp_path: Path) -> None:
    session_id = "sess-attach"
    try:
        log_path = setup_run_logging(session_id, tmp_path)

        assert log_path == tmp_path / f"{session_id}.log"
        assert session_id in _ATTACHED
        assert len(_session_handlers(session_id)) == 1

        logging.getLogger("test.attach").info("hello world")

        assert "hello world" in log_path.read_text(encoding="utf-8")
    finally:
        teardown_run_logging(session_id)


def test_teardown_detaches_and_stops_writing(tmp_path: Path) -> None:
    session_id = "sess-detach"
    log_path = setup_run_logging(session_id, tmp_path)
    logging.getLogger("test.detach").info("before teardown")

    teardown_run_logging(session_id)

    assert session_id not in _ATTACHED
    assert _session_handlers(session_id) == []

    logging.getLogger("test.detach").info("after teardown")

    contents = log_path.read_text(encoding="utf-8")
    assert "before teardown" in contents
    assert "after teardown" not in contents


def test_teardown_is_noop_for_unknown_session() -> None:
    teardown_run_logging("never-attached")  # must not raise


def test_setup_is_idempotent(tmp_path: Path) -> None:
    session_id = "sess-idem"
    try:
        p1 = setup_run_logging(session_id, tmp_path)
        p2 = setup_run_logging(session_id, tmp_path)

        assert p1 == p2
        assert len(_session_handlers(session_id)) == 1
    finally:
        teardown_run_logging(session_id)


def test_two_sessions_do_not_cross_contaminate(tmp_path: Path) -> None:
    sid_a, sid_b = "sess-a", "sess-b"
    try:
        path_a = setup_run_logging(sid_a, tmp_path)
        path_b = setup_run_logging(sid_b, tmp_path)

        # Both handlers are attached simultaneously; this is expected during
        # a session's active window. We verify both files receive records.
        logging.getLogger("test.cross").info("shared record")
        assert "shared record" in path_a.read_text(encoding="utf-8")
        assert "shared record" in path_b.read_text(encoding="utf-8")

        teardown_run_logging(sid_a)
        logging.getLogger("test.cross").info("only-b record")

        assert "only-b record" not in path_a.read_text(encoding="utf-8")
        assert "only-b record" in path_b.read_text(encoding="utf-8")
    finally:
        teardown_run_logging(sid_a)
        teardown_run_logging(sid_b)
