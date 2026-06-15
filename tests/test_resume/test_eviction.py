"""Idle-eviction sweep semantics.

Direct unit tests on ``SessionStore.evict_idle`` — the lifespan's
60-second cadence is incidental; what matters is that the predicate
"stage is idle AND idle for >= threshold" correctly identifies which
contexts to drop.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.search.config import SearchConfig
from src.session.state import SessionState
from src.scopes import Scope
from src.web_api.sessions import (
    STAGE_DONE,
    STAGE_HARVESTING,
    STAGE_IDLE,
    SessionStore,
)


class _CloseSpy:
    """Fake per-session resource that records whether close() ran."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RaisingClose:
    """Fake resource whose close() raises, to test failure isolation."""

    def close(self) -> None:
        raise RuntimeError("boom")


def _config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=3,
        embed_url="http://x",
        milvus_uri="http://x",
        collection="c1",
    )


def _scope() -> Scope:
    return Scope(
        name="Foo", description="d", milvus_collection="c1", postgres_table="c1"
    )


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    s = SessionStore(runs_dir=tmp_path, base_config=_config())
    s._make_logging_hook = lambda session_id, user_runs: None  # type: ignore[assignment]
    s._extractor_factory = lambda sid: None  # type: ignore[assignment]
    return s


def test_evict_idle_drops_idle_context_past_threshold(
    store: SessionStore,
) -> None:
    ctx = store.create(scope=_scope(), user_id="alice")
    sid = ctx.state.session_id
    # Backdate the activity so the threshold trips.
    ctx.last_activity_ts = time.monotonic() - 600.0
    assert ctx.progress.snapshot()["stage"] == STAGE_IDLE

    evicted = store.evict_idle(threshold_seconds=30.0)
    assert evicted == [sid]
    assert store.get(sid) is None


def test_evict_idle_skips_recent_context(store: SessionStore) -> None:
    ctx = store.create(scope=_scope(), user_id="alice")
    sid = ctx.state.session_id
    # Default last_activity_ts is "now" — well within the threshold.
    assert store.evict_idle(threshold_seconds=30.0) == []
    assert store.get(sid) is not None


def test_evict_idle_skips_non_idle_stage_even_when_aged_out(
    store: SessionStore,
) -> None:
    """A long-running harvest worker leaves stage != IDLE for the
    duration. The sweep must skip such contexts to avoid orphaning the
    daemon."""
    ctx = store.create(scope=_scope(), user_id="alice")
    sid = ctx.state.session_id
    ctx.progress.set_stage(STAGE_HARVESTING)
    ctx.last_activity_ts = time.monotonic() - 3600.0  # an hour ago

    assert store.evict_idle(threshold_seconds=30.0) == []
    assert store.get(sid) is not None


def test_evict_idle_handles_empty_store(store: SessionStore) -> None:
    assert store.evict_idle(threshold_seconds=30.0) == []


def test_touch_resets_last_activity(store: SessionStore) -> None:
    ctx = store.create(scope=_scope(), user_id="alice")
    sid = ctx.state.session_id
    ctx.last_activity_ts = time.monotonic() - 1000.0

    store.touch(sid)
    # touch resets to "now"; threshold of 1s should not fire.
    assert store.evict_idle(threshold_seconds=1.0) == []
    assert store.get(sid) is not None


def test_attach_seeds_last_activity_so_freshly_resumed_session_isnt_evicted(
    store: SessionStore,
) -> None:
    state = SessionState(current_config=_config())
    ctx = store.attach(state, user_id="alice")
    # The factory default for last_activity_ts is "now", so a sweep
    # immediately after attach must not drop the freshly-resumed
    # context.
    assert store.evict_idle(threshold_seconds=0.001) == [] != [ctx.state.session_id]


def test_evict_idle_drops_done_context_past_threshold(
    store: SessionStore,
) -> None:
    """A terminal (DONE) session is re-hydratable from disk, so the
    sweep must reclaim it just like an idle one (issue #60)."""
    ctx = store.create(scope=_scope(), user_id="alice")
    sid = ctx.state.session_id
    ctx.progress.set_stage(STAGE_DONE)
    ctx.last_activity_ts = time.monotonic() - 600.0

    evicted = store.evict_idle(threshold_seconds=30.0)
    assert evicted == [sid]
    assert store.get(sid) is None


def test_evict_calls_close_on_resources(store: SessionStore) -> None:
    """Eviction must release the per-session extractor + reflection
    hook, not just drop the dict entry (the thread/client leak)."""
    ctx = store.create(scope=_scope(), user_id="alice")
    ctx.extractor = _CloseSpy()  # type: ignore[assignment]
    ctx.reflection_hook = _CloseSpy()  # type: ignore[assignment]
    ctx.progress.set_stage(STAGE_DONE)
    ctx.last_activity_ts = time.monotonic() - 600.0

    store.evict_idle(threshold_seconds=30.0)
    assert ctx.extractor.closed  # type: ignore[union-attr]
    assert ctx.reflection_hook.closed  # type: ignore[union-attr]


def test_evict_close_failure_in_reflection_hook_isolated(
    store: SessionStore,
) -> None:
    """A failing reflection-hook close must not block the extractor
    close, nor stop the session from being evicted."""
    ctx = store.create(scope=_scope(), user_id="alice")
    spy = _CloseSpy()
    ctx.extractor = spy  # type: ignore[assignment]
    ctx.reflection_hook = _RaisingClose()  # type: ignore[assignment]
    ctx.progress.set_stage(STAGE_DONE)
    ctx.last_activity_ts = time.monotonic() - 600.0

    assert store.evict_idle(threshold_seconds=30.0) == [ctx.state.session_id]
    assert spy.closed


def test_evict_close_failure_in_extractor_isolated(
    store: SessionStore,
) -> None:
    """Isolation holds in the other order too: a failing extractor
    close must not block the reflection-hook close."""
    ctx = store.create(scope=_scope(), user_id="alice")
    spy = _CloseSpy()
    ctx.extractor = _RaisingClose()  # type: ignore[assignment]
    ctx.reflection_hook = spy  # type: ignore[assignment]
    ctx.progress.set_stage(STAGE_DONE)
    ctx.last_activity_ts = time.monotonic() - 600.0

    assert store.evict_idle(threshold_seconds=30.0) == [ctx.state.session_id]
    assert spy.closed


def test_drop_calls_close(store: SessionStore) -> None:
    """drop() leaks the same resources as eviction did; it must close
    them too (covers discard(), which routes through drop())."""
    ctx = store.create(scope=_scope(), user_id="alice")
    spy = _CloseSpy()
    ctx.extractor = spy  # type: ignore[assignment]

    assert store.drop(ctx.state.session_id) is True
    assert spy.closed
