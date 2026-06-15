"""In-process session store for the web API.

Holds :class:`SessionState` plus optional reflection + logging hooks per
session id. The TUI's ``SessionState`` class is reused verbatim so the
rating/turn lifecycle stays in lockstep with the terminal UI.

Each context carries an owner (``user_id``) and an idle timestamp.
The store is mutated by the FastAPI request thread on
create/get/drop, and (optionally) by a background asyncio task that
calls :meth:`SessionStore.evict_idle` to drop contexts the operator
has stopped touching.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.anchor.config import load_harvest_config
from src.anchor.errors import AnchorError
from src.anchor.runner import AnchorResult
from src.config_loader import session_overrides_path
from src.extraction import DualSpanExtractor, ExtractionError, SpanExtractor
from src.extraction.cache import SpanCache
from src.extraction.extractor import _load_config as _load_extraction_config
from src.logging import ProgressLogWriter, setup_run_logging, teardown_run_logging
from src.reflection import ReflectionAgent, ReflectionError
from src.search.config import SearchConfig, load_default_config, with_overrides
from src.session.state import ConvergenceThresholds, SessionState
from src.scopes import Scope

log = logging.getLogger(__name__)

# An extractor instance exposes the duck-typed interface
# ``_TurnExtractSpan`` consumes (``extract``, ``derive_span_text``,
# ``model`` / ``prompt_version`` / ``cache_hits`` / ``latency_ms_total``
# / ``call_count``). Either the single-vendor ``SpanExtractor`` or the
# dual-vendor wrapper (issue #54) satisfies it.
SpanExtractorLike = SpanExtractor | DualSpanExtractor


# Stage keys reported via /progress. Frontend switches labels from these.
STAGE_IDLE = "idle"
STAGE_PROBING = "probing"
STAGE_REFLECTING = "reflecting"
STAGE_ADAPTING = "adapting"
STAGE_SEARCHING = "searching_milvus"
STAGE_RETRYING = "searching_retry"
STAGE_EXTRACTING = "extracting_spans"
# Phase 2 stage. ``run_anchor`` emits free-form progress strings rather
# than machine-readable sub-stages; we surface them via
# ``Progress.set_detail`` and keep this single stage for the duration.
STAGE_HARVESTING = "harvesting"
# Phase 3 stages. Derive is one LLM call; judge fans out and reports
# ``processed/total`` via the polling endpoint; finalize is the writer.
STAGE_REFINE_DERIVING = "refine_deriving"
STAGE_REFINE_JUDGING = "refine_judging"
STAGE_REFINE_FINALIZING = "refine_finalizing"
# Phase 4 stages. Training fits LR on the Phase 3 sample; preparing
# warms the cohort projection so the panel renders with data ready;
# applying writes the labels across the full cohort.
STAGE_APPLY_TRAINING = "apply_training"
STAGE_APPLY_PREPARING = "apply_preparing"
STAGE_APPLY_APPLYING = "apply_applying"
STAGE_DONE = "done"

# Stages the idle-eviction sweep may reclaim. ``STAGE_IDLE`` is a
# session at rest; ``STAGE_DONE`` is terminal — every endpoint sets it
# on completion and a DONE session is fully re-hydratable from disk via
# ``hydrate_done_view``, so dropping it loses no persisted state. All
# other stages mean a daemon worker is mid-flight; evicting one would
# orphan it, so they are deliberately excluded.
EVICTABLE_STAGES = frozenset({STAGE_IDLE, STAGE_DONE})


class Progress:
    """Thread-safe mutable progress snapshot for a single in-flight operation.

    Written from the worker thread running _bootstrap_turn / _advance_turn
    and read from the FastAPI event loop via GET /progress. A single lock
    keeps the snapshot consistent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stage: str = STAGE_IDLE
        self.processed: int = 0
        self.total: int | None = None
        self.error: str | None = None
        self.detail: str | None = None

    def reset(self) -> None:
        with self._lock:
            self.stage = STAGE_IDLE
            self.processed = 0
            self.total = None
            self.error = None
            self.detail = None

    def set_stage(self, stage: str, *, total: int | None = None) -> None:
        with self._lock:
            self.stage = stage
            self.processed = 0
            self.total = total
            # Stage transitions clear stale per-attempt detail. Callers
            # that want to retain a message across a stage change must
            # call ``set_detail`` afterwards.
            self.detail = None

    def set_processed(self, n: int) -> None:
        with self._lock:
            self.processed = n

    def set_detail(self, message: str | None) -> None:
        with self._lock:
            self.detail = message

    def set_error(self, message: str) -> None:
        with self._lock:
            self.error = message

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stage": self.stage,
                "processed": self.processed,
                "total": self.total,
                "error": self.error,
                "detail": self.detail,
            }


@dataclass
class SessionContext:
    """Per-session state + hooks + coarse lock to serialize mutations."""

    state: SessionState
    logging_hook: ProgressLogWriter | None
    reflection_hook: ReflectionAgent | None
    extractor: SpanExtractorLike | None
    lock: threading.Lock
    # Owner — the auth user_id whose ``runs/<user_id>/`` subdir holds
    # this session's sidecars. The ownership check in ``_require``
    # 403s any cross-user access.
    user_id: str = "dev"
    # ``True`` means every mutating endpoint must 409. Set on
    # ``DONE_VIEW`` resume so a finalised session re-loaded for
    # reference can't have its sidecars overwritten by accident.
    read_only: bool = False
    # Replay mode: re-walk a past session read-only from Phase 1.
    # Implies ``read_only=True``. The replay-specific fields below
    # carry the pre-reconstructed turn history + phase-2/3/4 views
    # the advance endpoint rotates through.
    replay: bool = False
    replay_turns: list = field(default_factory=list)
    replay_cursor: int = 0
    # Ordered abstract steps for this replay (e.g. ``["TUNING",
    # "REPLAY_HARVEST", "REPLAY_REFINE", "REPLAY_APPLY"]``). Trimmed at
    # hydrate time to only the steps whose sidecars exist on disk.
    replay_phase_sequence: list[str] = field(default_factory=list)
    replay_phase_index: int = 0
    replay_anchor_result: AnchorResult | None = None
    replay_refine_state: Any = None
    replay_apply_view: Any = None
    # Epoch-seconds timestamp of the last successful ``_require`` for
    # this session. The idle-eviction sweep compares against this.
    last_activity_ts: float = field(default_factory=time.monotonic)
    progress: Progress = field(default_factory=Progress)
    # Phase 2 result, set by the harvest worker on success. Used by
    # ``GET /harvest/result`` and to render the workflow timeline.
    # Stored on the web-only context (not on SessionState) so the TUI's
    # in-memory contract stays unchanged.
    anchor_result: AnchorResult | None = None

    def close(self) -> None:
        """Best-effort release of the OS resources this context owns.

        Shuts down the span extractor's ``ThreadPoolExecutor`` and
        closes the extractor's / reflection hook's OpenAI httpx pools.
        Each resource is closed in its own guarded block so one failure
        can't leak the others. Called by the eviction sweep and by
        ``SessionStore.drop`` after the context is unreachable.

        ``logging_hook`` is intentionally not closed here —
        ``teardown_run_logging`` owns that handler's lifecycle.
        """
        for resource in (self.extractor, self.reflection_hook):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "SessionContext.close: %r close raised: %s",
                        resource,
                        exc,
                    )


class SessionStore:
    """Thin in-process registry keyed by session id.

    Per-user filesystem layout: every session's sidecars live under
    ``runs/<user_id>/<sid>.*``. The store does not enforce ownership
    on its API surface — that's the route layer's job via
    ``_require()``. The store only routes filesystem access to the
    right subdir given a ``user_id``.
    """

    def __init__(
        self,
        *,
        runs_dir: Path | None = None,
        reflection_factory: Callable[[], ReflectionAgent | None] | None = None,
        extractor_factory: Callable[[str], SpanExtractorLike | None] | None = None,
        base_config: SearchConfig | None = None,
    ) -> None:
        # Base directory; per-user subdirs hang off it. Existing call
        # sites that pass a flat ``runs_dir`` keep working — the value
        # becomes the base.
        self._base_runs_dir = runs_dir or Path("runs")
        self._reflection_factory = reflection_factory or _default_reflection_factory
        self._extractor_factory = (
            extractor_factory
            if extractor_factory is not None
            # Per-session callers pass the ``user_id``-aware runs_dir
            # they want; default extractor is built lazily per
            # session via the runs_dir_for() resolver.
            else None
        )
        self._base_config = base_config
        self._sessions: dict[str, SessionContext] = {}
        self._registry_lock = threading.Lock()

    @property
    def base_runs_dir(self) -> Path:
        """Root of the per-user tree, e.g. ``runs/``."""
        return self._base_runs_dir

    @property
    def runs_dir(self) -> Path:
        """Backwards-compatible alias for the base runs directory.

        Pre-auth call sites that read ``store.runs_dir`` directly
        continue to work; new code should use
        :meth:`runs_dir_for` to resolve a scoped path.
        """
        return self._base_runs_dir

    def runs_dir_for(self, user_id: str) -> Path:
        """Return ``runs/<user_id>/``, creating the directory if needed."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        path = self._base_runs_dir / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def base_config(self) -> SearchConfig:
        if self._base_config is None:
            self._base_config = load_default_config()
        return self._base_config

    def _build_search_config(
        self, overrides: dict[str, dict[str, Any]] | None
    ) -> SearchConfig:
        """Build the starting search config, applying per-session overrides
        for the ``search:`` section when present.

        When a ``base_config`` was injected (tests or in-process callers
        that don't have a real ``config.yaml``), overrides are folded in
        with :func:`with_overrides`. Otherwise we re-run the YAML loader
        so the validator catches bad override values with the same
        error path as a bad YAML edit.
        """
        search_block = overrides.get("search") if overrides else None
        if not search_block:
            return self.base_config()
        if self._base_config is not None:
            normalized = dict(search_block)
            if "active_paths" in normalized:
                normalized["active_paths"] = frozenset(normalized["active_paths"])
            return with_overrides(self._base_config, **normalized)
        return load_default_config(session_overrides=search_block)

    def create(
        self,
        *,
        scope: Scope,
        user_id: str = "dev",
        session_id: str | None = None,
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> SessionContext:
        search_config = self._build_search_config(overrides)
        session_config = with_overrides(
            search_config, collection=scope.milvus_collection
        )
        convergence = _resolve_convergence_thresholds(overrides)
        ctor_kwargs: dict[str, Any] = {
            "current_config": session_config,
            "convergence": convergence,
        }
        if session_id is not None:
            ctor_kwargs["session_id"] = session_id
        state = SessionState(**ctor_kwargs)
        state.scope = scope.name
        user_runs = self.runs_dir_for(user_id)
        try:
            log_path = setup_run_logging(state.session_id, user_runs)
            log.info("Run log file: %s", log_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Run log file unavailable: %s", exc)
        # Drop a one-line ownership marker on disk so a sysadmin
        # eyeballing a single sidecar can answer "whose session is
        # this?" without consulting the in-memory state.
        try:
            (user_runs / f"{state.session_id}.owner").write_text(
                f"{user_id}\n", encoding="utf-8"
            )
        except OSError as exc:
            log.warning(
                "Could not write ownership marker for %s: %s",
                state.session_id,
                exc,
            )
        if overrides:
            ov_path = session_overrides_path(
                state.session_id, self._base_runs_dir, user_id
            )
            try:
                ov_path.write_text(
                    json.dumps(overrides, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                log.info("Wrote session overrides sidecar: %s", ov_path)
            except OSError as exc:
                log.warning(
                    "Could not write overrides sidecar for %s: %s",
                    state.session_id,
                    exc,
                )
        logging_hook = self._make_logging_hook(state.session_id, user_runs)
        reflection_hook = self._reflection_factory()
        extractor = self._build_extractor(state.session_id, user_runs)
        ctx = SessionContext(
            state=state,
            logging_hook=logging_hook,
            reflection_hook=reflection_hook,
            extractor=extractor,
            lock=threading.Lock(),
            user_id=user_id,
        )
        with self._registry_lock:
            self._sessions[state.session_id] = ctx
        log.info("Created session %s for user %s", state.session_id, user_id)
        return ctx

    def attach(
        self,
        state: SessionState,
        *,
        user_id: str,
        read_only: bool = False,
        replay: bool = False,
    ) -> SessionContext:
        """Register an externally-built ``SessionState`` under its
        existing session id, wiring up the same hooks as :meth:`create`.

        Used by ``POST /api/session/<sid>/resume`` after rehydrating
        the in-memory state from disk. The ``<sid>.owner`` marker
        already exists from the original ``create`` call; this method
        does not re-write it. Logging is reopened in append mode so
        post-resume turns extend the same per-session log.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        user_runs = self.runs_dir_for(user_id)
        try:
            log_path = setup_run_logging(state.session_id, user_runs)
            log.info("Run log file (resumed): %s", log_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Run log file unavailable on resume: %s", exc)
        logging_hook = self._make_logging_hook(state.session_id, user_runs)
        reflection_hook = self._reflection_factory()
        extractor = self._build_extractor(state.session_id, user_runs)
        ctx = SessionContext(
            state=state,
            logging_hook=logging_hook,
            reflection_hook=reflection_hook,
            extractor=extractor,
            lock=threading.Lock(),
            user_id=user_id,
            read_only=read_only,
            replay=replay,
        )
        with self._registry_lock:
            self._sessions[state.session_id] = ctx
        log.info(
            "Resumed session %s for user %s (read_only=%s replay=%s)",
            state.session_id,
            user_id,
            read_only,
            replay,
        )
        return ctx

    def get(self, session_id: str) -> SessionContext | None:
        with self._registry_lock:
            return self._sessions.get(session_id)

    def drop(self, session_id: str) -> bool:
        with self._registry_lock:
            ctx = self._sessions.pop(session_id, None)
        # Release the log handler and per-session OS resources (extractor
        # thread pool + OpenAI clients) once the context is unreachable.
        teardown_run_logging(session_id)
        if ctx is not None:
            ctx.close()
        return ctx is not None

    def discard(self, session_id: str, user_id: str) -> bool:
        """Hard-delete a session: evict the in-memory context and unlink
        every sidecar under the caller's runs dir.

        Returns True if the session existed in memory OR at least one
        on-disk sidecar was removed. Returns False if neither — the
        caller can map that to a 404.

        Ownership is enforced implicitly by ``runs_dir_for(user_id)``:
        we only ever touch files under the calling user's subdir, so a
        guessed sid that belongs to another user is a no-op.
        """
        # Local import to avoid a circular import with resume.py.
        from .resume import _SESSION_SUFFIXES

        removed_in_memory = self.drop(session_id)
        user_runs = self.runs_dir_for(user_id)
        removed_any_file = False
        for suffix in _SESSION_SUFFIXES:
            path = user_runs / f"{session_id}{suffix}"
            try:
                path.unlink()
                removed_any_file = True
            except FileNotFoundError:
                pass
        return removed_in_memory or removed_any_file

    def touch(self, session_id: str) -> None:
        """Update a session's idle clock.

        Called from the FastAPI ``_require`` helper on every successful
        lookup, so the eviction sweep only ages out contexts the
        operator has actually stopped touching.
        """
        with self._registry_lock:
            ctx = self._sessions.get(session_id)
            if ctx is not None:
                ctx.last_activity_ts = time.monotonic()

    def evict_idle(self, *, threshold_seconds: float) -> list[str]:
        """Drop in-memory contexts idle past ``threshold_seconds`` whose
        ``progress.stage`` is evictable (see ``EVICTABLE_STAGES``).

        Evictable stages are ``STAGE_IDLE`` (at rest) and ``STAGE_DONE``
        (terminal). Excluding every other stage is critical: a session
        in the middle of a long-running harvest or judge run has its
        stage set to ``STAGE_HARVESTING`` / ``STAGE_REFINE_*`` for the
        duration, and evicting one mid-flight would orphan the daemon
        worker. Disk state is canonical, so a session evicted while idle
        or done is re-hydrated lazily on the next
        ``GET /api/session/<sid>`` or ``POST /api/session/<sid>/resume``
        (terminal sessions via ``hydrate_done_view``), losing no
        persisted state.
        """
        now = time.monotonic()
        with self._registry_lock:
            stale_ctxs = {
                sid: ctx
                for sid, ctx in self._sessions.items()
                if ctx.progress.snapshot().get("stage") in EVICTABLE_STAGES
                and (now - ctx.last_activity_ts) >= threshold_seconds
            }
            for sid in stale_ctxs:
                self._sessions.pop(sid, None)
        # Release per-session resources OUTSIDE the registry lock:
        # teardown_run_logging closes the log handler; ctx.close() shuts
        # down the extractor thread pool and OpenAI clients. Both are
        # safe now the context is unreachable via get()/touch().
        for sid, ctx in stale_ctxs.items():
            teardown_run_logging(sid)
            ctx.close()
        stale = list(stale_ctxs)
        if stale:
            log.info("Idle-evicted %d session(s): %s", len(stale), stale)
        return stale

    def find_owner_on_disk(self, session_id: str) -> str | None:
        """Walk per-user subdirs looking for a session of this id.

        Used by ``_require`` to distinguish a 404 (truly unknown
        session) from a 403 (session exists but belongs to another
        user). Reads only directory listings — no canonical jsonl
        parsing — so it stays fast even with many users.
        """
        if not self._base_runs_dir.exists():
            return None
        for user_dir in self._base_runs_dir.iterdir():
            if not user_dir.is_dir():
                continue
            if (user_dir / f"{session_id}.jsonl").exists() or (
                user_dir / f"{session_id}.owner"
            ).exists():
                return user_dir.name
        return None

    def _make_logging_hook(
        self, session_id: str, user_runs: Path
    ) -> ProgressLogWriter | None:
        try:
            return ProgressLogWriter(session_id, user_runs)
        except Exception as exc:  # noqa: BLE001
            log.warning("ProgressLogWriter unavailable: %s", exc)
            return None

    def _build_extractor(
        self, session_id: str, user_runs: Path
    ) -> SpanExtractorLike | None:
        if self._extractor_factory is not None:
            return self._extractor_factory(session_id)
        return _default_extractor_factory(user_runs, session_id)


def _resolve_convergence_thresholds(
    overrides: dict[str, dict[str, Any]] | None,
) -> ConvergenceThresholds:
    """Snapshot the Phase 1 convergence triple at session creation.

    Loads ``harvest.{min_fit, min_not_fit, precision_at_k}`` from YAML
    with the operator's ``[Edit parameters]`` overrides applied, so the
    Convergence panel and ``is_converged`` predicate both reflect the
    knobs the session actually started under. Falls back to the module
    defaults if the harvest section fails to validate — session
    creation shouldn't brick on a malformed YAML knob; ``load_harvest_config``
    will surface the real error at Phase 2 preflight.
    """
    harvest_block = overrides.get("harvest") if overrides else None
    try:
        cfg = load_harvest_config(session_overrides=harvest_block)
    except AnchorError as exc:
        log.warning(
            "Falling back to module convergence defaults; harvest config "
            "failed to load: %s",
            exc,
        )
        return ConvergenceThresholds.from_module_defaults()
    return ConvergenceThresholds(
        min_fit=cfg.min_fit,
        min_not_fit=cfg.min_not_fit,
        precision_at_k=cfg.precision_at_k,
    )


def _default_reflection_factory() -> ReflectionAgent | None:
    try:
        return ReflectionAgent()
    except ReflectionError as exc:
        log.warning("Reflection agent unavailable: %s", exc)
        return None


def _default_extractor_factory(
    runs_dir: Path, session_id: str
) -> SpanExtractorLike | None:
    """Construct a span extractor with a per-session SpanCache.

    When the extraction config carries the optional ``secondary_*``
    keys (issue #54), returns a :class:`DualSpanExtractor` wrapping
    a primary and secondary :class:`SpanExtractor` sharing the same
    cache. Otherwise returns a single :class:`SpanExtractor`.

    Degrades gracefully: any failure (missing API key, config error)
    returns ``None`` so the session still works without highlighted
    spans. A secondary-vendor construction failure logs a warning but
    falls back to the primary alone rather than aborting the session.
    """
    try:
        cache = SpanCache(runs_dir / f"{session_id}.span_cache.jsonl")
        primary = SpanExtractor(cache=cache)
    except ExtractionError as exc:
        log.warning("Span extractor unavailable: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Span extractor construction failed: %s", exc)
        return None

    secondary = _try_build_secondary(cache=cache)
    if secondary is None:
        log.info(
            "Span extractor ready: model=%s prompt_version=%s",
            primary.model,
            primary.prompt_version,
        )
        return primary

    try:
        dual = DualSpanExtractor(primary, secondary)
    except ExtractionError as exc:
        log.warning(
            "Dual span extractor construction failed (%s); "
            "falling back to primary alone.",
            exc,
        )
        return primary
    log.info(
        "Dual span extractor ready: model=%s prompt_version=%s",
        dual.model,
        dual.prompt_version,
    )
    return dual


def _try_build_secondary(*, cache: SpanCache) -> SpanExtractor | None:
    """Build the secondary :class:`SpanExtractor` if the optional
    config keys are set. Returns ``None`` when secondary extraction
    isn't configured. Any construction failure (e.g. missing API
    key in env) logs a warning and returns ``None`` — the caller
    falls back to primary-only extraction.
    """
    try:
        cfg = _load_extraction_config()
    except ExtractionError as exc:
        log.warning("Secondary span extractor: config load failed: %s", exc)
        return None
    if not cfg.has_secondary:
        return None
    try:
        return SpanExtractor(
            cache=cache,
            model_override=cfg.secondary_model,
            base_url_override=cfg.secondary_base_url,
            api_key_env_override=cfg.secondary_api_key_env,
        )
    except ExtractionError as exc:
        log.warning(
            "Secondary span extractor unavailable (%s); running primary alone.",
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Secondary span extractor construction failed: %s", exc)
        return None
