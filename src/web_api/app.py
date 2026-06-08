"""FastAPI app exposing the Deka session loop over HTTP.

Endpoints mirror the TUI's keyboard actions:

- ``POST /api/session`` — enter query, get first turn's candidates.
- ``POST /api/session/{sid}/rate`` — rate a fused row or per-path candidate.
- ``POST /api/session/{sid}/turn/next`` — complete turn, reflect, advance.
- ``GET /api/session/{sid}/reflection`` — last turn's reflection output.
- ``DELETE /api/session/{sid}`` — end session.

All long-running operations (search, reflection) are offloaded to a
thread via :func:`asyncio.to_thread` so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import secrets
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from src.extraction import build_extract_span
from src.logging import flush_run_logging
from src.postgres import (
    OriginalContentFetcher,
    PostgresFetchError,
    load_postgres_config,
)
from src.search.adapt import AdaptError, adapt_config
from src.search.config import with_overrides
from src.search.errors import ConfigError, MilvusSearchError
from src.search.evidence import Rating
from src.search.search import probe_only, run_search
from src.auth import User, UserRegistry, load_users
from src.auth.cookies import COOKIE_NAME, session_payload
from src.auth.dependencies import current_user
from src.scopes import (
    ScopeError,
    ScopeRegistry,
    load_scopes,
)

from . import serialize
from .preflight import (
    PreflightContext,
    first_failure,
    run_preflight,
)
from .resume import (
    ResumeError,
    ResumeNotImplementedError,
    ResumeTarget,
    advance_replay,
    classify,
    hydrate_done_view,
    hydrate_post_harvest,
    hydrate_post_rubric,
    hydrate_post_tuning,
    hydrate_replay,
    list_user_sessions,
)
from .schemas import (
    AnchorResultDTO,
    ApplyCalibrateResponse,
    ApplyFinalizeRequest,
    ApplyPreflightDTO,
    ApplySummaryDTO,
    AuthLoginRequest,
    AuthMeResponse,
    DeriveResultDTO,
    DropPathRequest,
    HarvestPreflightDTO,
    HarvestRunRequest,
    JudgeResultDTO,
    NextTurnResponse,
    OkResponse,
    OriginalContentResponse,
    PreflightCheckDTO,
    PreflightRequest,
    PreflightResponse,
    ProgressDTO,
    RateRequest,
    RecommendationDecisionRequest,
    ReflectionDTO,
    RefinePreflightDTO,
    RefineSummaryDTO,
    RubricMetadataDTO,
    RubricPromptDTO,
    RubricSaveRequest,
    SessionListItem,
    SessionSnapshot,
    StartSessionRequest,
    UpdateConfigRequest,
    ScopeDTO,
    ScopesResponse,
    VerdictDTO,
)
from .sessions import (
    STAGE_ADAPTING,
    STAGE_APPLY_APPLYING,
    STAGE_APPLY_PREPARING,
    STAGE_APPLY_TRAINING,
    STAGE_DONE,
    STAGE_HARVESTING,
    STAGE_PROBING,
    STAGE_REFINE_DERIVING,
    STAGE_REFINE_FINALIZING,
    STAGE_REFINE_JUDGING,
    STAGE_REFLECTING,
    SessionContext,
    SessionStore,
)

log = logging.getLogger(__name__)


def _snap(ctx: SessionContext) -> SessionSnapshot:
    """Serialize the session state with the harvest result threaded in
    so the workflow timeline reflects Phase 2 progress, and the
    ``read_only`` / ``replay`` flags forwarded to the frontend.
    """
    return serialize.snapshot(
        ctx.state,
        anchor_result=ctx.anchor_result,
        read_only=ctx.read_only,
        replay=ctx.replay,
    )


class ErrorResponse(BaseModel):
    detail: str


def create_app(
    store: SessionStore | None = None,
    *,
    pg_fetcher: OriginalContentFetcher | None = None,
    pg_disabled_by_admin: bool = False,
    scopes: ScopeRegistry | None = None,
    users: UserRegistry | None = None,
    session_secret: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Idle-eviction sweep. Threshold from
        # ``DEKA_SESSION_IDLE_MINUTES`` (default 30). Sweeps every
        # 60s; cancel-aware so a clean shutdown exits promptly.
        idle_minutes_env = os.environ.get("DEKA_SESSION_IDLE_MINUTES")
        try:
            idle_minutes = float(idle_minutes_env) if idle_minutes_env else 30.0
        except ValueError:
            log.warning(
                "Bad DEKA_SESSION_IDLE_MINUTES=%r — falling back to 30",
                idle_minutes_env,
            )
            idle_minutes = 30.0
        threshold_seconds = idle_minutes * 60.0

        async def _sweep():
            while True:
                try:
                    await asyncio.sleep(60.0)
                except asyncio.CancelledError:
                    break
                try:
                    app.state.store.evict_idle(threshold_seconds=threshold_seconds)
                except Exception as exc:  # noqa: BLE001
                    log.warning("idle-eviction sweep failed: %s", exc)

        sweep_task = asyncio.create_task(_sweep(), name="session-eviction-sweep")
        try:
            yield
        finally:
            sweep_task.cancel()
            try:
                await sweep_task
            except (asyncio.CancelledError, Exception):
                pass
            fetcher: OriginalContentFetcher | None = app.state.pg_fetcher
            if fetcher is not None:
                fetcher.close()
            scoped_cache: dict[str, OriginalContentFetcher] | None = getattr(
                app.state, "pg_fetcher_by_table", None
            )
            if scoped_cache:
                for scoped in scoped_cache.values():
                    try:
                        scoped.close()
                    except Exception as exc:  # noqa: BLE001
                        log.debug("pg fetcher close (scoped) raised: %s", exc)
                scoped_cache.clear()

    app = FastAPI(
        title="Deka Web API",
        description="HTTP surface for the Deka search-tuning session loop.",
        version="0.1.0",
        lifespan=lifespan,
    )
    # CORS first so browser preflight succeeds; SessionMiddleware next
    # so request.session is populated for every other handler.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    secret = session_secret or os.environ.get("DEKA_SESSION_SECRET")
    if secret is None:
        # Allow tests / fresh dev shells to boot without a configured
        # secret by falling back to a per-process random key. Cookies
        # signed under the random key are invalidated on every server
        # restart, so production deployments MUST set the env var.
        secret = secrets.token_urlsafe(32)
        log.warning(
            "DEKA_SESSION_SECRET unset; signing cookies with a "
            "per-process random key. Set the env var in any non-dev "
            "deployment so cookies survive restarts."
        )
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret,
        session_cookie=COOKIE_NAME,
        same_site="lax",
        https_only=False,  # operators flip via deployment proxy / env
        max_age=30 * 24 * 3600,  # 30 days
    )

    app.state.scopes = scopes or load_scopes()
    app.state.user_registry = users or load_users(scope_registry=app.state.scopes)
    app.state.store = store or SessionStore()
    app.state.pg_fetcher = pg_fetcher
    app.state.pg_fetcher_by_table = {}
    app.state.pg_disabled_by_admin = pg_disabled_by_admin

    def _store() -> SessionStore:
        return app.state.store

    def _pg_fetcher() -> OriginalContentFetcher:
        if app.state.pg_disabled_by_admin:
            raise HTTPException(
                status_code=403,
                detail="Context expansion is disabled by server configuration",
            )
        fetcher: OriginalContentFetcher | None = app.state.pg_fetcher
        if fetcher is None:
            raise HTTPException(
                status_code=503,
                detail="Postgres lookup is not configured on this server",
            )
        return fetcher

    def _resolve_pg_table_override(ctx: SessionContext) -> str | None:
        """Return the scope's ``postgres_table``, or ``None``.

        Reads the active scope name from ``ctx.state.scope`` and
        looks it up in ``app.state.scopes``. Returns ``None`` when
        no scope is bound or the registry rejects the name — callers
        should treat that as an error (there is no global default).
        """
        scope_name = ctx.state.scope
        if not scope_name:
            return None
        registry: ScopeRegistry | None = getattr(app.state, "scopes", None)
        if registry is None:
            return None
        try:
            scope = registry.get(scope_name)
        except Exception:  # noqa: BLE001 — unknown scope name
            return None
        return scope.postgres_table

    def _pg_fetcher_for_scope(scope_name: str | None) -> OriginalContentFetcher:
        """Return a fetcher whose table matches ``scope_name``.

        Raises 422 when no scope name is supplied or the scope is
        unknown to the registry — the global ``postgres.table`` was
        dropped, so a scope is required for any chunk-content lookup.
        Per-table fetchers are cached on ``app.state.pg_fetcher_by_table``
        and closed at lifespan shutdown.
        """
        _pg_fetcher()  # honours admin-disabled + unconfigured guards
        override: str | None = None
        if scope_name:
            registry: ScopeRegistry | None = getattr(app.state, "scopes", None)
            if registry is not None:
                try:
                    override = registry.get(scope_name).postgres_table
                except Exception:  # noqa: BLE001 — unknown scope name
                    override = None
        if not override:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Cannot resolve Postgres table for this session — "
                    "no scope bound, or the scope is missing a "
                    "'postgres_table' declaration in scopes.yaml."
                ),
            )
        from src.postgres.config import load_postgres_config
        from src.postgres.fetch import OriginalContentFetcher as _Fetcher

        cache: dict[str, OriginalContentFetcher] = (
            getattr(app.state, "pg_fetcher_by_table", None) or {}
        )
        if override in cache:
            return cache[override]
        from dataclasses import replace

        cfg = replace(load_postgres_config(), table=override)
        scoped = _Fetcher(cfg)
        cache[override] = scoped
        app.state.pg_fetcher_by_table = cache
        return scoped

    def _pg_fetcher_for_ctx(ctx: SessionContext) -> OriginalContentFetcher:
        """Return a fetcher whose table matches the session's bound scope."""
        return _pg_fetcher_for_scope(ctx.state.scope)

    def _require_session_files(session_id: str, user_id: str) -> Path:
        """Lightweight ownership check for read-only file downloads.

        Unlike :func:`_require`, this resolves directly against
        ``runs/<user>/`` so it works on cold sessions that haven't been
        hydrated into memory yet (e.g. clicked from the listing page).
        Returns the caller's per-user runs dir on success.
        """
        owner = _store().find_owner_on_disk(session_id)
        if owner is None:
            raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")
        if owner != user_id:
            raise HTTPException(status_code=403, detail="Not your session")
        return _store().runs_dir_for(user_id)

    def _require_llm_key(*, section: str, phase_label: str) -> None:
        """Raise 422 with the pre-flight-shaped typed body when the LLM
        key for ``section`` is missing.

        Mirrors the pre-flight check (``_check_llm_key`` in
        :mod:`src.web_api.preflight`) and is called synchronously *before*
        the refine worker thread spawns so a missing key surfaces as a
        4xx the UI can display, not a silent 200 + background WARNING
        (issue #33's "Related cleanup").
        """
        from src.config_loader import ConfigFileError, load_section

        try:
            raw = load_section(section)
        except ConfigFileError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "LLM_CONFIG_MISSING",
                    "phase": f"llm.{section}",
                    "env_var": "",
                    "detail": f"{section} config unreadable: {exc}",
                },
            )
        env_var = raw.get("api_key_env")
        if not isinstance(env_var, str) or not env_var.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "LLM_CONFIG_INVALID",
                    "phase": f"llm.{section}",
                    "env_var": "",
                    "detail": f"{section}.api_key_env is not configured.",
                },
            )
        if not os.environ.get(env_var):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "MISSING_LLM_KEY",
                    "phase": f"llm.{section}",
                    "env_var": env_var,
                    "detail": f"{phase_label} needs {env_var} in env or .env.",
                },
            )

    def _require_writable(ctx: SessionContext) -> None:
        """Reject mutations on a read-only context.

        Raised at the top of every mutating endpoint. The 409 status
        signals "the resource exists but its state forbids the
        operation" — the same semantics the harness already uses for
        "session has no active turn — call /turn/next" and similar
        state-conflict cases.
        """
        if ctx.read_only:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Session is read-only (DONE_VIEW). Re-running a "
                    "finalised session would overwrite the shipped "
                    "artifacts; start a new session instead."
                ),
            )

    def _require(session_id: str, user_id: str) -> SessionContext:
        """Fetch a session context owned by ``user_id`` or raise.

        - 404 if no session of that id exists in memory or on disk.
        - 403 if the session id exists (in memory or on disk) but is
          owned by another user.
        - Returns the in-memory context otherwise.

        Touches the idle clock on every successful lookup so the
        eviction sweep only fires on truly-quiet sessions.
        """
        ctx = _store().get(session_id)
        if ctx is not None:
            if ctx.user_id != user_id:
                raise HTTPException(status_code=403, detail="Not your session")
            _store().touch(session_id)
            return ctx
        owner = _store().find_owner_on_disk(session_id)
        if owner is not None and owner != user_id:
            raise HTTPException(status_code=403, detail="Not your session")
        # Either truly unknown, or owned by ``user_id`` but evicted /
        # not yet hydrated. POST /resume is the supported re-entry
        # path; everything else 404s.
        raise HTTPException(status_code=404, detail=f"Unknown session {session_id}")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    # ----- Auth ----- (no current_user dep — these routes establish it.)

    @app.post("/api/auth/login", response_model=AuthMeResponse)
    def auth_login(payload: AuthLoginRequest, request: Request) -> AuthMeResponse:
        sha = hashlib.sha256(payload.token.encode("utf-8")).hexdigest()
        registry: UserRegistry = app.state.user_registry
        user = registry.find_by_token_sha(sha)
        if user is None:
            raise HTTPException(status_code=401, detail="Unknown token")
        request.session.update(session_payload(user.id))
        return AuthMeResponse(user_id=user.id)

    @app.post("/api/auth/logout", status_code=204)
    def auth_logout(request: Request) -> Response:
        request.session.clear()
        return Response(status_code=204)

    @app.get("/api/auth/me", response_model=AuthMeResponse)
    def auth_me(user: User = Depends(current_user)) -> AuthMeResponse:
        return AuthMeResponse(user_id=user.id)

    @app.post(
        "/api/session/preflight",
        response_model=PreflightResponse,
    )
    async def preflight(
        body: PreflightRequest,
        user: User = Depends(current_user),
    ) -> PreflightResponse:
        """Run env / config sanity checks before a session is created.

        Issue #33 — fail fast. If any required check fails, return 400
        with the first failing check's typed fields (``code``, ``phase``,
        ``env_var``, ``detail``) at the top level plus the full result
        list under ``checks`` so the UI can render the staggered list
        with the failing item highlighted in context.

        Auth-required so unauthenticated probes can't enumerate the
        backend's config / reachability surface.
        """
        registry: ScopeRegistry = app.state.scopes
        try:
            scope = registry.get(body.scope)
        except ScopeError:
            raise HTTPException(
                status_code=422,
                detail=(f"unknown scope {body.scope!r}; available: {registry.names()}"),
            )
        if user.allowed_scopes is not None and scope.name not in user.allowed_scopes:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"scope {scope.name!r} is not in your allowlist; "
                    f"available: {sorted(user.allowed_scopes)}"
                ),
            )

        ctx = PreflightContext(
            scope_name=scope.name,
            users_registry=app.state.user_registry,
            scopes_registry=registry,
        )
        results = await asyncio.to_thread(run_preflight, ctx)
        check_dtos = [
            PreflightCheckDTO(
                name=r.name,
                status=r.status,  # type: ignore[arg-type]
                detail=r.detail,
                code=r.code,
                env_var=r.env_var,
            )
            for r in results
        ]
        failure = first_failure(results)
        if failure is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": failure.code,
                    "phase": failure.name,
                    "env_var": failure.env_var,
                    "detail": failure.detail,
                    "checks": [c.model_dump() for c in check_dtos],
                },
            )
        return PreflightResponse(checks=check_dtos, all_passed=True)

    @app.post("/api/session/{session_id}/resume", response_model=SessionSnapshot)
    async def resume_session(
        session_id: str,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        """Hydrate a previously-quit session and return its snapshot.

        Idempotent: if the session is already loaded in memory and
        owned by the calling user, this just returns the current
        snapshot. Otherwise the classifier picks the resume target
        and the matching hydrator rebuilds the in-memory context.

        Per-target behaviour:

        - ``POST_TUNING`` — full hydration via the replay loader.
        - ``POST_HARVEST`` — replay loader + Phase 2 sidecar
          reconstruction; lands in ``ANCHOR_DONE`` ready for refine.
        - ``POST_RUBRIC`` — adds rubric + judge verdicts; lands in
          ``REFINE_REVIEW`` for verdict decisions.
        - ``DONE_VIEW`` — adds the finalise sidecar and arms
          ``read_only=True`` so mutating endpoints 409.
        """
        existing = _store().get(session_id)
        if existing is not None:
            if existing.user_id != user.id:
                raise HTTPException(status_code=403, detail="Not your session")
            return _snap(existing)

        user_runs = _store().runs_dir_for(user.id)
        target = classify(session_id, user_runs)
        if target is None:
            # Distinguish a truly-absent session from one owned by
            # another user (which find_owner_on_disk catches below).
            owner = _store().find_owner_on_disk(session_id)
            if owner is not None and owner != user.id:
                raise HTTPException(status_code=403, detail="Not your session")
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No resumable state for session {session_id} "
                    "(abandoned or unknown)"
                ),
            )

        hydrators = {
            ResumeTarget.POST_TUNING: hydrate_post_tuning,
            ResumeTarget.POST_HARVEST: hydrate_post_harvest,
            ResumeTarget.POST_RUBRIC: hydrate_post_rubric,
            ResumeTarget.APPLY_PENDING: hydrate_done_view,
            ResumeTarget.DONE_VIEW: hydrate_done_view,
        }
        hydrate = hydrators.get(target)
        if hydrate is None:
            raise ResumeNotImplementedError(
                f"No hydrator registered for resume target {target.value}"
            )
        try:
            ctx = hydrate(
                session_id,
                user_runs_dir=user_runs,
                store=_store(),
                user_id=user.id,
            )
        except ResumeNotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except ResumeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return _snap(ctx)

    @app.post(
        "/api/session/{session_id}/replay",
        response_model=SessionSnapshot,
    )
    async def replay_start(
        session_id: str,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        """Open a past session in Replay Mode.

        Always lands at Phase 1, turn 1 with ``read_only=True`` and
        ``replay=True``. Subsequent ``/replay/advance`` calls walk the
        user forward through whichever phases the original session
        reached, surfacing the post-finalize summary view for each.

        If the session is already attached in memory for the calling
        user, the existing context is dropped first so replay starts
        clean from Phase 1 — otherwise a resumed live session and the
        replay would fight over the same context.
        """
        existing = _store().get(session_id)
        if existing is not None:
            if existing.user_id != user.id:
                raise HTTPException(status_code=403, detail="Not your session")
            _store().drop(session_id)

        user_runs = _store().runs_dir_for(user.id)
        canonical = user_runs / f"{session_id}.jsonl"
        if not canonical.exists():
            owner = _store().find_owner_on_disk(session_id)
            if owner is not None and owner != user.id:
                raise HTTPException(status_code=403, detail="Not your session")
            raise HTTPException(
                status_code=404,
                detail=f"No replayable state for session {session_id}",
            )

        try:
            ctx = hydrate_replay(
                session_id,
                user_runs_dir=user_runs,
                store=_store(),
                user_id=user.id,
            )
        except ResumeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return _snap(ctx)

    @app.post(
        "/api/session/{session_id}/replay/advance",
        response_model=SessionSnapshot,
    )
    async def replay_advance(
        session_id: str,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        """Step a replay forward by one section.

        Within Phase 1: rotates ``current_table`` / ``current_config``
        to the next historical turn. At Phase 1's last turn (and at
        every subsequent step) advances to the next replay phase
        (HARVEST → REFINE → APPLY → DONE), skipping any phase whose
        sidecars don't exist on disk. Idempotent at the terminal step.

        The 3-second "in replay-mode..." overlay is rendered
        client-side; this endpoint returns immediately so the client
        races its own timer.
        """
        ctx = _require(session_id, user.id)
        if not ctx.replay:
            raise HTTPException(
                status_code=409,
                detail="Session is not in Replay Mode; use /replay to start one.",
            )
        with ctx.lock:
            advance_replay(ctx)
        return _snap(ctx)

    @app.get("/api/sessions", response_model=list[SessionListItem])
    def list_sessions(
        user: User = Depends(current_user),
    ) -> list[SessionListItem]:
        """List the calling user's resumable sessions.

        Walks ``runs/<user_id>/`` and classifies each session via the
        disk classifier; abandoned sessions (no convergence marker
        and no phase 2/3 sidecars) are hidden. Sorted by most-recent
        modification descending so a returning user lands on what
        they were last working on.
        """
        user_runs = _store().runs_dir_for(user.id)
        return [
            SessionListItem(
                session_id=entry.session_id,
                query=entry.query,
                scope=entry.scope,
                resume_target=entry.resume_target.value,
                last_modified=entry.last_modified,
                n_turns=entry.n_turns,
                has_rubric=(
                    user_runs / f"{entry.session_id}.phase3.prompt.md"
                ).exists(),
                has_artifacts=(
                    user_runs / f"{entry.session_id}.phase4.labels.jsonl"
                ).exists(),
            )
            for entry in list_user_sessions(user_runs)
        ]

    @app.get("/api/config/defaults")
    def get_config_defaults(
        user: User = Depends(current_user),
    ) -> dict[str, dict[str, Any]]:
        """Return the curated-essentials projection of current YAML defaults.

        Used by the query page's [Edit parameters] modal to pre-fill
        placeholders. Restricted to the override allow-list, so this
        endpoint never leaks fixed/infra values (URIs, model ids, API
        key env vars, etc.).
        """
        return _config_defaults_projection()

    @app.get("/api/scopes", response_model=ScopesResponse)
    def list_scopes(
        user: User = Depends(current_user),
    ) -> ScopesResponse:
        registry: ScopeRegistry = app.state.scopes
        scopes = list(registry)
        if user.allowed_scopes is not None:
            allowed = set(user.allowed_scopes)
            scopes = [s for s in scopes if s.name in allowed]
        return ScopesResponse(
            scopes=[
                ScopeDTO(
                    name=s.name,
                    description=s.description,
                    milvus_collection=s.milvus_collection,
                )
                for s in scopes
            ]
        )

    @app.post("/api/session", response_model=SessionSnapshot)
    async def start_session(
        body: StartSessionRequest,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        query = body.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must not be empty")
        registry: ScopeRegistry = app.state.scopes
        try:
            scope = registry.get(body.scope)
        except ScopeError:
            raise HTTPException(
                status_code=422,
                detail=(f"unknown scope {body.scope!r}; available: {registry.names()}"),
            )
        if user.allowed_scopes is not None and scope.name not in user.allowed_scopes:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"scope {scope.name!r} is not in your allowlist; "
                    f"available: {sorted(user.allowed_scopes)}"
                ),
            )
        if body.session_id is not None and _store().get(body.session_id) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Session {body.session_id} already exists",
            )

        overrides_sidecar = (
            body.overrides.to_sidecar_dict() if body.overrides is not None else None
        )
        try:
            ctx = _store().create(
                session_id=body.session_id,
                scope=scope,
                user_id=user.id,
                overrides=overrides_sidecar or None,
            )
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        try:
            await asyncio.to_thread(_bootstrap_turn, ctx, query)
        except AdaptError as exc:
            ctx.progress.set_error(str(exc))
            _store().drop(ctx.state.session_id)
            raise HTTPException(status_code=422, detail=str(exc))
        except MilvusSearchError as exc:
            ctx.progress.set_error(str(exc))
            _store().drop(ctx.state.session_id)
            raise HTTPException(status_code=503, detail=str(exc))
        return _snap(ctx)

    @app.get("/api/session/{session_id}/progress", response_model=ProgressDTO)
    async def get_progress(
        session_id: str, user: User = Depends(current_user)
    ) -> ProgressDTO:
        ctx = _require(session_id, user.id)
        return ProgressDTO(**ctx.progress.snapshot())

    @app.get("/api/session/{session_id}", response_model=SessionSnapshot)
    async def get_session(
        session_id: str, user: User = Depends(current_user)
    ) -> SessionSnapshot:
        ctx = _require(session_id, user.id)
        if ctx.state.current_table is None and not ctx.state.is_converged:
            raise HTTPException(
                status_code=409,
                detail="Session has no active turn — call /turn/next",
            )
        return _snap(ctx)

    @app.get("/api/session/{session_id}/overrides")
    async def get_session_overrides(
        session_id: str, user: User = Depends(current_user)
    ) -> dict[str, dict[str, Any]]:
        """Return the session's overrides sidecar as a dict.

        Empty dict when the sidecar is absent (the common case — most
        sessions run with YAML defaults). Powers the sidebar phase
        panels that show effective config = defaults shadowed by these
        overrides.
        """
        ctx = _require(session_id, user.id)
        return _load_overrides_from_disk(
            ctx.state.session_id, _store().runs_dir_for(ctx.user_id)
        )

    @app.post("/api/session/{session_id}/rate", response_model=SessionSnapshot)
    async def rate(
        session_id: str,
        body: RateRequest,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        with ctx.lock:
            _apply_rating(ctx, body)
        return _snap(ctx)

    @app.post("/api/session/{session_id}/config", response_model=SessionSnapshot)
    async def update_config(
        session_id: str,
        body: UpdateConfigRequest,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        # Config edits are session-start only. Once turn 1 has completed,
        # the seed is locked; the only mid-session change is a path drop
        # via /audit + /drop_path.
        if len(ctx.state.turns) > 0:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Config is locked once turn 1 completes. The only "
                    "mid-session change is a path drop via /audit + /drop_path."
                ),
            )
        overrides = _config_overrides(body)
        if not overrides:
            raise HTTPException(
                status_code=400,
                detail="At least one of rrf_k, per_path_limit, top_k, active_paths must be provided",
            )
        with ctx.lock:
            new_config = with_overrides(ctx.state.current_config, **overrides)
            ctx.state.set_next_config(new_config)
        return _snap(ctx)

    @app.post("/api/session/{session_id}/audit", response_model=SessionSnapshot)
    async def trigger_audit(
        session_id: str, user: User = Depends(current_user)
    ) -> SessionSnapshot:
        """Convert the in-progress turn into a path audit.

        Per-path candidates that ``run_search`` already materialised are
        now expected to be rated alongside the fused top-K. Idempotent:
        a second call on the same turn is a no-op.
        """
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        with ctx.lock:
            state = ctx.state
            if state.current_table is None:
                raise HTTPException(
                    status_code=400, detail="No turn in progress to audit"
                )
            state.audit_mode_active = True
        return _snap(ctx)

    @app.post("/api/session/{session_id}/drop_path", response_model=SessionSnapshot)
    async def drop_path(
        session_id: str,
        body: DropPathRequest,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        """Drop a path after the audit clears Rule B.

        Returns 409 with the rejection reason if Rule B blocks the
        drop; the session config is unchanged on rejection.
        """
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        with ctx.lock:
            state = ctx.state
            if not state.audit_mode_active:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Path drop is only available in audit mode — POST /audit first."
                    ),
                )
            if not state.all_rated():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Rate every fused row and every per-path candidate "
                        "before dropping a path."
                    ),
                )
            allowed, err = state.apply_path_drop(body.path)
            if not allowed:
                raise HTTPException(status_code=409, detail=err or "Drop blocked")
        return _snap(ctx)

    @app.post(
        "/api/session/{session_id}/recommendation",
        response_model=SessionSnapshot,
    )
    async def recommendation_decision(
        session_id: str,
        body: RecommendationDecisionRequest,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        """Apply the operator's apply/ignore choice on the agent's
        latest path-drop recommendation.

        ``apply`` drops the recommended path from ``active_paths``
        immediately via :meth:`SessionState.apply_recommended_drop` —
        no audit step, no Rule B at the apply site (the agent is
        expected to have weighed Rule B1 itself before recommending).
        Mechanical guards still apply: a 409 is returned if the path
        is no longer active or is the last active path. ``ignore`` is
        a no-op state-wise. Both decisions log a sidecar event for
        validation tooling. Single-shot: a second call on the same
        recommendation returns 404.
        """
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        with ctx.lock:
            state = ctx.state
            if not state.turns:
                raise HTTPException(status_code=404, detail="No turns completed yet")
            record = state.turns[-1]
            reflection = record.reflection
            if reflection is None:
                raise HTTPException(
                    status_code=404,
                    detail="No reflection recorded for the last turn",
                )
            rec = reflection.get("path_drop_recommendation")
            if not rec or reflection.get("_recommendation_consumed"):
                raise HTTPException(
                    status_code=404,
                    detail="No pending path-drop recommendation",
                )

            decision = body.decision
            note: str | None = None
            http_error: HTTPException | None = None

            if decision == "apply":
                allowed, err = state.apply_recommended_drop(rec["path"])
                if not allowed:
                    # Mechanical guard rejected (path inactive / last
                    # path). Match the TUI: log the attempt with a
                    # ``note`` so analytics see the rejection, then
                    # raise after the log lands. ``_recommendation_consumed``
                    # stays unset so the operator can re-decide.
                    if rec["path"] not in state.current_config.active_paths:
                        note = "path_inactive"
                    else:
                        note = "last_path"
                    http_error = HTTPException(
                        status_code=409,
                        detail=err or "Drop blocked",
                    )
                else:
                    # One-shot session safeguard: any further
                    # path_drop_recommendation from the agent will be
                    # suppressed at the reflection boundary.
                    state.recommended_drop_applied = True
                    reflection["_recommendation_consumed"] = True
            else:
                reflection["_recommendation_consumed"] = True

            hook = ctx.logging_hook
            if hook is not None and hasattr(hook, "log_event"):
                payload: dict = {"recommendation": rec, "decision": decision}
                if note is not None:
                    payload["note"] = note
                try:
                    hook.log_event(
                        turn=record.turn_number,
                        kind="path_drop_recommendation_decision",
                        **payload,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("log_event failed for recommendation decision: %s", exc)

            if http_error is not None:
                raise http_error

        return _snap(ctx)

    @app.post("/api/session/{session_id}/turn/next", response_model=NextTurnResponse)
    async def next_turn(
        session_id: str, user: User = Depends(current_user)
    ) -> NextTurnResponse:
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        if not ctx.state.all_rated():
            raise HTTPException(
                status_code=409,
                detail="Turn is not complete — some rows are still unrated",
            )
        try:
            reflection = await asyncio.to_thread(_advance_turn, ctx)
        except MilvusSearchError as exc:
            ctx.progress.set_error(str(exc))
            raise HTTPException(status_code=503, detail=str(exc))
        return NextTurnResponse(
            snapshot=_snap(ctx),
            reflection=serialize.reflection_to_dto(reflection),
        )

    @app.get("/api/session/{session_id}/reflection", response_model=ReflectionDTO)
    async def get_reflection(
        session_id: str, user: User = Depends(current_user)
    ) -> ReflectionDTO:
        ctx = _require(session_id, user.id)
        if not ctx.state.turns:
            raise HTTPException(status_code=404, detail="No turns completed yet")
        last = ctx.state.turns[-1].reflection
        dto = serialize.reflection_to_dto(last)
        if dto is None:
            raise HTTPException(
                status_code=404, detail="No reflection recorded for the last turn"
            )
        return dto

    @app.get(
        "/api/session/{session_id}/chunks/{pk}/original",
        response_model=OriginalContentResponse,
    )
    async def get_original_content(
        session_id: str,
        pk: str,
        user: User = Depends(current_user),
    ) -> OriginalContentResponse:
        ctx = _require(session_id, user.id)
        fetcher = _pg_fetcher_for_ctx(ctx)
        key: int | str = int(pk) if pk.lstrip("-").isdigit() else pk
        try:
            original = await asyncio.to_thread(fetcher.fetch_original, key)
        except PostgresFetchError as exc:
            log.warning("Postgres fetch failed for pk=%r: %s", key, exc)
            raise HTTPException(status_code=502, detail=str(exc))
        if original is None:
            raise HTTPException(
                status_code=404,
                detail=f"No original_content for pk {key!r}",
            )
        return OriginalContentResponse(pk=key, original_content=original)

    @app.post(
        "/api/session/{session_id}/harvest/start",
        response_model=HarvestPreflightDTO,
    )
    async def harvest_preflight(
        session_id: str, user: User = Depends(current_user)
    ) -> HarvestPreflightDTO:
        """Compute the harvest preflight payload shown in the confirm modal.

        Fields: FIT count, batch_size, max_k, radius scheme.
        """
        from src.anchor.config import load_harvest_config
        from src.anchor.errors import AnchorError
        from src.anchor.retrieve import DEFAULT_BATCH_SIZE, DEFAULT_MAX_K

        ctx = _require(session_id, user.id)
        if not ctx.state.is_converged:
            raise HTTPException(
                status_code=409,
                detail="Phase 1 has not converged — harvest unavailable.",
            )
        if ctx.state.phase == "ANCHOR_RUNNING":
            raise HTTPException(status_code=409, detail="Harvest already in progress.")
        overrides = _load_overrides_from_disk(
            ctx.state.session_id, _store().runs_dir_for(ctx.user_id)
        )
        try:
            cfg = load_harvest_config(session_overrides=overrides.get("harvest"))
            batch_size = cfg.batch_size
            max_k = cfg.max_k
            radius_scheme = cfg.radius_scheme.value
        except AnchorError:
            # Bad harvest section shouldn't brick the preflight — fall
            # back to module defaults; the run will surface the real
            # error if the config is genuinely broken.
            batch_size = DEFAULT_BATCH_SIZE
            max_k = DEFAULT_MAX_K
            radius_scheme = "per_fit"
        return HarvestPreflightDTO(
            n_fit=len(ctx.state.cumulative_fit_pks),
            batch_size=batch_size,
            max_k=max_k,
            radius_scheme=radius_scheme,
        )

    @app.post(
        "/api/session/{session_id}/harvest/run",
        response_model=SessionSnapshot,
    )
    async def harvest_run(
        session_id: str,
        body: HarvestRunRequest,
        user: User = Depends(current_user),
    ) -> SessionSnapshot:
        """Kick off the harvest worker.

        Idempotent: a second call while a run is in flight returns 409.
        On error the phase rolls to ``ANCHOR_FAILED`` and the snapshot
        carries that state; ``GET /harvest/result`` returns the
        canonical 422 with the runner's error message.
        """
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        if not body.confirm:
            raise HTTPException(status_code=400, detail="confirm must be true")
        if not ctx.state.is_converged:
            raise HTTPException(
                status_code=409,
                detail="Phase 1 has not converged — harvest unavailable.",
            )
        if ctx.state.phase == "ANCHOR_RUNNING":
            raise HTTPException(status_code=409, detail="Harvest already in progress.")
        if ctx.anchor_result is not None and ctx.state.phase == "ANCHOR_DONE":
            raise HTTPException(
                status_code=409, detail="Harvest already completed for this session."
            )

        with ctx.lock:
            ctx.state.set_phase("ANCHOR_RUNNING")
            ctx.progress.reset()
            ctx.progress.set_stage(STAGE_HARVESTING)

        # Spin up a daemon thread — ``run_anchor`` is sync and the
        # progress class is thread-safe. Caller polls ``/progress`` and
        # then ``GET /harvest/result``.
        runs_dir = _store().runs_dir_for(ctx.user_id)
        threading.Thread(
            target=_run_harvest_worker,
            args=(ctx, runs_dir),
            daemon=True,
            name=f"harvest-{ctx.state.session_id[:8]}",
        ).start()
        return _snap(ctx)

    @app.get(
        "/api/session/{session_id}/harvest/result",
        response_model=AnchorResultDTO,
    )
    async def harvest_result(
        session_id: str, user: User = Depends(current_user)
    ) -> AnchorResultDTO:
        ctx = _require(session_id, user.id)
        if ctx.state.phase == "ANCHOR_RUNNING":
            raise HTTPException(status_code=409, detail="Harvest is still running.")
        if ctx.state.phase == "ANCHOR_FAILED":
            err = ctx.progress.snapshot().get("error") or "Harvest failed."
            raise HTTPException(status_code=422, detail=err)
        if ctx.anchor_result is None:
            raise HTTPException(
                status_code=404, detail="No harvest result for this session."
            )
        return serialize.anchor_result_to_dto(ctx.anchor_result)

    # ------------------------------------------------------------------
    # Phase 3 (Refine) — derive → editor → judge → review → finalise
    # ------------------------------------------------------------------

    @app.post(
        "/api/session/{session_id}/refine/start",
        response_model=RefinePreflightDTO,
    )
    async def refine_preflight(
        session_id: str, user: User = Depends(current_user)
    ) -> RefinePreflightDTO:
        from src.refine.config import load_refine_config
        from src.refine.errors import RefineConfigError

        ctx = _require(session_id, user.id)
        if ctx.state.phase != "ANCHOR_DONE" or ctx.anchor_result is None:
            raise HTTPException(
                status_code=409,
                detail="Refine requires a completed harvest pass (ANCHOR_DONE).",
            )
        overrides = _load_overrides_from_disk(
            ctx.state.session_id, _store().runs_dir_for(ctx.user_id)
        )
        try:
            cfg = load_refine_config(session_overrides=overrides.get("refine"))
        except RefineConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not cfg.enabled:
            raise HTTPException(
                status_code=422,
                detail="refine.enabled=false in config; Phase 3 is disabled.",
            )
        retained = (
            ctx.anchor_result.write.n_records
            if ctx.anchor_result.write is not None
            else len(ctx.anchor_result.retrieval.candidates)
        )
        return RefinePreflightDTO(
            phase2_count=retained,
            sample_size=cfg.sample_size,
            n_bins=cfg.n_bins,
            derive_model=cfg.derive_model,
            judge_model=cfg.judge_model,
        )

    @app.post(
        "/api/session/{session_id}/refine/derive",
        response_model=SessionSnapshot,
    )
    async def refine_derive(
        session_id: str, user: User = Depends(current_user)
    ) -> SessionSnapshot:
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        if ctx.state.phase not in ("ANCHOR_DONE", "REFINE_EDITING"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Derive is available after harvest completes "
                    "(ANCHOR_DONE) or to re-derive after a disagreement."
                ),
            )
        if ctx.state.phase == "REFINE_DERIVING":
            raise HTTPException(status_code=409, detail="Derive already in progress.")
        # Pre-flight key check — prevents the silent 200 + background
        # WARNING when OPENROUTER_API_KEY (or whatever ``refine.api_key_env``
        # names) isn't set. See issue #33's "Related cleanup".
        _require_llm_key(section="refine", phase_label="Refine's derive step")

        with ctx.lock:
            ctx.state.set_phase("REFINE_DERIVING")
            ctx.progress.reset()
            ctx.progress.set_stage(STAGE_REFINE_DERIVING)

        runs_dir = _store().runs_dir_for(ctx.user_id)
        threading.Thread(
            target=_run_refine_derive_worker,
            args=(ctx, runs_dir),
            daemon=True,
            name=f"refine-derive-{ctx.state.session_id[:8]}",
        ).start()
        return _snap(ctx)

    @app.get(
        "/api/session/{session_id}/refine/derive_result",
        response_model=DeriveResultDTO,
    )
    async def refine_derive_result(
        session_id: str, user: User = Depends(current_user)
    ) -> DeriveResultDTO:
        ctx = _require(session_id, user.id)
        if ctx.state.phase == "REFINE_DERIVING":
            raise HTTPException(status_code=409, detail="Derive is still running.")
        if ctx.state.phase == "REFINE_FAILED":
            err = ctx.progress.snapshot().get("error") or "Derive failed."
            raise HTTPException(status_code=422, detail=err)
        rs = ctx.state.refine_state
        if rs is None or rs.derive_result is None:
            raise HTTPException(status_code=404, detail="No derive result available.")
        return serialize.derive_result_to_dto(rs.derive_result, rs.rubric_metadata)

    @app.get(
        "/api/session/{session_id}/refine/prompt",
        response_model=RubricPromptDTO,
    )
    async def refine_prompt(
        session_id: str, user: User = Depends(current_user)
    ) -> RubricPromptDTO:
        """Return the shipped rubric prompt + parsed metadata.

        Decoupled from ``derive_result`` so it stays available after
        ``DONE_VIEW`` resume from disk, where the raw derive trace is
        not re-hydrated but the rubric text + metadata are.
        """
        ctx = _require(session_id, user.id)
        rs = ctx.state.refine_state
        if rs is None or rs.rubric_text is None or rs.rubric_metadata is None:
            raise HTTPException(status_code=404, detail="No rubric available.")
        return RubricPromptDTO(
            rubric_text=rs.rubric_text,
            metadata=serialize.rubric_metadata_to_dto(rs.rubric_metadata),
        )

    @app.post(
        "/api/session/{session_id}/refine/rubric",
        response_model=RubricMetadataDTO,
    )
    async def refine_save_rubric(
        session_id: str,
        body: RubricSaveRequest,
        user: User = Depends(current_user),
    ) -> RubricMetadataDTO:
        from src.refine.derive import parse_rubric_prompt
        from src.refine.errors import RefineParseError

        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        rs = ctx.state.refine_state
        if rs is None or rs.rubric_metadata is None:
            raise HTTPException(
                status_code=409,
                detail="Run derive before editing the rubric.",
            )
        old = rs.rubric_metadata
        try:
            new_meta = parse_rubric_prompt(
                body.rubric_text,
                source_session_id=old.source_session_id,
                derive_model_id=old.derive_model_id,
                meta_prompt_path=old.meta_prompt_path,
                meta_prompt_sha256=old.meta_prompt_sha256,
                prompt_path=old.prompt_path,
                version=old.version + 1,
            )
        except RefineParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        rs.rubric_text = body.rubric_text
        rs.rubric_metadata = new_meta
        # Once the operator edits the rubric we drop any stale verdicts
        # so finalize cannot mix decisions across rubric versions.
        rs.judge_result = None
        rs.sample = None
        with ctx.lock:
            ctx.state.set_phase("REFINE_EDITING")
        return serialize.rubric_metadata_to_dto(new_meta)

    @app.post(
        "/api/session/{session_id}/refine/judge",
        response_model=SessionSnapshot,
    )
    async def refine_judge(
        session_id: str, user: User = Depends(current_user)
    ) -> SessionSnapshot:
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        rs = ctx.state.refine_state
        if rs is None or rs.rubric_text is None or rs.rubric_metadata is None:
            raise HTTPException(
                status_code=409,
                detail="Run derive (and optionally save an edited rubric) before judging.",
            )
        if ctx.state.phase == "REFINE_JUDGING":
            raise HTTPException(status_code=409, detail="Judge already in progress.")
        _require_llm_key(section="refine", phase_label="Refine's judge step")

        with ctx.lock:
            ctx.state.set_phase("REFINE_JUDGING")
            ctx.progress.reset()
            ctx.progress.set_stage(STAGE_REFINE_JUDGING)

        runs_dir = _store().runs_dir_for(ctx.user_id)
        pg_table_override = _resolve_pg_table_override(ctx)
        threading.Thread(
            target=_run_refine_judge_worker,
            args=(ctx, runs_dir),
            kwargs={"pg_table_override": pg_table_override},
            daemon=True,
            name=f"refine-judge-{ctx.state.session_id[:8]}",
        ).start()
        return _snap(ctx)

    @app.get(
        "/api/session/{session_id}/refine/judge_result",
        response_model=JudgeResultDTO,
    )
    async def refine_judge_result(
        session_id: str, user: User = Depends(current_user)
    ) -> JudgeResultDTO:
        ctx = _require(session_id, user.id)
        if ctx.state.phase == "REFINE_JUDGING":
            raise HTTPException(status_code=409, detail="Judge is still running.")
        if ctx.state.phase == "REFINE_FAILED":
            err = ctx.progress.snapshot().get("error") or "Judge failed."
            raise HTTPException(status_code=422, detail=err)
        rs = ctx.state.refine_state
        if rs is None or rs.judge_result is None:
            raise HTTPException(status_code=404, detail="No judge result available.")
        return serialize.judge_result_to_dto(rs.judge_result)

    @app.get(
        "/api/session/{session_id}/refine/verdicts",
        response_model=list[VerdictDTO],
    )
    async def refine_verdicts(
        session_id: str, user: User = Depends(current_user)
    ) -> list[VerdictDTO]:
        ctx = _require(session_id, user.id)
        rs = ctx.state.refine_state
        if rs is None or rs.judge_result is None:
            raise HTTPException(status_code=404, detail="No verdicts available.")
        return serialize.verdicts_to_dto_list(rs.judge_result.verdicts)

    @app.post(
        "/api/session/{session_id}/refine/discard",
        response_model=RubricMetadataDTO,
    )
    async def refine_discard(
        session_id: str, user: User = Depends(current_user)
    ) -> RubricMetadataDTO:
        """Discard this judged round and reopen the rubric editor.

        Mirrors the TUI's overall re-edit: bumps ``rubric.version`` by
        one, clears the judge result + sample, and routes the session
        back to ``REFINE_EDITING``. The rubric text is unchanged, so
        the bump is a metadata-only copy — no re-parse. Stage A
        artefacts on disk stay put until the next judge run overwrites
        them.
        """
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        rs = ctx.state.refine_state
        if (
            rs is None
            or rs.rubric_metadata is None
            or rs.rubric_text is None
            or rs.judge_result is None
        ):
            raise HTTPException(
                status_code=409,
                detail="Run judge before discarding.",
            )
        old = rs.rubric_metadata
        new_meta = old.model_copy(update={"version": old.version + 1})
        rs.rubric_metadata = new_meta
        rs.judge_result = None
        rs.sample = None
        with ctx.lock:
            ctx.state.set_phase("REFINE_EDITING")
        return serialize.rubric_metadata_to_dto(new_meta)

    @app.get(
        "/api/session/{session_id}/refine/summary",
        response_model=RefineSummaryDTO,
    )
    async def refine_summary(
        session_id: str, user: User = Depends(current_user)
    ) -> RefineSummaryDTO:
        """Return the Phase 3 summary for an already-finalised session.

        Read-only safe — the endpoint is GET and does not call
        ``_require_writable``, so DONE_VIEW resumes can render the
        summary without 409ing. The live finalize path keeps returning
        the same DTO directly from POST /refine/finalize, so this
        endpoint exists primarily for the resume-from-disk case.
        """
        ctx = _require(session_id, user.id)
        rs = ctx.state.refine_state
        if rs is None or rs.judge_result is None:
            raise HTTPException(status_code=404, detail="No refine summary available.")
        return serialize.refine_summary_to_dto(rs)

    @app.post(
        "/api/session/{session_id}/refine/finalize",
        response_model=RefineSummaryDTO,
    )
    async def refine_finalize(
        session_id: str, user: User = Depends(current_user)
    ) -> RefineSummaryDTO:
        from src.refine.errors import RefineConfigError
        from src.refine.runner import finalize_refine

        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        rs = ctx.state.refine_state
        if rs is None or rs.judge_result is None:
            raise HTTPException(status_code=409, detail="Run judge before finalising.")
        with ctx.lock:
            ctx.progress.reset()
            ctx.progress.set_stage(STAGE_REFINE_FINALIZING)
        try:
            await asyncio.to_thread(
                finalize_refine,
                rs,
                runs_dir=_store().runs_dir_for(ctx.user_id),
                operator_decision="agree",
            )
        except RefineConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        with ctx.lock:
            ctx.state.set_phase("DONE")
        ctx.progress.set_stage(STAGE_DONE)
        return serialize.refine_summary_to_dto(rs)

    # ------------------------------------------------------------------
    # Phase 4 (Apply) — train → threshold review → finalize
    # ------------------------------------------------------------------

    @app.post(
        "/api/session/{session_id}/apply/start",
        response_model=ApplyPreflightDTO,
    )
    async def apply_preflight(
        session_id: str, user: User = Depends(current_user)
    ) -> ApplyPreflightDTO:
        """Preview the Phase 4 input scope before training kicks off.

        Returns the cohort + labels-count + threshold/min_precision the
        runner will use, so the web UI's confirm modal can show what
        the operator's about to commit to. State-neutral; does not
        transition the session.
        """
        from src.apply.config import load_apply_config
        from src.apply.errors import ApplyConfigError, ApplyLoadError
        from src.apply.load_session import load_phase4_session_inputs

        ctx = _require(session_id, user.id)
        if ctx.state.phase not in ("DONE", "APPLY_CONFIRM", "APPLY_FAILED"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Apply requires a finalised Phase 3 pass "
                    "(DONE or APPLY_CONFIRM after refine finalize)."
                ),
            )
        overrides = _load_overrides_from_disk(
            ctx.state.session_id, _store().runs_dir_for(ctx.user_id)
        )
        try:
            cfg = load_apply_config(session_overrides=overrides.get("apply"))
        except ApplyConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not cfg.enabled:
            raise HTTPException(
                status_code=422,
                detail="apply.enabled=false in config; Phase 4 is disabled.",
            )
        try:
            inputs = load_phase4_session_inputs(
                session_id, runs_dir=_store().runs_dir_for(user.id)
            )
        except ApplyLoadError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return ApplyPreflightDTO(
            phase3_finalised=True,
            cohort_count=len(inputs.cohort),
            labels_count=len(inputs.labels),
            confidence_threshold=cfg.confidence_threshold,
            min_precision=cfg.min_precision,
            embedding_dim=cfg.embedding_dim,
        )

    @app.post(
        "/api/session/{session_id}/apply/train",
        response_model=SessionSnapshot,
    )
    async def apply_train(
        session_id: str, user: User = Depends(current_user)
    ) -> SessionSnapshot:
        """Kick off Phase 4 training in a daemon worker."""
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        if ctx.state.phase not in ("DONE", "APPLY_CONFIRM", "APPLY_FAILED"):
            raise HTTPException(
                status_code=409,
                detail="Apply training requires a finalised Phase 3 pass.",
            )
        if ctx.state.phase in ("APPLY_TRAINING", "APPLY_PREPARING"):
            raise HTTPException(
                status_code=409, detail="Apply training already in progress."
            )
        with ctx.lock:
            ctx.state.set_phase("APPLY_TRAINING")
            ctx.progress.reset()
            ctx.progress.set_stage(STAGE_APPLY_TRAINING)
        runs_dir = _store().runs_dir_for(ctx.user_id)
        threading.Thread(
            target=_run_apply_train_worker,
            args=(ctx, runs_dir),
            daemon=True,
            name=f"apply-train-{ctx.state.session_id[:8]}",
        ).start()
        return _snap(ctx)

    @app.get(
        "/api/session/{session_id}/apply/eval",
        response_model=ApplyCalibrateResponse,
    )
    async def apply_eval(
        session_id: str,
        user: User = Depends(current_user),
        threshold: float | None = None,
    ) -> ApplyCalibrateResponse:
        """Live projection + borderline samples at a candidate threshold.

        Polled by the calibration screen while training is in flight
        (returns 409) and again on every slider movement once
        APPLY_REVIEW is live. ``threshold`` defaults to the
        config / training-time default when omitted.
        """
        from src.apply.errors import ApplyError
        from src.apply.evaluator import precision_recall_at
        from src.apply.runner import run_apply_calibrate
        from src.apply.schema import EvalReport

        ctx = _require(session_id, user.id)
        if ctx.state.phase in ("APPLY_TRAINING", "APPLY_PREPARING"):
            raise HTTPException(
                status_code=409, detail="Apply training is still running."
            )
        if ctx.state.phase == "APPLY_FAILED":
            err = ctx.progress.snapshot().get("error") or "Apply training failed."
            raise HTTPException(status_code=422, detail=err)
        ap = ctx.state.apply_state
        if ap is None or ap.eval_report is None:
            raise HTTPException(
                status_code=404, detail="No apply eval available — run training first."
            )
        candidate_threshold = (
            threshold if threshold is not None else ap.eval_report.threshold_default
        )
        if not 0.0 <= candidate_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"threshold must be in [0, 1]; got {candidate_threshold}",
            )
        try:
            projection, borderlines = run_apply_calibrate(
                ap,
                runs_dir=_store().runs_dir_for(ctx.user_id),
                threshold=candidate_threshold,
            )
        except ApplyError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Pooled k-fold predictions back the recompute-at-threshold path
        # so the headline number agrees with the histogram dot for dot.
        pooled_p_keep = (
            ap.eval_output.pooled_p_keep if ap.eval_output is not None else None
        )
        pooled_y = ap.eval_output.pooled_y if ap.eval_output is not None else None
        per_row_p_keep = (
            ap.eval_output.per_row_p_keep if ap.eval_output is not None else None
        )
        training_labels = (
            ap.training_frame.labels if ap.training_frame is not None else None
        )
        if pooled_p_keep is not None and pooled_y is not None:
            precision_at, recall_at = precision_recall_at(
                pooled_y, pooled_p_keep, threshold=candidate_threshold
            )
        else:
            precision_at = ap.eval_report.precision_at_threshold
            recall_at = ap.eval_report.recall_at_threshold
        eval_at_threshold = EvalReport(
            precision_at_threshold=float(precision_at),
            recall_at_threshold=float(recall_at),
            pr_curve=ap.eval_report.pr_curve,
            threshold_default=candidate_threshold,
            threshold_selected_by_cv=ap.eval_report.threshold_selected_by_cv,
            cv_precision_mean=ap.eval_report.cv_precision_mean,
            cv_precision_std=ap.eval_report.cv_precision_std,
            min_precision=ap.eval_report.min_precision,
            eval_n=ap.eval_report.eval_n,
            eval_keep_n=ap.eval_report.eval_keep_n,
            eval_drop_n=ap.eval_report.eval_drop_n,
            eval_methodology=ap.eval_report.eval_methodology,
            n_splits=ap.eval_report.n_splits,
            n_repeats=ap.eval_report.n_repeats,
            borderline_samples=ap.eval_report.borderline_samples,
        )
        # Once eval has been served at least once, the session is in
        # interactive review. Promote the phase if not already there.
        if ctx.state.phase == "APPLY_TRAINING":
            with ctx.lock:
                ctx.state.set_phase("APPLY_REVIEW")
        # Surface per-row averaged p_keep across the R repeats — one
        # entry per labelled chunk, aligned with training_frame.labels.
        # The histogram on the operator's screen now covers all N
        # labelled rows (up from ~100 of held-out eval).
        if per_row_p_keep is not None and training_labels is not None:
            eval_scores_payload = [float(p) for p in per_row_p_keep.tolist()]
            eval_labels_payload = [int(v) for v in training_labels]
        else:
            eval_scores_payload = []
            eval_labels_payload = []
        return ApplyCalibrateResponse(
            projection=serialize.cohort_projection_to_dto(projection),
            borderline_samples=serialize.borderline_samples_to_dtos(borderlines),
            eval_at_threshold=serialize.apply_eval_report_to_dto(
                eval_at_threshold,
                eval_scores=eval_scores_payload,
                eval_labels=eval_labels_payload,
            ),
        )

    @app.post(
        "/api/session/{session_id}/apply/finalize",
        response_model=ApplySummaryDTO,
    )
    async def apply_finalize(
        session_id: str,
        body: ApplyFinalizeRequest,
        user: User = Depends(current_user),
    ) -> ApplySummaryDTO:
        """Apply classifier at the operator-confirmed threshold."""
        from src.apply.errors import ApplyError, ApplyPrecisionError
        from src.apply.runner import finalize_apply

        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        ap = ctx.state.apply_state
        if ap is None or ap.eval_report is None:
            raise HTTPException(
                status_code=409,
                detail="Run apply training before finalising.",
            )
        if not 0.0 <= body.threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=f"threshold must be in [0, 1]; got {body.threshold}",
            )
        with ctx.lock:
            ctx.state.set_phase("APPLY_APPLYING")
            ctx.progress.reset()
            ctx.progress.set_stage(STAGE_APPLY_APPLYING)
        try:
            await asyncio.to_thread(
                finalize_apply,
                ap,
                runs_dir=_store().runs_dir_for(ctx.user_id),
                threshold=body.threshold,
                allow_low_precision=body.allow_low_precision,
            )
        except ApplyPrecisionError as exc:
            with ctx.lock:
                ctx.state.set_phase("APPLY_REVIEW")
            raise HTTPException(
                status_code=422,
                detail={"code": "APPLY_PRECISION_BELOW_BAR", "detail": str(exc)},
            )
        except ApplyError as exc:
            with ctx.lock:
                ctx.state.set_phase("APPLY_FAILED")
            ctx.progress.set_error(str(exc))
            raise HTTPException(status_code=422, detail=str(exc))
        with ctx.lock:
            ctx.state.set_phase("DONE")
        ctx.progress.set_stage(STAGE_DONE)
        return serialize.apply_summary_to_dto(
            ap,
            query=ctx.state.query or "",
            runs_dir=_store().runs_dir_for(ctx.user_id),
            session_id=session_id,
        )

    @app.post(
        "/api/session/{session_id}/apply/cancel",
        response_model=SessionSnapshot,
    )
    async def apply_cancel(
        session_id: str, user: User = Depends(current_user)
    ) -> SessionSnapshot:
        """Bail out of Phase 4 — drop in-memory state, return to DONE.

        Does not delete the Stage A sidecars: they're cheap to leave on
        disk and the operator can re-train without re-running Phase 3.
        """
        ctx = _require(session_id, user.id)
        _require_writable(ctx)
        if ctx.state.phase not in ("APPLY_REVIEW", "APPLY_CONFIRM", "APPLY_FAILED"):
            raise HTTPException(
                status_code=409,
                detail="Cancel is only valid in APPLY_CONFIRM/REVIEW/FAILED.",
            )
        with ctx.lock:
            ctx.state.apply_state = None
            ctx.state.set_phase("DONE")
        ctx.progress.reset()
        ctx.progress.set_stage(STAGE_DONE)
        return _snap(ctx)

    @app.get(
        "/api/session/{session_id}/apply/summary",
        response_model=ApplySummaryDTO,
    )
    async def apply_summary(
        session_id: str, user: User = Depends(current_user)
    ) -> ApplySummaryDTO:
        ctx = _require(session_id, user.id)
        ap = ctx.state.apply_state
        if ap is None or ap.classifier_metadata is None:
            raise HTTPException(status_code=404, detail="No apply summary available.")
        return serialize.apply_summary_to_dto(
            ap,
            query=ctx.state.query or "",
            runs_dir=_store().runs_dir_for(ctx.user_id),
            session_id=session_id,
        )

    @app.get("/api/session/{session_id}/logs/download")
    async def download_logs(
        session_id: str, user: User = Depends(current_user)
    ) -> Response:
        runs_dir = _require_session_files(session_id, user.id)
        flush_run_logging(session_id)
        suffixes = (
            ".log",
            ".jsonl",
            ".details.jsonl",
            ".span_cache.jsonl",
            ".phase2.jsonl",
            ".phase2.meta.json",
            ".phase3.prompt.md",
            ".phase3.rubric.json",
            ".phase3.evidence.jsonl",
            ".phase3.meta.json",
            ".phase4.classifier.json",
            ".phase4.eval.json",
            ".phase4.labels.jsonl",
            ".phase4.meta.json",
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for suffix in suffixes:
                path = runs_dir / f"{session_id}{suffix}"
                if path.exists():
                    zf.write(path, arcname=path.name)
        filename = f"session-{session_id[:8]}.zip"
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/session/{session_id}/rubric/download")
    async def download_rubric(
        session_id: str, user: User = Depends(current_user)
    ) -> Response:
        runs_dir = _require_session_files(session_id, user.id)
        path = runs_dir / f"{session_id}.phase3.prompt.md"
        if not path.exists():
            raise HTTPException(status_code=404, detail="rubric not found")
        filename = f"session-{session_id[:8]}-rubric.md"
        return Response(
            content=path.read_bytes(),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/session/{session_id}/artifacts/download")
    async def download_artifacts(
        session_id: str, user: User = Depends(current_user)
    ) -> Response:
        runs_dir = _require_session_files(session_id, user.id)
        labels_path = runs_dir / f"{session_id}.phase4.labels.jsonl"
        if not labels_path.exists():
            raise HTTPException(status_code=404, detail="artifacts not ready")
        scope, query = _read_session_scope_and_query(runs_dir, session_id)
        fetcher = _pg_fetcher_for_scope(scope)
        payload = await asyncio.to_thread(
            _build_artifact_zip,
            runs_dir,
            session_id,
            scope,
            query,
            fetcher,
        )
        filename = f"session-{session_id[:8]}-artifacts.zip"
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.delete("/api/session/{session_id}", response_model=OkResponse)
    async def end_session(
        session_id: str, user: User = Depends(current_user)
    ) -> OkResponse:
        _store().drop(session_id)
        return OkResponse()

    @app.post("/api/session/{session_id}/discard", response_model=OkResponse)
    async def discard_session(
        session_id: str, user: User = Depends(current_user)
    ) -> OkResponse:
        # Flush the open log handler before unlinking the .log sidecar so we
        # don't race with the logging module on the file we're about to remove.
        flush_run_logging(session_id)
        removed = _store().discard(session_id, user.id)
        if not removed:
            raise HTTPException(status_code=404, detail="session not found")
        return OkResponse()

    # Serve the React SPA's build output. Mount last so /api/* routes
    # match first. DEKA_WEB_DIST overrides the path for tests / unusual
    # layouts; default is <repo>/web/dist relative to this file.
    dist_override = os.environ.get("DEKA_WEB_DIST")
    if dist_override:
        web_dist = Path(dist_override)
    else:
        web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"

    index_html = web_dist / "index.html"
    if index_html.is_file():
        app.mount(
            "/assets",
            StaticFiles(directory=web_dist / "assets"),
            name="assets",
        )

        # SPA catch-all: any non-/api/* GET that didn't match a registered
        # route falls back to index.html so client-side routing works on
        # deep links / refresh. Registered last so it's the lowest-priority
        # match in FastAPI's router. Explicitly 404 /api/* paths so an
        # unknown API route doesn't masquerade as the SPA HTML.
        @app.get("/{full_path:path}", include_in_schema=False)
        def _spa_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/") or full_path == "api":
                raise HTTPException(status_code=404)
            return FileResponse(index_html)
    else:
        log.warning(
            "web/dist not found at %s — SPA will not be served. "
            "Run `npm run build` in web/ for dev parity.",
            web_dist,
        )

    return app


def _default_pg_setup() -> tuple[OriginalContentFetcher | None, bool]:
    """Load Postgres config and build a fetcher.

    Returns ``(fetcher, disabled_by_admin)``. The two None cases are
    distinguished so the endpoint can respond with 503 (not configured)
    vs 403 (explicitly turned off).
    """
    try:
        config = load_postgres_config()
    except ConfigError as exc:
        log.warning(
            "Postgres section not configured; /original endpoint disabled: %s", exc
        )
        return None, False
    if not config.enabled:
        log.info("Context expansion disabled via config (postgres.enabled=false)")
        return None, True
    return OriginalContentFetcher(config), False


_pg_fetcher_instance, _pg_disabled = _default_pg_setup()
app = create_app(
    pg_fetcher=_pg_fetcher_instance,
    pg_disabled_by_admin=_pg_disabled,
)


def _bootstrap_turn(ctx: SessionContext, query: str) -> None:
    """Turn-0: probe, adapt, run fused search, seed the session."""
    ctx.progress.reset()
    config = ctx.state.current_config
    ctx.progress.set_stage(STAGE_PROBING)
    probe = probe_only(query, config)
    ctx.progress.set_stage(STAGE_ADAPTING)
    adapted = adapt_config(config, probe)
    adapted_config = adapted.config
    ctx.state.current_config = adapted_config
    ctx.state.probe_summary = _build_probe_summary(probe, adapted)
    table = run_search(
        query,
        adapted_config,
        probe=probe,
        seen_pks=frozenset(ctx.state.seen_pks),
        extract_span=_build_extract_span(ctx, query),
        on_stage=_stage_observer(ctx),
    )
    _persist_auto_retry_bump(ctx, table)
    ctx.state.start_turn(table)
    ctx.progress.set_stage(STAGE_DONE)


def _persist_auto_retry_bump(ctx: SessionContext, table: Any) -> None:
    """Carry an auto-retry's bumped per_path_limit into ``state.current_config``.

    The retry inside ``run_search`` chose a deeper per_path_limit to recover
    a usable survivor set; subsequent turns should start from that floor
    rather than the seed. We mark the override on the state so the next
    turn's reflection prompt sees the bumped value naturally and the
    progress log records what was actually used.
    """
    if not table.search_diagnostics:
        return
    auto_retry = table.search_diagnostics.get("auto_retry")
    if not auto_retry or not auto_retry.get("triggered"):
        return
    final = auto_retry.get("final_per_path_limit")
    if not isinstance(final, int) or final <= ctx.state.current_config.per_path_limit:
        return
    ctx.state.current_config = with_overrides(
        ctx.state.current_config, per_path_limit=final
    )
    log.info(
        "Persisted auto-retry per_path_limit %d into session %s",
        final,
        ctx.state.session_id,
    )


def _build_extract_span(ctx: SessionContext, query: str):
    """Build the per-turn span-extraction closure, or ``None`` if unavailable."""
    if ctx.extractor is None:
        return None
    return build_extract_span(
        ctx.extractor,
        ctx.state,
        query,
        on_progress=ctx.progress.set_processed,
    )


def _build_probe_summary(probe, adapted) -> dict[str, Any]:
    """JSON-safe snapshot of the Turn-0 probe + adapt diagnostics.

    Stored on ``state.probe_summary`` so the web UI can show a
    dismissible banner without re-running the probe.
    """
    return {
        "query": probe.query,
        "stats_by_path": {
            path: {k: stats[k] for k in stats}
            for path, stats in probe.stats_by_path.items()
        },
        "rationale": list(adapted.rationale),
        "flags": list(adapted.flags),
    }


def _build_artifact_zip(
    runs_dir: Path,
    session_id: str,
    scope: str | None,
    query: str,
    fetcher: OriginalContentFetcher,
) -> bytes:
    """Build the `merged.csv` + `metadata.json` zip for a finished session.

    Streams `<sid>.phase4.labels.jsonl`, binds each PK's
    ``original_content`` via Postgres, and produces an in-memory zip.
    ``user_id`` / ``counselor_id`` are parsed from the canonical PK
    prefix (`{user_id}-{counselor_id}-{chunk_id}-{phase}-p`) rather
    than fetched from Postgres.
    """
    labels_path = runs_dir / f"{session_id}.phase4.labels.jsonl"
    pks: list[str] = []
    verdicts: list[str] = []
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pks.append(str(row["pk"]))
            verdicts.append(str(row["verdict"]))

    bound = fetcher.fetch_originals(pks)

    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf,
        fieldnames=["user_id", "counselor_id", "original_content", "label"],
    )
    writer.writeheader()
    missing = 0
    for pk, verdict in zip(pks, verdicts):
        parts = pk.rsplit("-", 3)
        sample = parts[0] if len(parts) == 4 else ""
        uid, _, cid = sample.partition("-")
        content = bound.get(pk)
        if content is None:
            missing += 1
            content = ""
        writer.writerow(
            {
                "user_id": uid,
                "counselor_id": cid,
                "original_content": content,
                "label": verdict,
            }
        )

    n_keep = sum(1 for v in verdicts if v == "KEEP")
    n_drop = sum(1 for v in verdicts if v == "DROP")

    phase4_meta = _read_json(runs_dir / f"{session_id}.phase4.meta.json")
    session_end = phase4_meta.get("ts", "") if phase4_meta else ""
    threshold = phase4_meta.get("threshold") if phase4_meta else None
    eval_metrics = (phase4_meta or {}).get("eval_metrics") or {}
    precision = eval_metrics.get("precision_at_threshold")

    session_start = _first_turn_timestamp(runs_dir / f"{session_id}.jsonl")
    if not session_start:
        phase2_meta = _read_json(runs_dir / f"{session_id}.phase2.meta.json")
        session_start = (phase2_meta or {}).get("ts", "")

    metadata = {
        "session_id": session_id,
        "scope": scope,
        "query": query,
        "session_start": session_start,
        "session_end": session_end,
        "n_chunks_total": len(pks),
        "n_keep": n_keep,
        "n_drop": n_drop,
        "threshold": threshold,
        "precision_at_threshold": precision,
        "n_missing_content": missing,
    }

    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("merged.csv", csv_buf.getvalue())
        zf.writestr(
            "metadata.json",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )
    return out.getvalue()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _first_turn_timestamp(jsonl_path: Path) -> str:
    if not jsonl_path.exists():
        return ""
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ts = row.get("timestamp")
                if isinstance(ts, str):
                    return ts
                return ""
    except (OSError, json.JSONDecodeError):
        return ""
    return ""


def _read_session_scope_and_query(
    runs_dir: Path, session_id: str
) -> tuple[str | None, str]:
    """Read ``(scope_name, query)`` from the canonical jsonl's first turn.

    Used by read-only download endpoints that operate on cold sessions
    and therefore can't rely on the in-memory SessionContext.
    """
    jsonl_path = runs_dir / f"{session_id}.jsonl"
    if not jsonl_path.exists():
        return None, ""
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                scope = row.get("scope")
                query = row.get("query")
                return (
                    scope if isinstance(scope, str) else None,
                    query if isinstance(query, str) else "",
                )
    except (OSError, json.JSONDecodeError):
        pass
    return None, ""


def _stage_observer(ctx: SessionContext):
    """Callback passed to ``run_search`` to mirror its stage transitions into progress."""

    def _observe(stage: str, info: dict[str, Any]) -> None:
        info = info if isinstance(info, dict) else {}
        total = info.get("total")
        ctx.progress.set_stage(stage, total=total)
        if stage == "searching_retry":
            attempt = info.get("attempt")
            ppl = info.get("per_path_limit")
            previous = info.get("previous_survivors")
            target = info.get("min_survivors")
            ctx.progress.set_detail(
                f"Only {previous} survivor(s) after filters "
                f"(target {target}) — re-fusing with per_path_limit={ppl} "
                f"(attempt {attempt}/3)"
            )

    return _observe


def _config_defaults_projection() -> dict[str, dict[str, Any]]:
    """Read each phase section from config.yaml and project the curated keys.

    The projection is restricted to :data:`override_allowlist` so this
    endpoint never echoes infrastructure values (URIs, model ids, API
    key env vars). Used by ``GET /api/config/defaults`` to pre-fill the
    [Edit parameters] modal in the query page.
    """
    from src.config_loader import load_section
    from .schemas import override_allowlist

    out: dict[str, dict[str, Any]] = {}
    for section, keys in override_allowlist().items():
        block = load_section(section)
        out[section] = {k: _jsonify_default(block[k]) for k in keys if k in block}
    return out


def _jsonify_default(value: Any) -> Any:
    """Convert YAML-loaded scalars to JSON-friendly equivalents.

    The override allow-list only covers scalars / lists / bools / enum
    strings; nothing here needs deep conversion, but ``frozenset`` and
    enum members occasionally sneak through if a future loader caches
    coerced values back into the section dict.
    """
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    return value


def _config_overrides(body: UpdateConfigRequest) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if body.rrf_k is not None:
        overrides["rrf_k"] = body.rrf_k
    if body.per_path_limit is not None:
        overrides["per_path_limit"] = body.per_path_limit
    if body.top_k is not None:
        overrides["top_k"] = body.top_k
    if body.active_paths is not None:
        if not body.active_paths:
            raise HTTPException(
                status_code=400, detail="active_paths must be non-empty"
            )
        overrides["active_paths"] = frozenset(body.active_paths)
    return overrides


def _apply_rating(ctx: SessionContext, body: RateRequest) -> None:
    rating: Rating = body.rating
    if body.rank is not None:
        ctx.state.rate(body.rank, rating)
        return
    if body.path is not None and body.rank_in_path is not None:
        ctx.state.rate_candidate(body.path, body.rank_in_path, rating)
        return
    raise HTTPException(
        status_code=400,
        detail="rate request must include either 'rank' or both 'path' and 'rank_in_path'",
    )


def _advance_turn(ctx: SessionContext) -> dict[str, Any] | None:
    """Complete the current turn, reflect, and run the next search.

    The session config is locked once turn 1 starts; the only mid-session
    config change is a path drop performed via the operator-triggered
    audit flow (``state.apply_path_drop``). Reflection contributes the
    reasoning trace and a CONVERGED signal — it does not prescribe.
    """
    state = ctx.state
    ctx.progress.reset()
    record = state.complete_turn()
    # Probe diagnostics are a Turn-1 banner only. Clear once the first
    # turn completes so subsequent snapshots don't re-render the banner.
    state.probe_summary = None

    reflection: dict[str, Any] | None = None
    if ctx.reflection_hook is not None and not state.is_converged:
        ctx.progress.set_stage(STAGE_REFLECTING)
        try:
            reflection = ctx.reflection_hook.reflect(state)
        except Exception as exc:  # noqa: BLE001 — surface reflection failures but don't crash the turn
            log.warning("Reflection failed: %s", exc)
            reflection = {"validation_error": f"reflection failed: {exc}"}
        record.reflection = reflection

    if ctx.logging_hook is not None:
        try:
            ctx.logging_hook.log_turn(state, record)
        except Exception as exc:  # noqa: BLE001
            log.warning("Progress log write failed: %s", exc)

    if state.is_converged:
        ctx.progress.set_stage(STAGE_DONE)
        return reflection

    table = run_search(
        state.query or "",
        state.current_config,
        seen_pks=frozenset(state.seen_pks),
        extract_span=_build_extract_span(ctx, state.query or ""),
        on_stage=_stage_observer(ctx),
    )
    _persist_auto_retry_bump(ctx, table)
    state.start_turn(table)
    ctx.progress.set_stage(STAGE_DONE)
    return reflection


def _load_overrides_from_disk(
    session_id: str, runs_dir: Any
) -> dict[str, dict[str, Any]]:
    """Read ``<sid>.overrides.json`` from a scoped runs dir.

    Returns ``{}`` when the file is absent (the common case — most
    sessions have no overrides). A malformed file logs a warning and
    returns ``{}`` so a corrupt sidecar can't dead-end a Phase 2/3/4
    run; the operator can still edit YAML defaults and re-trigger.
    """
    from src.config_loader import ConfigFileError, load_session_overrides

    try:
        return load_session_overrides(session_id, runs_dir, user_id=None)
    except ConfigFileError as exc:
        log.warning(
            "session overrides sidecar unreadable for %s: %s (proceeding with YAML defaults)",
            session_id,
            exc,
        )
        return {}


def _run_harvest_worker(ctx: SessionContext, runs_dir: Any) -> None:
    """Run :func:`src.anchor.runner.run_anchor` in a daemon thread.

    Marshals the runner's ``progress`` callback (free-form strings) into
    ``ctx.progress.set_detail`` so the /progress endpoint surfaces each
    ``_emit`` line.
    """
    from src.anchor.errors import AnchorError
    from src.anchor.runner import run_anchor

    def on_progress(msg: str) -> None:
        ctx.progress.set_detail(msg)

    overrides = _load_overrides_from_disk(ctx.state.session_id, runs_dir)
    try:
        result = run_anchor(
            ctx.state.session_id,
            runs_dir=runs_dir,
            config=ctx.state.current_config,
            progress=on_progress,
            harvest_overrides=overrides.get("harvest"),
        )
    except AnchorError as exc:
        log.warning("harvest: validation/runtime error: %s", exc)
        with ctx.lock:
            ctx.state.set_phase("ANCHOR_FAILED")
        ctx.progress.set_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 — log + surface, don't crash the loop
        log.exception("harvest: unexpected error")
        with ctx.lock:
            ctx.state.set_phase("ANCHOR_FAILED")
        ctx.progress.set_error(f"Unexpected error: {exc}")
        return

    with ctx.lock:
        ctx.anchor_result = result
        ctx.state.set_phase("ANCHOR_DONE")
    ctx.progress.set_stage(STAGE_DONE)
    log.info(
        "harvest: complete (verdict=%s, retained=%d)",
        result.recovery.verdict,
        result.write.n_records
        if result.write is not None
        else len(result.retrieval.candidates),
    )


def _apply_enabled() -> bool:
    """Return True when ``apply.enabled`` is set in config.

    Defaults to False on any config error so we don't dead-end an
    operator into a phase that's deliberately turned off.
    """
    try:
        from src.apply.config import load_apply_config

        return load_apply_config().enabled
    except Exception as exc:  # noqa: BLE001 — operator-visible elsewhere
        log.debug("apply config unavailable: %s", exc)
        return False


def _run_apply_train_worker(ctx: SessionContext, runs_dir: Any) -> None:
    """Run :func:`src.apply.runner.run_apply_train` in a daemon thread.

    Flow: APPLY_TRAINING (fit LR + Stage A write) → APPLY_PREPARING
    (warm the cohort projection at the default threshold so the
    calibration panel renders with data already in hand) →
    APPLY_REVIEW (operator can drag the slider). Without the pre-warm
    the panel would mount on a cold cache and the first slider value
    would block on a multi-second cohort embedding fetch.
    """
    from src.apply.errors import ApplyError
    from src.apply.runner import run_apply_calibrate, run_apply_train

    overrides = _load_overrides_from_disk(ctx.state.session_id, runs_dir)
    try:
        state = run_apply_train(
            ctx.state.session_id,
            runs_dir=runs_dir,
            apply_overrides=overrides.get("apply"),
        )
    except ApplyError as exc:
        log.warning("apply train: %s", exc)
        with ctx.lock:
            ctx.state.set_phase("APPLY_FAILED")
        ctx.progress.set_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("apply train: unexpected error")
        with ctx.lock:
            ctx.state.set_phase("APPLY_FAILED")
        ctx.progress.set_error(f"Unexpected error: {exc}")
        return

    # Stash the trained state and move into the prep sub-phase. apply_state
    # is set here so /apply/eval can find the trained classifier even if
    # the operator happens to hit it during the prep window.
    with ctx.lock:
        ctx.state.apply_state = state
        ctx.state.set_phase("APPLY_PREPARING")
    ctx.progress.set_stage(STAGE_APPLY_PREPARING)
    eval_report = state.eval_report
    log.info(
        "apply train: complete (eval_precision=%.3f passes_bar=%s); "
        "preparing cohort projection at τ=%.2f",
        eval_report.precision_at_threshold if eval_report else 0.0,
        eval_report.passes_bar if eval_report else False,
        state.cfg.confidence_threshold,
    )

    try:
        run_apply_calibrate(
            state,
            runs_dir=runs_dir,
            threshold=state.cfg.confidence_threshold,
        )
    except ApplyError as exc:
        log.warning("apply prepare: %s", exc)
        with ctx.lock:
            ctx.state.set_phase("APPLY_FAILED")
        ctx.progress.set_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("apply prepare: unexpected error")
        with ctx.lock:
            ctx.state.set_phase("APPLY_FAILED")
        ctx.progress.set_error(f"Unexpected error: {exc}")
        return

    with ctx.lock:
        ctx.state.set_phase("APPLY_REVIEW")
    ctx.progress.set_stage(STAGE_DONE)
    log.info("apply prepare: cohort projection ready; panel armed")


def _run_refine_derive_worker(ctx: SessionContext, runs_dir: Any) -> None:
    """Run :func:`src.refine.runner.run_refine_derive` in a daemon thread.

    Derive is a single LLM call; the runner doesn't expose progress
    updates, so we just flip the phase and let the response payload
    carry the result.
    """
    from src.refine.errors import RefineError
    from src.refine.runner import run_refine_derive

    overrides = _load_overrides_from_disk(ctx.state.session_id, runs_dir)
    try:
        state = run_refine_derive(
            ctx.state.session_id,
            runs_dir=runs_dir,
            refine_overrides=overrides.get("refine"),
        )
    except RefineError as exc:
        log.warning("refine derive: %s", exc)
        with ctx.lock:
            ctx.state.set_phase("REFINE_FAILED")
        ctx.progress.set_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("refine derive: unexpected error")
        with ctx.lock:
            ctx.state.set_phase("REFINE_FAILED")
        ctx.progress.set_error(f"Unexpected error: {exc}")
        return

    with ctx.lock:
        ctx.state.refine_state = state
        ctx.state.set_phase("REFINE_EDITING")
    ctx.progress.set_stage(STAGE_DONE)
    log.info(
        "refine derive: complete (model=%s attempts=%d)",
        state.derive_result.derive_model_id if state.derive_result else "?",
        state.derive_result.attempts if state.derive_result else 0,
    )


def _run_refine_judge_worker(
    ctx: SessionContext,
    runs_dir: Any,
    *,
    pg_table_override: str | None = None,
) -> None:
    """Run :func:`src.refine.runner.run_refine_judge` in a daemon thread.

    The judge fans out per-chunk LLM calls and reports
    ``progress(done, total)`` after each. We mirror those into the
    ``Progress`` snapshot so the polling endpoint surfaces the counter.

    ``pg_table_override`` is the chosen :class:`Scope`'s
    ``postgres_table`` — the global ``postgres.table`` was dropped, so a
    non-None override is required for any judge call that needs to
    fetch chunk content.
    """
    from dataclasses import replace
    from src.postgres.config import load_postgres_config
    from src.postgres.fetch import OriginalContentFetcher
    from src.refine.errors import RefineConfigError, RefineError
    from src.refine.runner import run_refine_judge

    rs = ctx.state.refine_state
    if rs is None:
        log.error("refine judge: no refine_state on context")
        with ctx.lock:
            ctx.state.set_phase("REFINE_FAILED")
        ctx.progress.set_error("Internal: refine_state missing.")
        return

    try:
        pg_cfg = load_postgres_config()
        if not pg_cfg.enabled:
            raise RefineConfigError(
                "Phase 3 judge needs Postgres for chunk content but "
                "postgres.enabled=false in config.yaml."
            )
        if not pg_table_override:
            raise RefineConfigError(
                "Phase 3 judge needs a scope-routed Postgres table but "
                "the session has no scope bound (or the scope is "
                "missing 'postgres_table' in scopes.yaml)."
            )
        log.info(
            "refine judge: using scope-routed Postgres table %r", pg_table_override
        )
        pg_cfg = replace(pg_cfg, table=pg_table_override)
        fetcher = OriginalContentFetcher(pg_cfg)
    except RefineConfigError as exc:
        with ctx.lock:
            ctx.state.set_phase("REFINE_FAILED")
        ctx.progress.set_error(str(exc))
        return

    def on_progress(done: int, total: int) -> None:
        ctx.progress.set_processed(done)
        # set_processed only updates ``processed``; total is set on the
        # initial set_stage call. Re-set when total surfaces for the
        # first time to keep the polling counter accurate.
        snap = ctx.progress.snapshot()
        if snap.get("total") != total:
            ctx.progress.set_stage(STAGE_REFINE_JUDGING, total=total)
            ctx.progress.set_processed(done)

    try:
        run_refine_judge(
            rs,
            runs_dir=runs_dir,
            fetcher=fetcher,
            progress=on_progress,
        )
    except RefineError as exc:
        log.warning("refine judge: %s", exc)
        with ctx.lock:
            ctx.state.set_phase("REFINE_FAILED")
        ctx.progress.set_error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("refine judge: unexpected error")
        with ctx.lock:
            ctx.state.set_phase("REFINE_FAILED")
        ctx.progress.set_error(f"Unexpected error: {exc}")
        return
    finally:
        try:
            fetcher.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("refine judge: fetcher close raised: %s", exc)

    with ctx.lock:
        ctx.state.set_phase("REFINE_REVIEW")
    ctx.progress.set_stage(STAGE_DONE)
    log.info(
        "refine judge: complete (verdicts=%d)",
        len(rs.judge_result.verdicts) if rs.judge_result else 0,
    )
