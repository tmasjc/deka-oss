"""Pre-flight environment / config checks run before a session starts.

Issue #33: lazy config validation lets an operator spend 10–20 minutes on
Phase 1 + harvest only to have ``/refine/derive`` silently fail because
``OPENROUTER_API_KEY`` was never set. The pre-flight runs every check the
session lifecycle could plausibly hit (LLM keys for reflection / refine /
extractor, embed-service reachability, Milvus connection + the scope's
collection, Postgres reachability + table) and surfaces failures with a
typed code so the UI can name the missing knob.

The checks return :class:`PreflightCheckResult` records in a stable order;
``run_preflight`` runs the network-bound ones in a thread pool so total
wall time is bounded by the slowest single check rather than the sum.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from src.config_loader import ConfigFileError, load_section
from src.paths import resolve_prompts_dir

log = logging.getLogger(__name__)


# Stable ordering — the UI uses the list index for the staggered reveal.
_CHECK_ORDER: tuple[str, ...] = (
    "prompts",
    "users.yaml",
    "scopes.yaml",
    "embed_service",
    "milvus.collection",
    "postgres",
    "llm.reflection",
    "llm.refine.derive",
    "llm.refine.judge",
    "llm.span_extractor",
)


_REQUIRED_PROMPT_FILES: tuple[str, ...] = (
    "SYSTEM.md",
    "REFLECTION.md",
    "EXTRACTION.md",
    "RUBRIC_DERIVE.md",
)


@dataclass(frozen=True)
class PreflightCheckResult:
    """Outcome of one pre-flight check.

    ``code`` is the machine-readable failure label the UI keys off
    (e.g. ``MISSING_LLM_KEY``); empty when the check passes.
    ``env_var`` carries the offending env var when the failure is a
    missing API key, so the UI can render *exactly* which knob to set.
    """

    name: str
    status: str  # "ok" or "fail"
    detail: str
    code: str = ""
    env_var: str = ""


@dataclass
class PreflightContext:
    """Bundle the inputs every check needs into one object so the
    orchestrator can hand each check exactly what it requires.

    The ``*_probe`` callables exist as test seams — production checks
    use the real loaders below; tests inject callables that return
    canned shapes so the suite doesn't need a live Milvus / Postgres /
    embed service to run.
    """

    scope_name: str
    users_registry: Any = None
    scopes_registry: Any = None
    network_timeout: float = 3.0
    load_section_fn: Callable[[str], dict[str, Any]] = field(
        default=lambda section: load_section(section)
    )
    embed_probe: Callable[[str, float], bool] | None = None
    milvus_probe: Callable[[str, float], list[str]] | None = None
    postgres_probe: Callable[[Any], tuple[bool, str]] | None = None


def _ok(name: str, detail: str = "") -> PreflightCheckResult:
    return PreflightCheckResult(name=name, status="ok", detail=detail)


def _fail(
    name: str,
    *,
    code: str,
    detail: str,
    env_var: str = "",
) -> PreflightCheckResult:
    return PreflightCheckResult(
        name=name, status="fail", detail=detail, code=code, env_var=env_var
    )


# ---------------------------------------------------------------------------
# Per-check implementations
# ---------------------------------------------------------------------------


def check_users_yaml(ctx: PreflightContext) -> PreflightCheckResult:
    if ctx.users_registry is None:
        return _fail(
            "users.yaml",
            code="USERS_YAML_MISSING",
            detail="users.yaml registry not loaded into the app.",
        )
    try:
        n = sum(1 for _ in ctx.users_registry)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "users.yaml",
            code="USERS_YAML_INVALID",
            detail=f"users.yaml registry unreadable: {exc}",
        )
    return _ok("users.yaml", f"{n} user(s) loaded")


def check_scopes_yaml(ctx: PreflightContext) -> PreflightCheckResult:
    if ctx.scopes_registry is None:
        return _fail(
            "scopes.yaml",
            code="SCOPES_YAML_MISSING",
            detail="scopes.yaml registry not loaded into the app.",
        )
    try:
        names = ctx.scopes_registry.names()
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "scopes.yaml",
            code="SCOPES_YAML_INVALID",
            detail=f"scopes.yaml registry unreadable: {exc}",
        )
    return _ok("scopes.yaml", f"{len(names)} scope(s) loaded")


def check_prompts(ctx: PreflightContext) -> PreflightCheckResult:
    """Verify the prompts directory exists and contains the four files
    every harness step loads at runtime.
    """
    prompts_dir = resolve_prompts_dir()
    if not prompts_dir.is_dir():
        return _fail(
            "prompts",
            code="PROMPTS_DIR_MISSING",
            env_var="DEKA_PROMPTS_DIR",
            detail=(
                f"Prompts directory {prompts_dir!r} does not exist. "
                f"Set DEKA_PROMPTS_DIR to the directory holding "
                f"{', '.join(_REQUIRED_PROMPT_FILES)}."
            ),
        )
    missing = [
        name for name in _REQUIRED_PROMPT_FILES if not (prompts_dir / name).is_file()
    ]
    if missing:
        return _fail(
            "prompts",
            code="PROMPTS_FILE_MISSING",
            env_var="DEKA_PROMPTS_DIR",
            detail=(
                f"Prompts directory {prompts_dir!r} is missing required "
                f"file(s): {', '.join(missing)}."
            ),
        )
    return _ok(
        "prompts",
        f"all {len(_REQUIRED_PROMPT_FILES)} files present at {prompts_dir}",
    )


def _default_embed_probe(embed_url: str, timeout: float) -> bool:
    """Return True if ``embed_url`` is reachable.

    Probes ``<embed_url>/model`` (already used by ``src.anchor.loader``).
    A 200 means the service exposes the optional model endpoint; a 404
    means the service is up but doesn't expose ``/model`` — both count
    as reachable for pre-flight purposes. Anything else is treated as
    unreachable.
    """
    url = embed_url.rstrip("/") + "/model"
    resp = requests.get(url, timeout=timeout)
    return resp.status_code in (200, 404)


def check_embed_service(ctx: PreflightContext) -> PreflightCheckResult:
    try:
        search_cfg = ctx.load_section_fn("search")
    except ConfigFileError as exc:
        return _fail(
            "embed_service",
            code="EMBED_CONFIG_MISSING",
            detail=f"search config unreadable: {exc}",
        )
    embed_url = search_cfg.get("embed_url", "")
    if not isinstance(embed_url, str) or not embed_url.strip():
        return _fail(
            "embed_service",
            code="EMBED_URL_MISSING",
            detail="search.embed_url is not configured.",
        )
    probe = ctx.embed_probe or _default_embed_probe
    try:
        ok = probe(embed_url, ctx.network_timeout)
    except requests.RequestException as exc:
        return _fail(
            "embed_service",
            code="EMBED_UNREACHABLE",
            detail=f"Embed service at {embed_url} unreachable: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "embed_service",
            code="EMBED_UNREACHABLE",
            detail=f"Embed service at {embed_url} probe failed: {exc}",
        )
    if not ok:
        return _fail(
            "embed_service",
            code="EMBED_UNREACHABLE",
            detail=f"Embed service at {embed_url} returned an unexpected status.",
        )
    return _ok("embed_service", f"reachable at {embed_url}")


def _default_milvus_probe(milvus_uri: str, timeout: float) -> list[str]:
    """Return the list of collections on the Milvus instance at
    ``milvus_uri``. ``timeout`` is honoured by the underlying gRPC call."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=milvus_uri, timeout=timeout)
    try:
        return list(client.list_collections())
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                log.debug("MilvusClient.close() raised: %s", exc)


def check_milvus(ctx: PreflightContext) -> PreflightCheckResult:
    try:
        search_cfg = ctx.load_section_fn("search")
    except ConfigFileError as exc:
        return _fail(
            "milvus.collection",
            code="MILVUS_CONFIG_MISSING",
            detail=f"search config unreadable: {exc}",
        )
    milvus_uri = search_cfg.get("milvus_uri", "")
    if not isinstance(milvus_uri, str) or not milvus_uri.strip():
        return _fail(
            "milvus.collection",
            code="MILVUS_URI_MISSING",
            detail="search.milvus_uri is not configured.",
        )
    if ctx.scopes_registry is None:
        return _fail(
            "milvus.collection",
            code="SCOPES_YAML_MISSING",
            detail="scopes.yaml registry not loaded; cannot resolve target collection.",
        )
    try:
        scope = ctx.scopes_registry.get(ctx.scope_name)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "milvus.collection",
            code="SCOPE_UNKNOWN",
            detail=f"scope {ctx.scope_name!r} not in registry: {exc}",
        )
    expected = scope.milvus_collection
    probe = ctx.milvus_probe or _default_milvus_probe
    try:
        collections = probe(milvus_uri, ctx.network_timeout)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "milvus.collection",
            code="MILVUS_UNREACHABLE",
            detail=f"Milvus at {milvus_uri} unreachable: {exc}",
        )
    if expected not in collections:
        return _fail(
            "milvus.collection",
            code="MILVUS_NO_COLLECTION",
            detail=(
                f"Collection {expected!r} not found on Milvus at {milvus_uri}; "
                f"available: {sorted(collections)[:10]}"
            ),
        )
    return _ok("milvus.collection", f"{expected!r} on {milvus_uri}")


def _default_postgres_probe(pg_cfg: Any) -> tuple[bool, str]:
    """``SELECT 1`` against the configured DSN; verify the configured
    table exists. Returns ``(ok, detail)``.
    """
    import psycopg

    try:
        with psycopg.connect(
            pg_cfg.dsn, connect_timeout=pg_cfg.connect_timeout
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
                cur.execute("SELECT to_regclass(%s) IS NOT NULL", (pg_cfg.table,))
                row = cur.fetchone()
                table_exists = bool(row and row[0])
    except psycopg.Error as exc:
        return False, f"connection / query failed: {exc}"
    if not table_exists:
        return False, f"table {pg_cfg.table!r} does not exist"
    return True, f"reachable; table {pg_cfg.table!r} exists"


def check_postgres(ctx: PreflightContext) -> PreflightCheckResult:
    """Postgres is optional — when ``postgres.enabled=false`` we report
    OK with a "disabled" detail so the operator can see the feature is
    deliberately turned off rather than silently absent.

    The probe targets the chosen scope's ``postgres_table`` (the
    global ``postgres.table`` was dropped — every scope declares its
    own table in ``scopes.yaml``). Mirrors ``check_milvus``.
    """
    from dataclasses import replace
    from src.postgres.config import load_postgres_config
    from src.search.errors import ConfigError

    try:
        pg_cfg = load_postgres_config()
    except ConfigError as exc:
        return _fail(
            "postgres",
            code="POSTGRES_CONFIG_INVALID",
            detail=f"postgres config invalid: {exc}",
        )
    if not pg_cfg.enabled:
        return _ok("postgres", "disabled (postgres.enabled=false)")
    if ctx.scopes_registry is None:
        return _fail(
            "postgres",
            code="SCOPES_YAML_MISSING",
            detail="scopes.yaml registry not loaded; cannot resolve target table.",
        )
    try:
        scope = ctx.scopes_registry.get(ctx.scope_name)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "postgres",
            code="SCOPE_UNKNOWN",
            detail=f"scope {ctx.scope_name!r} not in registry: {exc}",
        )
    pg_cfg = replace(pg_cfg, table=scope.postgres_table)
    probe = ctx.postgres_probe or _default_postgres_probe
    try:
        ok, detail = probe(pg_cfg)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "postgres",
            code="POSTGRES_UNREACHABLE",
            detail=f"Postgres probe failed: {exc}",
        )
    if not ok:
        # Distinguish missing-table from connection failure via the
        # detail string the probe returned, so the typed code is right.
        code = (
            "POSTGRES_NO_TABLE"
            if "does not exist" in detail
            else "POSTGRES_UNREACHABLE"
        )
        return _fail("postgres", code=code, detail=detail)
    return _ok("postgres", detail)


def _check_llm_key(
    name: str,
    *,
    section: str,
    phase_label: str,
    load_section_fn: Callable[[str], dict[str, Any]],
    override_key_field: str | None = None,
) -> PreflightCheckResult:
    """Verify the ``api_key_env`` named in a config section resolves to
    a non-empty environment variable.

    The env var name is read from ``config.yaml`` (per-section
    ``api_key_env`` field) — same lookup path the live agents use, so a
    pass here means the live agent's ``os.environ.get(...)`` will hit.

    ``override_key_field`` lets a phase declare its own bearer source
    (e.g. ``refine.judge_api_key_env``) that takes precedence over the
    section's default ``api_key_env`` when present. Falls back to
    ``api_key_env`` when the override is unset.
    """
    try:
        raw = load_section_fn(section)
    except ConfigFileError as exc:
        return _fail(
            name,
            code="LLM_CONFIG_MISSING",
            detail=f"{section} config unreadable: {exc}",
        )
    env_var: str | None = None
    if override_key_field:
        candidate = raw.get(override_key_field)
        if isinstance(candidate, str) and candidate.strip():
            env_var = candidate
    if env_var is None:
        env_var = raw.get("api_key_env")
    if not isinstance(env_var, str) or not env_var.strip():
        return _fail(
            name,
            code="LLM_CONFIG_INVALID",
            detail=f"{section}.api_key_env is not configured.",
        )
    value = os.environ.get(env_var, "")
    if not value:
        return _fail(
            name,
            code="MISSING_LLM_KEY",
            env_var=env_var,
            detail=f"{phase_label} needs {env_var} in env or .env.",
        )
    return _ok(name, f"{env_var} is set")


def check_llm_reflection(ctx: PreflightContext) -> PreflightCheckResult:
    return _check_llm_key(
        "llm.reflection",
        section="reflection",
        phase_label="Reflection LLM",
        load_section_fn=ctx.load_section_fn,
    )


def check_llm_refine_derive(ctx: PreflightContext) -> PreflightCheckResult:
    # Refine derive + judge share the same ``api_key_env`` (refine
    # section). Two checks because the failure-surface is per-phase —
    # the operator should see exactly which step the missing key blocks.
    return _check_llm_key(
        "llm.refine.derive",
        section="refine",
        phase_label="Refine's derive step",
        load_section_fn=ctx.load_section_fn,
    )


def check_llm_refine_judge(ctx: PreflightContext) -> PreflightCheckResult:
    return _check_llm_key(
        "llm.refine.judge",
        section="refine",
        phase_label="Refine's judge step",
        load_section_fn=ctx.load_section_fn,
        override_key_field="judge_api_key_env",
    )


def check_llm_span_extractor(ctx: PreflightContext) -> PreflightCheckResult:
    return _check_llm_key(
        "llm.span_extractor",
        section="extraction",
        phase_label="Span extractor (highlighter)",
        load_section_fn=ctx.load_section_fn,
    )


CHECK_FUNCTIONS: dict[str, Callable[[PreflightContext], PreflightCheckResult]] = {
    "prompts": check_prompts,
    "users.yaml": check_users_yaml,
    "scopes.yaml": check_scopes_yaml,
    "embed_service": check_embed_service,
    "milvus.collection": check_milvus,
    "postgres": check_postgres,
    "llm.reflection": check_llm_reflection,
    "llm.refine.derive": check_llm_refine_derive,
    "llm.refine.judge": check_llm_refine_judge,
    "llm.span_extractor": check_llm_span_extractor,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_preflight(ctx: PreflightContext) -> list[PreflightCheckResult]:
    """Run every check and return the results in the canonical order.

    Network-bound checks (embed / Milvus / Postgres) execute in a
    thread pool so wall time is the slowest single check (~1-2s) rather
    than the sum. Cheap checks run inline. Results are reordered to
    match :data:`_CHECK_ORDER` so the UI's reveal sequence is stable.
    """
    network_names = {"embed_service", "milvus.collection", "postgres"}
    cheap_names = [n for n in _CHECK_ORDER if n not in network_names]
    network_check_names = [n for n in _CHECK_ORDER if n in network_names]

    results: dict[str, PreflightCheckResult] = {}

    for name in cheap_names:
        try:
            results[name] = CHECK_FUNCTIONS[name](ctx)
        except Exception as exc:  # noqa: BLE001
            results[name] = _fail(
                name,
                code="UNEXPECTED_ERROR",
                detail=f"check raised unexpectedly: {exc}",
            )

    if network_check_names:
        with ThreadPoolExecutor(
            max_workers=len(network_check_names),
            thread_name_prefix="preflight",
        ) as pool:
            futures = {
                pool.submit(CHECK_FUNCTIONS[name], ctx): name
                for name in network_check_names
            }
            for fut, name in futures.items():
                try:
                    results[name] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[name] = _fail(
                        name,
                        code="UNEXPECTED_ERROR",
                        detail=f"check raised unexpectedly: {exc}",
                    )

    return [results[name] for name in _CHECK_ORDER]


def first_failure(
    checks: list[PreflightCheckResult],
) -> PreflightCheckResult | None:
    """Return the first failing check in canonical order, or ``None``.

    Used by the endpoint to decide which check's typed-error fields
    bubble up to the top of the 4xx response body.
    """
    for r in checks:
        if r.status == "fail":
            return r
    return None


__all__ = [
    "CHECK_FUNCTIONS",
    "PreflightCheckResult",
    "PreflightContext",
    "check_embed_service",
    "check_llm_reflection",
    "check_llm_refine_derive",
    "check_llm_refine_judge",
    "check_llm_span_extractor",
    "check_milvus",
    "check_postgres",
    "check_prompts",
    "check_scopes_yaml",
    "check_users_yaml",
    "first_failure",
    "run_preflight",
]
