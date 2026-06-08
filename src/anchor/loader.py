"""Load FIT-rated chunks from a converged session and assemble
span-anchored retrieval inputs.

Pipeline:

1. Parse the progress log (``runs/{id}.jsonl``) to identify FIT rows
   and their spans, dedup-by-chunk_id across turns.
2. Read the details sidecar (``runs/{id}.details.jsonl``) for the
   ``collection``, ``embed_url``, and recorded ``embed_model_id``.
3. Sanity-cross-check against the span cache (``.span_cache.jsonl``);
   drift logs a warning.
4. Verify the embed service's live ``model_id`` matches the one in
   the details sidecar. Hard-fail on mismatch.
5. Fresh-embed each FIT's ``span_text`` via the embed service.
6. Fetch each FIT's chunk ``dense_embedding`` from Milvus by pk.
7. Build and return an ``AnchorInputs`` with everything downstream
   needs.

Empty spans, missing details fields, model mismatch, and missing pks
are all hard errors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from src.replay.loader import ReplaySession, load_session
from src.replay.metrics import (
    _cumulative_discard_pks,
    _cumulative_fit_pks,
    _cumulative_not_fit_pks,
)
from src.search.config import SearchConfig, load_default_config
from src.search.embedding import get_embeddings
from src.search.errors import EmbeddingServiceError
from src.search.evidence import PrimaryKey

from .config import load_harvest_config
from .errors import AnchorLoadError, AnchorValidationError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitChunk:
    """One Phase-1 FIT, fully hydrated for span-anchored retrieval."""

    pk: PrimaryKey
    chunk_id: str
    chunk_content: str
    sample_id: str
    counselor_id: str
    term: str
    span_text: str
    span_line_indices: list[int]
    span_embedding: list[float]
    chunk_embedding: list[float]


@dataclass(frozen=True)
class AnchorInputs:
    """Everything the Phase-2 run needs about the source session."""

    session_id: str
    query: str
    collection: str
    embed_url: str
    embed_model_id: str
    milvus_index_type: str
    milvus_index_params: dict[str, Any]
    fits: list[FitChunk]
    not_fit_pks: frozenset[PrimaryKey]
    # Pks the operator invalidated in Phase 1 (issue #46) — Phase 2
    # filters them out of the harvest output so DISCARDs never reach
    # Phase 3.
    discard_pks: frozenset[PrimaryKey]


# Indirection seams for monkeypatching in tests.
_get_embeddings: Callable[..., dict] = get_embeddings


def _read_embed_model_id(embed_url: str, *, timeout: int = 5) -> str:
    """Ask the embed service for its current model identifier.

    If the service exposes a ``/model`` endpoint that returns
    ``{"model_id": "..."}``, return that id. If the endpoint is
    missing (404), return an empty string so the caller can decide
    whether to fail or proceed without a live check; any other
    network failure is treated as a hard error. Wrapped for
    monkeypatching.
    """
    url = f"{embed_url}/model"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 404:
            log.warning(
                "Embed service at %s does not expose /model — "
                "skipping live model-version check",
                embed_url,
            )
            return ""
        resp.raise_for_status()
        return str(resp.json().get("model_id", ""))
    except requests.RequestException as exc:
        raise AnchorLoadError(
            f"Embed service {url} unreachable for model-version check: {exc}"
        ) from exc


def _fit_rows_deduped(session: ReplaySession) -> list[dict[str, Any]]:
    """Walk every turn and collect FIT rows, keyed by primary key.

    Last-write-wins: later turns override earlier ones. A pk flip-
    flopping between FIT and NOT_FIT across turns logs a warning but
    isn't fatal.
    """
    by_pk: dict[PrimaryKey, dict[str, Any]] = {}
    flipped: set[str] = set()
    for turn in session.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "FIT":
                prev = by_pk.get(row.pk)
                if prev is not None and prev.get("rating") == "NOT_FIT":
                    flipped.add(str(row.pk))
                by_pk[row.pk] = {
                    "pk": row.pk,
                    "chunk_id": row.chunk_id,
                    "chunk_content": row.chunk_content,
                    "sample_id": row.sample_id,
                    "counselor_id": row.counselor_id,
                    "term": row.term,
                    "span_text": row.span_text,
                    "span_line_indices": list(row.span_line_indices),
                    "rating": "FIT",
                }
            elif row.rating == "NOT_FIT":
                prev = by_pk.get(row.pk)
                if prev is not None and prev.get("rating") == "FIT":
                    flipped.add(str(row.pk))
                    by_pk[row.pk] = {**prev, "rating": "NOT_FIT"}
        for cands in turn.evidence_table.per_path_candidates.values():
            for cand in cands:
                if cand.rating == "FIT":
                    prev = by_pk.get(cand.pk)
                    if prev is not None and prev.get("rating") == "NOT_FIT":
                        flipped.add(str(cand.pk))
                    by_pk[cand.pk] = {
                        "pk": cand.pk,
                        "chunk_id": cand.chunk_id,
                        "chunk_content": cand.chunk_content,
                        "sample_id": cand.sample_id,
                        "counselor_id": cand.counselor_id,
                        "term": cand.term,
                        "span_text": cand.span_text,
                        "span_line_indices": list(cand.span_line_indices),
                        "rating": "FIT",
                    }
    if flipped:
        log.warning(
            "Pks flipped FIT/NOT_FIT across turns: %s (last-write-wins)",
            sorted(flipped),
        )
    return [r for r in by_pk.values() if r["rating"] == "FIT"]


def _read_details_search_block(runs_dir: Path, session_id: str) -> dict[str, Any]:
    path = runs_dir / f"{session_id}.details.jsonl"
    if not path.exists():
        raise AnchorLoadError(f"Details sidecar missing: {path}")
    first: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnchorLoadError(f"{path} line malformed: {exc}") from exc
        if "search" in rec:
            first = rec["search"]
            break
    if first is None:
        raise AnchorLoadError(f"No ``search`` block found in {path}")
    return first


def _cross_check_span_cache(
    runs_dir: Path, session_id: str, fits: list[dict[str, Any]]
) -> None:
    path = runs_dir / f"{session_id}.span_cache.jsonl"
    if not path.exists():
        log.warning("Span cache %s not found — skipping drift check", path)
        return
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = entry.get("chunk_id")
        if cid:
            cache[str(cid)] = entry
    for fit in fits:
        entry = cache.get(fit["chunk_id"])
        if entry is None:
            continue
        if entry.get("span_text") != fit["span_text"]:
            log.warning(
                "Span cache/log drift on chunk %s (log wins)", fit["chunk_id"]
            )
        if list(entry.get("span_line_indices") or []) != fit["span_line_indices"]:
            log.warning(
                "Span indices drift on chunk %s (log wins)", fit["chunk_id"]
            )


def _render_pk_literal(pk: PrimaryKey) -> str:
    if isinstance(pk, int):
        return str(pk)
    return json.dumps(pk, ensure_ascii=False)


def _fetch_chunk_embeddings(
    pks: list[PrimaryKey],
    *,
    collection: str,
    client: MilvusClient,
    pk_field: str = "id",
) -> dict[PrimaryKey, list[float]]:
    if not pks:
        return {}
    rendered = ", ".join(_render_pk_literal(pk) for pk in pks)
    expr = f"{pk_field} in [{rendered}]"
    try:
        rows = client.query(
            collection_name=collection,
            filter=expr,
            output_fields=[pk_field, "dense_embedding"],
        )
    except MilvusException as exc:
        raise AnchorLoadError(
            f"Milvus query for FIT chunk embeddings failed: {exc}"
        ) from exc
    out: dict[PrimaryKey, list[float]] = {}
    for entry in rows:
        pk = entry.get(pk_field)
        vec = entry.get("dense_embedding")
        if pk is None or vec is None:
            continue
        out[pk] = [float(x) for x in vec]
    return out


def _read_index_info(
    client: MilvusClient, collection: str
) -> tuple[str, dict[str, Any]]:
    """Best-effort probe of the Milvus index type/params on the target
    collection. Returns ``("UNKNOWN", {})`` on failure — a reproducibility
    hedge, not a gate, so we never hard-fail here. Every fallback path
    logs a warning so the operator notices the meta sidecar is shipping
    placeholder values.

    pymilvus ``MilvusClient.describe_index(collection_name, index_name,
    ...)`` requires ``index_name`` — calling without it raises a
    TypeError. We resolve the name via ``list_indexes`` first, preferring
    the ``dense_embedding`` index because that is the field Phase 2
    actually queries; if that name is not present we fall back to the
    first listed index.
    """
    describe = getattr(client, "describe_index", None)
    list_indexes = getattr(client, "list_indexes", None)
    if describe is None or list_indexes is None:
        log.warning(
            "Milvus client missing describe_index/list_indexes on %s; "
            "phase2.meta will record milvus_index_type=UNKNOWN.",
            collection,
        )
        return ("UNKNOWN", {})

    try:
        names = list_indexes(collection_name=collection)
    except Exception as exc:  # noqa: BLE001 - best effort
        log.warning("list_indexes failed on %s: %s", collection, exc)
        return ("UNKNOWN", {})

    if not names:
        log.warning(
            "list_indexes on %s returned no indexes; "
            "phase2.meta will record milvus_index_type=UNKNOWN.",
            collection,
        )
        return ("UNKNOWN", {})

    target = "dense_embedding" if "dense_embedding" in names else names[0]

    try:
        info = describe(collection_name=collection, index_name=target)
    except Exception as exc:  # noqa: BLE001 - best effort
        log.warning("describe_index failed on %s: %s", collection, exc)
        return ("UNKNOWN", {})
    if isinstance(info, list) and info:
        info = info[0]
    if not isinstance(info, dict):
        log.warning(
            "describe_index on %s returned %s; "
            "phase2.meta will record milvus_index_type=UNKNOWN.",
            collection,
            type(info).__name__,
        )
        return ("UNKNOWN", {})
    return (str(info.get("index_type", "UNKNOWN")), dict(info.get("params") or {}))


def load_anchor_inputs(
    session_target: str | Path,
    *,
    runs_dir: Path | None = None,
    config: SearchConfig | None = None,
    milvus_client: MilvusClient | None = None,
    embed_timeout: int = 10,
    allow_unconverged: bool = False,
    harvest_overrides: dict[str, Any] | None = None,
) -> AnchorInputs:
    """Top-level loader: from a session id to a ready ``AnchorInputs``.

    Aborts with :class:`AnchorValidationError` when the source session
    has not converged on Phase 1's triple gate (latest P@K, cumulative
    FIT count, cumulative NOT_FIT count), unless ``allow_unconverged``
    is set. The TUI hand-off only reaches this loader after the same
    gate fires; the CLI relies on this check.

    ``harvest_overrides`` carries the session's
    ``<sid>.overrides.json`` ``harvest`` block so the gate reads the
    same thresholds Phase 1's ``state.is_converged`` snapshotted at
    session creation. Without forwarding, a session that converges
    under a relaxed override would be re-rejected here against the
    YAML default.
    """
    session = load_session(session_target, runs_dir=runs_dir)
    runs_dir_path = Path(runs_dir) if runs_dir is not None else Path("runs")

    fit_rows = _fit_rows_deduped(session)
    if not fit_rows:
        # Catch the "0 FITs" shape with the more specific load error
        # before the convergence check would generically blame missing
        # FITs as "not converged".
        raise AnchorLoadError(
            f"Session {session.session_id} has no FIT-rated chunks; "
            "anchor retrieval needs at least 2 FITs."
        )

    if not allow_unconverged:
        # Read live harvest config rather than the import-frozen
        # constants in ``replay.metrics`` so tests / runtime overrides
        # of ``load_harvest_config`` flow through. Pass
        # ``session_overrides`` so per-session ``harvest`` overrides
        # (e.g. a relaxed ``precision_at_k``) reach the gate.
        _harvest = load_harvest_config(session_overrides=harvest_overrides)
        latest_precision = (
            session.turns[-1].precision if session.turns else 0.0
        )
        cumulative_fit = len(_cumulative_fit_pks(session))
        cumulative_not_fit = len(_cumulative_not_fit_pks(session))
        if (
            latest_precision < _harvest.precision_at_k
            or cumulative_fit < _harvest.min_fit
            or cumulative_not_fit < _harvest.min_not_fit
        ):
            raise AnchorValidationError(
                f"Session {session.session_id} has not converged: "
                f"latest P@K={latest_precision:.2f} (need >= "
                f"{_harvest.precision_at_k:.2f}), cumulative unique FIT "
                f"pks={cumulative_fit} (need >= {_harvest.min_fit}), "
                f"cumulative unique NOT_FIT pks={cumulative_not_fit} "
                f"(need >= {_harvest.min_not_fit}). "
                "Continue Phase 1 tuning or pass --allow-unconverged "
                "to override. No output written."
            )

    # Drop FIT rows without a usable span. A missing span on a single
    # row is a Phase-1 extraction miss, not a reason to abort the
    # whole Phase-2 run — the remaining FITs can still calibrate T'.
    # Hard-fail only if we fall below the 2-FIT floor needed by the
    # calibration step.
    usable: list[dict[str, Any]] = []
    dropped: list[str] = []
    for row in fit_rows:
        if not row["span_text"] or not row["span_line_indices"]:
            dropped.append(row["chunk_id"])
            continue
        usable.append(row)
    if dropped:
        log.warning(
            "Dropping %d FIT(s) with empty span fields from the "
            "Phase 2 anchor set: %s",
            len(dropped),
            dropped,
        )
    if len(usable) < 2:
        raise AnchorLoadError(
            f"Session {session.session_id} has only {len(usable)} "
            "FIT(s) with usable span fields after dropping "
            f"{len(dropped)} missing-span row(s) "
            f"({dropped[:5]}{' ...' if len(dropped) > 5 else ''}); "
            "Phase 2 calibration needs at least 2."
        )
    fit_rows = usable

    _cross_check_span_cache(runs_dir_path, session.session_id, fit_rows)

    details = _read_details_search_block(runs_dir_path, session.session_id)
    collection = str(details.get("collection") or "")
    embed_url = str(details.get("embed_url") or "")
    recorded_model_id = details.get("embed_model_id")  # may be None on older logs
    if not collection:
        raise AnchorLoadError(
            f"Details sidecar for {session.session_id} missing `collection`"
        )
    if not embed_url:
        raise AnchorLoadError(
            f"Details sidecar for {session.session_id} missing `embed_url`"
        )

    # Model-version check. Hard-fail on observed drift; if the service
    # does not expose /model (``live_model_id`` empty), log and proceed.
    live_model_id = _read_embed_model_id(embed_url, timeout=embed_timeout)
    if recorded_model_id and live_model_id and live_model_id != recorded_model_id:
        raise AnchorLoadError(
            f"Embed-model drift: details recorded "
            f"{recorded_model_id!r}, service now reports {live_model_id!r}. "
            "Re-run Phase 1 or roll the service back; Phase 2 cannot "
            "measure δ across embedding regimes."
        )
    if recorded_model_id and not live_model_id:
        log.warning(
            "Embed service does not expose /model; cannot verify "
            "recorded embed_model_id=%r. Proceeding on the assumption "
            "the service has not changed since Phase 1.",
            recorded_model_id,
        )
    embed_model_id = live_model_id or str(recorded_model_id or "")
    if not embed_model_id:
        log.warning(
            "Embed model id is unknown — neither the live service nor "
            "the details sidecar reported one; phase2.meta will record "
            "embed_model_id=''. This breaks reproducibility tracking; "
            "ensure /model is exposed and the Phase 1 details sidecar "
            "carries embed_model_id."
        )

    # Fresh-embed span texts.
    span_texts = [row["span_text"] for row in fit_rows]
    try:
        embed_resp = _get_embeddings(
            span_texts, embed_url, timeout=embed_timeout
        )
    except EmbeddingServiceError as exc:
        raise AnchorLoadError(
            f"Failed to embed FIT spans via {embed_url}: {exc}"
        ) from exc
    # BGE-M3's /embed-all returns ``dense_embeddings``; legacy/mocks
    # may use ``dense`` — accept either.
    span_dense = embed_resp.get("dense_embeddings")
    if span_dense is None:
        span_dense = embed_resp.get("dense")
    if not isinstance(span_dense, list) or len(span_dense) != len(fit_rows):
        raise AnchorLoadError(
            f"Embed service returned "
            f"{len(span_dense) if isinstance(span_dense, list) else 'non-list'} "
            f"vectors for {len(fit_rows)} spans"
        )

    # Resolve Milvus client / config.
    if config is None:
        config = load_default_config()
    owns_client = False
    client = milvus_client
    if client is None:
        try:
            client = MilvusClient(uri=config.milvus_uri)
        except MilvusException as exc:
            raise AnchorLoadError(
                f"Failed to connect to Milvus at {config.milvus_uri}: {exc}"
            ) from exc
        owns_client = True

    try:
        pk_list = [row["pk"] for row in fit_rows]
        chunk_embeddings = _fetch_chunk_embeddings(
            pk_list, collection=collection, client=client
        )
        missing = [pk for pk in pk_list if pk not in chunk_embeddings]
        if missing:
            raise AnchorLoadError(
                "Milvus did not return ``dense_embedding`` for "
                f"{len(missing)} FIT pk(s): {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}"
            )

        index_type, index_params = _read_index_info(client, collection)
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("MilvusClient.close() raised: %s", exc)

    fits: list[FitChunk] = []
    for row, dense in zip(fit_rows, span_dense):
        fits.append(
            FitChunk(
                pk=row["pk"],
                chunk_id=row["chunk_id"],
                chunk_content=row["chunk_content"],
                sample_id=row["sample_id"],
                counselor_id=row["counselor_id"],
                term=row["term"],
                span_text=row["span_text"],
                span_line_indices=row["span_line_indices"],
                span_embedding=[float(x) for x in dense],
                chunk_embedding=chunk_embeddings[row["pk"]],
            )
        )

    query = session.turns[0].query if session.turns else ""
    return AnchorInputs(
        session_id=session.session_id,
        query=query,
        collection=collection,
        embed_url=embed_url,
        embed_model_id=embed_model_id,
        milvus_index_type=index_type,
        milvus_index_params=index_params,
        fits=fits,
        not_fit_pks=frozenset(_cumulative_not_fit_pks(session)),
        discard_pks=frozenset(_cumulative_discard_pks(session)),
    )
