"""Hybrid search executor with per-path provenance.

Exposes one public function :func:`run_search` that takes a query plus a
:class:`SearchConfig` and returns an :class:`EvidenceTable` ready for
human rating and reflection. Follows the four-step procedure in
``harness/schemas/evidence.md``:

1. Embed the query via BGE-M3.
2. Probe each of the three retrieval paths independently at
   ``per_path_limit`` to build a PK-keyed provenance map.
3. Run the fused ``hybrid_search`` at ``top_k`` with the configured
   ranker.
4. Assemble rows, joining fused hits against the provenance map.

The public entry point accepts an optional ``client`` argument so tests
can inject a fake ``MilvusClient``.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from pymilvus.exceptions import MilvusException

from .config import SearchConfig, with_overrides
from .embedding import get_embeddings, sparse_to_milvus
from .errors import MilvusSearchError
from .evidence import (
    CandidateRow,
    EvidenceRow,
    EvidenceTable,
    PathName,
    PrimaryKey,
    sort_paths,
)
from .fusion import rrf_merge

log = logging.getLogger(__name__)

_DENSE_FIELD = "dense_embedding"
_SPARSE_FIELD = "sparse_embedding"

_DENSE_PARAM: dict[str, Any] = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}
_SPARSE_PARAM: dict[str, Any] = {
    "metric_type": "IP",
    "params": {"drop_ratio_search": 0.2},
}

_FUSED_OUTPUT_FIELDS = [
    "sample_id",
    "counselor_id",
    "term",
    "chunk_content",
    "chunk_id",  # integer chunk sequence on the live collection
]

# Defensive ceiling on the number of PKs rendered into a single Milvus
# boolean expression. Realistic sessions stay well under 200; 5000 leaves
# several orders of magnitude of headroom before we'd approach the gRPC
# message cap. When the seen set exceeds this, the oldest-insertion-order
# overflow is dropped with a warning rather than silently truncated.
_MAX_EXPR_PKS = 5000

_MIN_CHUNK_CHARS = 50


def _char_count(content: str) -> int:
    return len(content.strip())


def _dedup_candidates_by_sample_id(
    cands: list["CandidateRow"], fused_sample_ids: set[str]
) -> list["CandidateRow"]:
    """Return candidates whose sample_id is not in fused rows and not seen yet."""
    seen = set(fused_sample_ids)
    result = []
    for c in cands:
        if c.sample_id not in seen:
            seen.add(c.sample_id)
            result.append(c)
    return result


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of an independent per-path probe (no fusion).

    Carries everything needed to either (a) feed the adaptive-config step
    or (b) hand off to a subsequent :func:`run_search` so the embed and
    per-path probes don't need to be re-run.
    """

    query: str
    config: SearchConfig
    dense_vec: list[float]
    sparse_vec: dict[int, float]
    provenance: dict[PrimaryKey, set[PathName]]
    scores_by_path: dict[PathName, dict[PrimaryKey, float]]
    stats_by_path: dict[PathName, dict[str, Any]]
    entities_by_pk: dict[PrimaryKey, dict[str, Any]] = field(default_factory=dict)
    embed_diagnostics: dict[str, Any] = field(default_factory=dict)


def probe_only(
    query: str,
    config: SearchConfig,
    *,
    client: MilvusClient | None = None,
    seen_pks: frozenset[PrimaryKey] | None = None,
) -> ProbeResult:
    """Embed the query and run each path independently — no fusion.

    Used for the Turn-0 diagnostic step. Returns the same provenance,
    scores, and per-path stats that :func:`run_search` builds internally,
    plus the embedded vectors so the caller can reuse them in a later
    :func:`run_search` call without re-embedding.

    ``seen_pks`` excludes chunks the human has already rated in prior
    turns so each path's probe reaches deeper into its ranking. Empty or
    ``None`` means no filter is applied.
    """

    if not query.strip():
        raise ValueError("query must be a non-empty string")

    dense_vec, sparse_vec, embed_diag = _embed_query(query, config)

    owns_client = False
    if client is None:
        try:
            client = MilvusClient(uri=config.milvus_uri)
        except MilvusException as exc:
            raise MilvusSearchError(
                f"Failed to connect to Milvus at {config.milvus_uri}: {exc}"
            ) from exc
        owns_client = True

    filter_expr = _build_expr(seen_pks)
    seen_count = len(seen_pks) if seen_pks else 0

    try:
        provenance, scores_by_path, stats_by_path, entities_by_pk = _probe_all_paths(
            client=client,
            collection=config.collection,
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            limit=config.per_path_limit,
            filter_expr=filter_expr,
            seen_pks_count=seen_count,
        )
    finally:
        if owns_client:
            _close_quietly(client)

    return ProbeResult(
        query=query,
        config=config,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        provenance=provenance,
        scores_by_path=scores_by_path,
        stats_by_path=stats_by_path,
        entities_by_pk=entities_by_pk,
        embed_diagnostics=embed_diag,
    )


def run_search(
    query: str,
    config: SearchConfig,
    *,
    client: MilvusClient | None = None,
    probe: ProbeResult | None = None,
    seen_pks: frozenset[PrimaryKey] | None = None,
    extract_span: Callable[[str], tuple[list[int], str]] | None = None,
    on_stage: Callable[[str, dict[str, Any]], None] | None = None,
) -> EvidenceTable:
    """Run a three-path hybrid search and assemble an evidence table.

    Parameters
    ----------
    query:
        Raw query text (Chinese for the sales-conversation corpus).
    config:
        Tunable parameters + service endpoints.
    client:
        Optional Milvus client. If ``None``, one is constructed from
        ``config.milvus_uri``. Tests inject a stub here.
    probe:
        Optional pre-computed :class:`ProbeResult`. When provided, the
        embed step and per-path probes are skipped; the caller has
        already paid those costs (e.g. during Turn-0 adaptation). The
        probe must have been run against the same query — config may
        differ in the tunable axes (the fused step uses ``config``).
    seen_pks:
        Optional set of primary keys that have already been rated by the
        human in prior turns. When provided, Milvus excludes these PKs at
        the query level (both per-path probes and the fused
        ``hybrid_search``) so RRF reaches deeper into each path's
        ranking. Empty or ``None`` means no filter is applied.
    extract_span:
        Optional callable ``chunk_content -> (span_line_indices, span_text)``
        invoked once per fused row and per per-path candidate after the
        quality gates have run. Mutates the rows in place with the
        extracted span. ``None`` disables extraction (used by legacy
        callers and tests that don't exercise the extraction path).
    """

    if not query.strip():
        raise ValueError("query must be a non-empty string")

    if on_stage is not None:
        on_stage("searching_milvus", {})

    turn_start = time.perf_counter()
    filter_expr = _build_expr(seen_pks)
    seen_count = len(seen_pks) if seen_pks else 0
    diagnostics: dict[str, Any] = {
        "query": query,
        "collection": config.collection,
        "embed_url": config.embed_url,
        "milvus_uri": config.milvus_uri,
        "ranker_params": _ranker_params(config),
        "per_path_limit": config.per_path_limit,
        "top_k": config.top_k,
        "active_paths": sorted(config.active_paths),
        "seen_set_size": seen_count,
        "milvus_search_params": {
            "dense": _DENSE_PARAM,
            "sparse": _SPARSE_PARAM,
        },
    }

    if probe is not None:
        if probe.query != query:
            raise ValueError(
                f"Probe query {probe.query!r} does not match search query {query!r}"
            )
        dense_vec = probe.dense_vec
        sparse_vec = probe.sparse_vec
        diagnostics["embedding"] = dict(probe.embed_diagnostics)
        diagnostics["embedding"]["reused_from_probe"] = True
    else:
        dense_vec, sparse_vec, embed_diag = _embed_query(query, config)
        diagnostics["embedding"] = embed_diag

    owns_client = False
    if client is None:
        try:
            client = MilvusClient(uri=config.milvus_uri)
        except MilvusException as exc:
            raise MilvusSearchError(
                f"Failed to connect to Milvus at {config.milvus_uri}: {exc}"
            ) from exc
        owns_client = True

    rows: list[EvidenceRow] = []
    per_path_candidates: dict[PathName, list[CandidateRow]] = {
        p: [] for p in ("dense", "sparse")
    }
    filtered_short_chunk = 0
    filtered_duplicate_sample = 0
    current_per_path_limit = config.per_path_limit
    attempts_log: list[dict[str, Any]] = []
    final_attempt_diag: dict[str, Any] = {}

    # Auto-retry: if the post-filter survivor count is below ``config.min_survivors``,
    # double ``per_path_limit`` and re-probe + re-fuse + re-filter. Up to 3 attempts
    # total. The probe argument is honoured on attempt 1 only (its stats are bounded
    # by the seed limit; deeper passes need fresh probe data).

    try:
        for attempt_index in range(3):
            attempt_number = attempt_index + 1
            if attempt_number > 1:
                current_per_path_limit *= 2
                if on_stage is not None:
                    on_stage(
                        "searching_retry",
                        {
                            "attempt": attempt_number,
                            "per_path_limit": current_per_path_limit,
                            "previous_survivors": len(rows),
                            "min_survivors": config.min_survivors,
                        },
                    )
                log.info(
                    "Auto-retry attempt %d/3: bumping per_path_limit %d -> %d "
                    "(previous survivors=%d, min_survivors=%d)",
                    attempt_number,
                    current_per_path_limit // 2,
                    current_per_path_limit,
                    len(rows),
                    config.min_survivors,
                )

            attempt_config = with_overrides(
                config, per_path_limit=current_per_path_limit
            )

            if attempt_number == 1 and probe is not None:
                provenance = probe.provenance
                scores_by_path = probe.scores_by_path
                probe_stats = probe.stats_by_path
                entities_by_pk = probe.entities_by_pk
            else:
                (
                    provenance,
                    scores_by_path,
                    probe_stats,
                    entities_by_pk,
                ) = _probe_all_paths(
                    client=client,
                    collection=attempt_config.collection,
                    dense_vec=dense_vec,
                    sparse_vec=sparse_vec,
                    limit=current_per_path_limit,
                    filter_expr=filter_expr,
                    seen_pks_count=seen_count,
                )

            reqs = _build_requests(
                dense_vec=dense_vec,
                sparse_vec=sparse_vec,
                limit=current_per_path_limit,
                active_paths=attempt_config.active_paths,
                filter_expr=filter_expr,
            )
            ranker = _build_ranker(attempt_config)

            # Fused over-fetch: pull max(2×top_k, per_path_limit) so the
            # auto-retry's depth bumps actually expand the fused candidate
            # pool, not just the per-path probes. Without this, hybrid_search
            # caps at 2×top_k and the same RRF winners survive each retry —
            # so the post-filter survivor count flatlines even as we double
            # per_path_limit. The 2×top_k floor is preserved so small-config
            # in-process callers see no regression.
            fused_limit = max(attempt_config.top_k * 2, current_per_path_limit)
            log.info(
                "Running fused hybrid_search (attempt %d/3, RRFRanker k=%d, "
                "top_k=%d, per_path_limit=%d, fused_limit=%d, active_paths=%s)",
                attempt_number,
                attempt_config.rrf_k,
                attempt_config.top_k,
                current_per_path_limit,
                fused_limit,
                sorted(attempt_config.active_paths),
            )
            fused_start = time.perf_counter()
            try:
                fused = client.hybrid_search(
                    collection_name=attempt_config.collection,
                    reqs=reqs,
                    ranker=ranker,
                    limit=fused_limit,
                    output_fields=_FUSED_OUTPUT_FIELDS,
                )
            except MilvusException as exc:
                raise MilvusSearchError(
                    f"hybrid_search failed on collection "
                    f"{attempt_config.collection!r}: {exc}"
                ) from exc
            fused_ms = (time.perf_counter() - fused_start) * 1000.0

            final_attempt_diag = {
                "probes": probe_stats,
                "fused": {
                    "latency_ms": round(fused_ms, 2),
                    "hit_count": len(fused[0]) if fused else 0,
                },
                "drop_previews": _build_drop_previews(
                    scores_by_path=scores_by_path,
                    entities_by_pk=entities_by_pk,
                    active_paths=attempt_config.active_paths,
                    rrf_k=attempt_config.rrf_k,
                    top_k=attempt_config.top_k,
                    per_path_limit=current_per_path_limit,
                    fused_limit=fused_limit,
                ),
            }

            rows = _assemble_rows(
                fused, provenance, scores_by_path, attempt_config.active_paths
            )

            # Gate 1: short chunk — drop chunks whose stripped content is
            # shorter than _MIN_CHUNK_CHARS (too thin to align a concept).
            # Fused rows were over-fetched to max(2×top_k, per_path_limit) to
            # absorb losses from both gates and let the retry's depth bumps
            # actually reach the filter; truncate back to top_k after filtering.
            # Compute the gate's drop count before truncating so the metric
            # reflects what the gate actually filtered, not the slice that
            # follows.
            pre_rows = len(rows)
            rows = [r for r in rows if _char_count(r.chunk_content) >= _MIN_CHUNK_CHARS]
            filtered_short_chunk = pre_rows - len(rows)
            rows = rows[: attempt_config.top_k]

            # Gate 2: sample_id uniqueness — one chunk per source transcript per turn.
            pre_dedup_rows = len(rows)
            seen_sample_ids: set[str] = set()
            unique_rows: list[EvidenceRow] = []
            for row in rows:
                if row.sample_id not in seen_sample_ids:
                    seen_sample_ids.add(row.sample_id)
                    unique_rows.append(row)
            rows = unique_rows
            filtered_duplicate_sample = pre_dedup_rows - len(rows)

            # Per-path candidates are built AFTER both gates so ``fused_pks``
            # reflects what the user will actually see, not the raw RRF
            # output. This is the candidate view's whole job: surface the
            # per-path top-3 that fusion dropped — including chunks that
            # made the raw fused list but were then killed by the gates.
            # Per-path candidates intentionally bypass Gate 1 (a path whose
            # top-3 are all short utterances would otherwise be invisible)
            # and apply only Gate 2 (avoid showing the same transcript twice).
            fused_pks = {row.pk for row in rows}
            per_path_candidates = _build_per_path_candidates(
                probe_stats, fused_pks, scores_by_path
            )
            per_path_candidates = {
                path: _dedup_candidates_by_sample_id(cands, seen_sample_ids)
                for path, cands in per_path_candidates.items()
            }

            attempts_log.append(
                {
                    "attempt": attempt_number,
                    "per_path_limit": current_per_path_limit,
                    "fused_hit_count": final_attempt_diag["fused"]["hit_count"],
                    "filtered_short_chunk": filtered_short_chunk,
                    "filtered_duplicate_sample": filtered_duplicate_sample,
                    "survivors": len(rows),
                }
            )

            if len(rows) >= config.min_survivors:
                break

        diagnostics["probes"] = final_attempt_diag.get("probes", {})
        diagnostics["fused"] = final_attempt_diag.get("fused", {})
        diagnostics["drop_previews"] = final_attempt_diag.get("drop_previews", {})
        diagnostics["per_path_limit"] = current_per_path_limit
        diagnostics["auto_retry"] = {
            "triggered": current_per_path_limit > config.per_path_limit,
            "attempts": attempts_log,
            "final_per_path_limit": current_per_path_limit,
            "min_survivors": config.min_survivors,
            "met": len(rows) >= config.min_survivors,
        }
    finally:
        if owns_client:
            _close_quietly(client)

    if diagnostics["auto_retry"]["triggered"]:
        log.info(
            "Auto-retry final: per_path_limit=%d after %d attempt(s); "
            "survivors=%d (target=%d, met=%s)",
            current_per_path_limit,
            len(attempts_log),
            len(rows),
            config.min_survivors,
            diagnostics["auto_retry"]["met"],
        )

    # Re-number ranks after both gates (final attempt's results).
    for new_rank, row in enumerate(rows, start=1):
        row.rank = new_rank

    dropped_by_extractor = 0
    if extract_span is not None:
        if on_stage is not None:
            total = len(rows) + sum(len(c) for c in per_path_candidates.values())
            on_stage("extracting_spans", {"total": total})

        # Histogram buckets are seeded for the single-vendor cap (0–3
        # lines) and grow on demand when a dual-vendor union (issue
        # #54) produces 4+ lines. ``setdefault`` keeps existing bucket
        # zeros stable for callers that read the structure shape, and
        # avoids a KeyError on never-seen-before lengths.
        span_diag: dict[str, Any] = {
            "chunks_extracted": 0,
            "empty_span_count": 0,
            "span_length_distribution": {"0": 0, "1": 0, "2": 0, "3": 0},
        }

        def _apply(chunk_content: str) -> tuple[list[int], str] | None:
            outcome = extract_span(chunk_content)
            if outcome is None:
                return None
            idx, text = outcome
            span_diag["chunks_extracted"] += 1
            if not idx:
                span_diag["empty_span_count"] += 1
            bucket = str(len(idx))
            dist = span_diag["span_length_distribution"]
            dist[bucket] = dist.get(bucket, 0) + 1
            return list(idx), text

        surviving_rows: list[EvidenceRow] = []
        for row in rows:
            outcome = _apply(row.chunk_content)
            if outcome is None:
                dropped_by_extractor += 1
                continue
            row.span_line_indices, row.span_text = outcome
            surviving_rows.append(row)
        rows = surviving_rows

        for path, cands in list(per_path_candidates.items()):
            surviving_cands: list[CandidateRow] = []
            for cand in cands:
                outcome = _apply(cand.chunk_content)
                if outcome is None:
                    dropped_by_extractor += 1
                    continue
                cand.span_line_indices, cand.span_text = outcome
                surviving_cands.append(cand)
            per_path_candidates[path] = surviving_cands

        # Re-number ranks once more after drops so display ranks stay
        # contiguous (1..N). Per-path rank_in_path is retained verbatim
        # because it annotates the probe's ordering, not the display.
        for new_rank, row in enumerate(rows, start=1):
            row.rank = new_rank

        # Optional meta from the closure — callables without these
        # attributes (e.g. test stubs) contribute nothing, and the block
        # still carries the deterministic count fields.
        for attr in (
            "model",
            "prompt_version",
            "latency_ms_total",
            "cache_hits",
            "failures",
        ):
            if hasattr(extract_span, attr):
                span_diag[attr] = getattr(extract_span, attr)
        span_diag["dropped_by_extractor"] = dropped_by_extractor
        diagnostics["span_extraction"] = span_diag

    if filtered_short_chunk:
        log.info(
            "Short-chunk gate removed %d chunk(s); %d fused + %d candidates remain.",
            filtered_short_chunk,
            len(rows),
            sum(len(v) for v in per_path_candidates.values()),
        )
    if filtered_duplicate_sample:
        log.info(
            "Sample-ID dedup gate removed %d chunk(s); %d fused + %d candidates remain.",
            filtered_duplicate_sample,
            len(rows),
            sum(len(v) for v in per_path_candidates.values()),
        )

    diagnostics["total_latency_ms"] = round(
        (time.perf_counter() - turn_start) * 1000.0, 2
    )
    log.debug("search diagnostics: %s", diagnostics)

    # Stamp the table's config with the per_path_limit actually used so
    # downstream readers (turn record, progress log, reflection prompt)
    # see the real value rather than the seed.
    table_config = (
        with_overrides(config, per_path_limit=current_per_path_limit)
        if current_per_path_limit != config.per_path_limit
        else config
    )

    return EvidenceTable(
        query=query,
        config=table_config,
        rows=rows,
        per_path_candidates=per_path_candidates,
        search_diagnostics=diagnostics,
        filtered_short_chunk=filtered_short_chunk,
        filtered_duplicate_sample=filtered_duplicate_sample,
        dropped_by_extractor=dropped_by_extractor,
    )


def _embed_query(
    query: str, config: SearchConfig
) -> tuple[list[float], dict[int, float], dict[str, Any]]:
    """Embed ``query`` via BGE-M3 and return dense + sparse + diagnostics."""

    log.info("Embedding query: %r", query)
    embed_start = time.perf_counter()
    emb = get_embeddings([query], config.embed_url, timeout=config.http_timeout)
    dense_vec = emb["dense_embeddings"][0]
    raw_sparse = emb["sparse_embeddings"][0]
    embed_ms = (time.perf_counter() - embed_start) * 1000.0
    diag: dict[str, Any] = {
        "latency_ms": round(embed_ms, 2),
        "dense_dim": len(dense_vec),
        "sparse_nnz": len(raw_sparse),
        "sparse_empty": not raw_sparse,
    }
    if not raw_sparse:
        log.warning(
            "BGE-M3 returned an empty sparse dict for the query; "
            "sparse path will contribute no hits"
        )
    sparse_vec = sparse_to_milvus(raw_sparse)
    return dense_vec, sparse_vec, diag


def _ranker_params(config: SearchConfig) -> dict[str, Any]:
    """Flatten the ranker-specific params for diagnostic logging."""
    return {"rrf_k": config.rrf_k}


def _probe_all_paths(
    *,
    client: MilvusClient,
    collection: str,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    limit: int,
    filter_expr: str | None = None,
    seen_pks_count: int = 0,
) -> tuple[
    dict[PrimaryKey, set[PathName]],
    dict[PathName, dict[PrimaryKey, float]],
    dict[PathName, dict[str, Any]],
    dict[PrimaryKey, dict[str, Any]],
]:
    """Run each retrieval path independently to collect PK-level provenance.

    Returns a 4-tuple of (provenance, scores_by_path, stats_by_path,
    entities_by_pk). ``stats_by_path`` carries per-path diagnostic info
    (hit count, score min/max/mean, top-3 pks, wall-clock latency,
    skipped flag) for the logging hook. ``entities_by_pk`` holds the
    sample_id + chunk_content for every probed PK so the drop-impact
    simulator can apply the same post-RRF gates the live pipeline
    applies. When ``filter_expr`` is supplied it is passed as
    ``filter=`` to every :meth:`MilvusClient.search` call so seen
    chunks are excluded at the server.
    """

    provenance: dict[PrimaryKey, set[PathName]] = defaultdict(set)
    scores: dict[PathName, dict[PrimaryKey, float]] = {
        "dense": {},
        "sparse": {},
    }
    stats: dict[PathName, dict[str, Any]] = {
        "dense": _empty_path_stats(),
        "sparse": _empty_path_stats(),
    }
    entities_by_pk: dict[PrimaryKey, dict[str, Any]] = {}
    for path_stats in stats.values():
        path_stats["filtered_by_seen"] = seen_pks_count

    probes: list[tuple[PathName, str, Any, dict[str, Any]]] = [
        ("dense", _DENSE_FIELD, [dense_vec], _DENSE_PARAM),
        ("sparse", _SPARSE_FIELD, [sparse_vec], _SPARSE_PARAM),
    ]

    for path_name, anns_field, data, param in probes:
        if path_name == "sparse" and not sparse_vec:
            log.debug("Skipping sparse probe — empty sparse vector")
            stats[path_name]["skipped"] = True
            continue
        probe_start = time.perf_counter()
        try:
            # Request entity content so the top-3 candidates can be
            # surfaced for human rating without a follow-up query.
            results = client.search(
                collection_name=collection,
                data=data,
                anns_field=anns_field,
                search_params=param,
                limit=limit,
                output_fields=_FUSED_OUTPUT_FIELDS,
                filter=filter_expr or "",
            )
        except MilvusException as exc:
            raise MilvusSearchError(
                f"{path_name} path probe failed on collection {collection!r}: {exc}"
            ) from exc
        stats[path_name]["latency_ms"] = round(
            (time.perf_counter() - probe_start) * 1000.0, 2
        )

        if not results:
            continue
        hits = results[0]
        path_scores: list[float] = []
        top3: list[PrimaryKey] = []
        top3_entities: list[dict[str, Any]] = []
        for i, hit in enumerate(hits):
            pk = hit["id"]  # opaque: may be int64 or VARCHAR depending on collection
            provenance[pk].add(path_name)
            score = float(hit["distance"])
            scores[path_name][pk] = score
            path_scores.append(score)
            entity = hit.get("entity") or {}
            if pk not in entities_by_pk:
                entities_by_pk[pk] = {
                    "sample_id": str(entity.get("sample_id", "")),
                    "chunk_content": str(entity.get("chunk_content", "")),
                }
            if i < 3:
                top3.append(pk)
                top3_entities.append(dict(entity))

        if path_scores:
            stats[path_name]["hit_count"] = len(path_scores)
            stats[path_name]["score_min"] = round(min(path_scores), 6)
            stats[path_name]["score_max"] = round(max(path_scores), 6)
            stats[path_name]["score_mean"] = round(
                sum(path_scores) / len(path_scores), 6
            )
            stats[path_name]["top3_pks"] = [_json_safe_pk(pk) for pk in top3]
            # ``top3_entities`` carries the full entity dict for each top-3
            # PK so :func:`_build_per_path_candidates` can construct
            # CandidateRow objects without an extra Milvus round-trip.
            # Excluded from the canonical progress log via the JSON-safe
            # subset selected in ``_top3_entities_for_log``.
            stats[path_name]["top3_entities"] = top3_entities

    return provenance, scores, stats, entities_by_pk


def _empty_path_stats() -> dict[str, Any]:
    return {
        "skipped": False,
        "hit_count": 0,
        "latency_ms": 0.0,
        "score_min": None,
        "score_max": None,
        "score_mean": None,
        "top3_pks": [],
        "top3_entities": [],
        "filtered_by_seen": 0,
    }


def _build_per_path_candidates(
    probe_stats: dict[PathName, dict[str, Any]],
    fused_pks: set[PrimaryKey],
    scores_by_path: dict[PathName, dict[PrimaryKey, float]],
) -> dict[PathName, list[CandidateRow]]:
    """Construct CandidateRow lists for each path's top-3 NOT in the fused set.

    The agent uses these to distinguish "this path's strong candidates were
    FIT but lost in fusion" from "this path is genuinely noisy for this query".
    Candidates already present in the fused top-K are skipped — they're
    already rated as part of the main evidence table.
    """

    out: dict[PathName, list[CandidateRow]] = {
        "dense": [],
        "sparse": [],
    }
    for path in ("dense", "sparse"):
        path_stats = probe_stats.get(path, {})
        top3_pks = path_stats.get("top3_pks") or []
        top3_entities = path_stats.get("top3_entities") or []
        rank_in_path = 0
        for pk_logged, entity in zip(top3_pks, top3_entities):
            rank_in_path += 1
            # ``top3_pks`` was JSON-safe-coerced; use the live PK from
            # ``scores_by_path`` keys to match the fused_pks set type.
            pk_native = _match_native_pk(pk_logged, scores_by_path[path])
            if pk_native in fused_pks:
                continue
            sample_id = str(entity.get("sample_id", ""))
            counselor_id = str(entity.get("counselor_id", ""))
            term = str(entity.get("term", ""))
            chunk_content = str(entity.get("chunk_content", ""))
            chunk_seq = entity.get("chunk_id")
            score = scores_by_path[path].get(pk_native, 0.0)
            out[path].append(
                CandidateRow(
                    path=path,
                    rank_in_path=rank_in_path,
                    pk=pk_native,
                    chunk_id=_build_chunk_id(sample_id, chunk_seq, pk_native),
                    chunk_content=chunk_content,
                    sample_id=sample_id,
                    counselor_id=counselor_id,
                    term=term,
                    score=score,
                )
            )
    return out


def _build_drop_previews(
    *,
    scores_by_path: dict[PathName, dict[PrimaryKey, float]],
    entities_by_pk: dict[PrimaryKey, dict[str, Any]],
    active_paths: frozenset[PathName],
    rrf_k: int,
    top_k: int,
    per_path_limit: int,
    fused_limit: int,
) -> dict[PathName, list[PrimaryKey]]:
    """Simulate the fused top-K when each active path is dropped in turn.

    Returns ``{path: simulated_top_k}`` for each path in ``active_paths``.
    ``simulated_top_k`` mirrors the live pipeline: RRF-fuse the remaining
    active paths up to ``fused_limit``, then apply the short-chunk and
    sample_id gates, then truncate to ``top_k``. Without the gates the
    diff against the actual top-K conflates the dropped-path effect
    with gate filtering, producing phantom losses/gains. Returns an
    empty dict when ``active_paths`` has at most one entry — nothing
    meaningful to drop.
    """
    if len(active_paths) <= 1:
        return {}

    rankings: dict[PathName, list[PrimaryKey]] = {}
    for path in active_paths:
        scored = sorted(
            scores_by_path.get(path, {}).items(),
            key=lambda kv: -kv[1],
        )[:per_path_limit]
        rankings[path] = [pk for pk, _ in scored]

    previews: dict[PathName, list[PrimaryKey]] = {}
    for dropped in active_paths:
        remaining = {p: r for p, r in rankings.items() if p != dropped}
        fused_pool = rrf_merge(remaining, rrf_k=rrf_k, top_k=fused_limit)
        previews[dropped] = _apply_post_rrf_gates(fused_pool, entities_by_pk, top_k)
    return previews


def _apply_post_rrf_gates(
    pool: list[PrimaryKey],
    entities_by_pk: dict[PrimaryKey, dict[str, Any]],
    top_k: int,
) -> list[PrimaryKey]:
    """Mirror the live pipeline's post-RRF filtering on a simulated pool.

    Applies, in order: the short-chunk gate (chunk_content length below
    ``_MIN_CHUNK_CHARS`` is dropped) and the sample_id uniqueness gate
    (one chunk per source transcript), then truncates to ``top_k``.
    PKs missing from ``entities_by_pk`` are dropped — without metadata
    we cannot judge whether they would survive the gates, and including
    them would reintroduce the asymmetry this function exists to fix.
    """
    seen_sample_ids: set[str] = set()
    out: list[PrimaryKey] = []
    for pk in pool:
        meta = entities_by_pk.get(pk)
        if meta is None:
            continue
        if _char_count(meta.get("chunk_content", "")) < _MIN_CHUNK_CHARS:
            continue
        sample_id = meta.get("sample_id", "")
        if sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)
        out.append(pk)
        if len(out) >= top_k:
            break
    return out


def _match_native_pk(
    json_safe_pk: Any, scores_for_path: dict[PrimaryKey, float]
) -> PrimaryKey:
    """Return the native PK matching ``json_safe_pk`` from the scores map.

    ``_json_safe_pk`` coerces non-int PKs to ``str``; the live scores map
    holds the original PK type. Compare both raw and coerced forms so the
    lookup works for INT64 and VARCHAR collections without branching.
    """
    if json_safe_pk in scores_for_path:
        return json_safe_pk
    for pk in scores_for_path:
        if _json_safe_pk(pk) == json_safe_pk:
            return pk
    return json_safe_pk


def _json_safe_pk(pk: PrimaryKey) -> str | int:
    """Coerce a Milvus PK to a JSON-serialisable type."""
    if isinstance(pk, int):
        return pk
    return str(pk)


def _render_pk_literal(pk: PrimaryKey) -> str:
    """Render a PK as a Milvus boolean-expression literal.

    Integers go in bare (``42``). VARCHAR PKs route through ``json.dumps``
    so embedded double quotes and backslashes are escaped correctly;
    Milvus's expression parser accepts JSON-style double-quoted strings.
    """
    if isinstance(pk, int):
        return str(pk)
    return json.dumps(pk, ensure_ascii=False)


def _build_expr(
    seen_pks: frozenset[PrimaryKey] | None,
    *,
    pk_field: str = "id",
) -> str | None:
    """Translate a seen-PK set into a Milvus exclusion expression.

    Returns ``None`` when there is nothing to filter so callers can pass the
    result straight into ``AnnSearchRequest(expr=...)`` or
    ``client.search(filter=...)`` without special-casing.
    """
    if not seen_pks:
        return None
    pks = list(seen_pks)
    if len(pks) > _MAX_EXPR_PKS:
        log.warning(
            "seen_pks size %d exceeds ceiling %d; truncating",
            len(pks),
            _MAX_EXPR_PKS,
        )
        pks = pks[:_MAX_EXPR_PKS]
    rendered = ", ".join(_render_pk_literal(pk) for pk in pks)
    return f"{pk_field} not in [{rendered}]"


def _build_requests(
    *,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
    limit: int,
    active_paths: frozenset[PathName],
    filter_expr: str | None = None,
) -> list[AnnSearchRequest]:
    """Build an AnnSearchRequest for each active retrieval path.

    Inactive paths are probed upstream for diagnostics but do not feed
    the fused ranking. ``filter_expr`` is passed as ``expr=`` to every
    request so the seen-chunk filter is applied identically across paths.
    """

    reqs: list[AnnSearchRequest] = []
    if "dense" in active_paths:
        reqs.append(
            AnnSearchRequest(
                data=[dense_vec],
                anns_field=_DENSE_FIELD,
                param=_DENSE_PARAM,
                limit=limit,
                expr=filter_expr,
            )
        )
    if "sparse" in active_paths:
        reqs.append(
            AnnSearchRequest(
                data=[sparse_vec],
                anns_field=_SPARSE_FIELD,
                param=_SPARSE_PARAM,
                limit=limit,
                expr=filter_expr,
            )
        )
    return reqs


def _build_ranker(config: SearchConfig):
    """Return the RRFRanker parameterized by ``config.rrf_k``."""
    return RRFRanker(k=config.rrf_k)


def _assemble_rows(
    fused: Any,
    provenance: dict[PrimaryKey, set[PathName]],
    scores_by_path: dict[PathName, dict[PrimaryKey, float]],
    active_paths: frozenset[PathName],
) -> list[EvidenceRow]:
    """Join fused hits against provenance + per-path scores.

    ``source_paths`` reflects only active paths — the fusion's actual
    contributors. Inactive paths are still probed, so a chunk may have
    a non-zero diagnostic score from an inactive path; that score is
    preserved in ``row.scores`` but the path is stripped from
    ``source_paths`` so the per-path breakdown reflects fusion state.
    """

    if not fused:
        return []
    hits = fused[0]

    rows: list[EvidenceRow] = []
    for rank, hit in enumerate(hits, start=1):
        pk: PrimaryKey = hit["id"]
        entity = hit.get("entity") or {}
        sample_id = str(entity.get("sample_id", ""))
        counselor_id = str(entity.get("counselor_id", ""))
        term = str(entity.get("term", ""))
        chunk_content = str(entity.get("chunk_content", ""))
        chunk_seq = entity.get("chunk_id")

        raw_path_set = provenance.get(pk, set()).copy()
        path_set: set[PathName] = {p for p in raw_path_set if p in active_paths}

        row_scores: dict[PathName, float] = {
            "dense": scores_by_path["dense"].get(pk, 0.0),
            "sparse": scores_by_path["sparse"].get(pk, 0.0),
        }

        if not path_set:
            inferred: set[PathName] = {
                name
                for name, score in row_scores.items()
                if score > 0.0 and name in active_paths
            }
            if inferred:
                log.warning(
                    "Fused hit pk=%r missing from active-path provenance; "
                    "inferring source_paths from non-zero scores: %s",
                    pk,
                    sorted(inferred),
                )
                path_set = inferred
            else:
                log.warning(
                    "Fused hit pk=%r has no active-path provenance and no "
                    "non-zero scores on active paths; source_paths will be empty",
                    pk,
                )

        rows.append(
            EvidenceRow(
                rank=rank,
                pk=pk,
                chunk_id=_build_chunk_id(sample_id, chunk_seq, pk),
                chunk_content=chunk_content,
                sample_id=sample_id,
                counselor_id=counselor_id,
                term=term,
                source_paths=sort_paths(path_set),
                scores=row_scores,
            )
        )

    return rows


def _build_chunk_id(
    sample_id: str,
    chunk_seq: Any,
    pk: PrimaryKey,
) -> str:
    """Render the display identifier for a chunk.

    Preferred source: the integer ``chunk_id`` field returned by the
    live Milvus collection (the per-transcript chunk sequence). When
    that is unavailable the primary key is used as a fallback —
    INT64 PKs zero-padded to seven digits, VARCHAR PKs verbatim.
    """

    if isinstance(chunk_seq, int):
        return f"{sample_id}_C{chunk_seq:07d}"
    if isinstance(pk, int):
        return f"{sample_id}_C{pk:07d}"
    return f"{sample_id}_C{pk}"


def _close_quietly(client: MilvusClient) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:  # noqa: BLE001 — close errors must not mask real failures
            log.debug("MilvusClient.close() raised: %s", exc)
