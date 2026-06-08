"""LOO FIT-recovery primary gate — one batched Milvus round-trip.

For each FIT i in 0..N-1:

1. Hold FIT i out; recalibrate ``T_rest = p90(span-LOO over rest)``.
   Each remaining FIT j keeps its own structural δ_j = d(span_j,
   chunk_j); its per-anchor threshold is ``T'_rest_j = T_rest + δ_j``.
2. Per-slice: check whether held-out i's chunk pk surfaced in any
   rest FIT j's T'_rest_j-filtered neighbourhood.

Algorithm: full δ vector + per-slice T_rest values computed first
(pure CPU, microseconds), then a single ``client.search`` over the
flattened ``N * (N-1)`` span vectors; slices are filtered per-
rest-FIT with that anchor's own T'. One round-trip plus O(N²) CPU
replaces N·(N-1) iterator opens — the iterator-per-slice model
regressed LOO wall time by two orders of magnitude (2500 s on
N=15 vs. <60 s for the batched shape PR #17 shipped).

Why batched ``search`` is correct here even though the main pass
needs ``search_iterator``: LOO only answers "is own pk inside the
rest set's top-K?". A fixed-K cap of ``min(max_k, 16384)`` with
``output_fields=[]`` answers that question exactly; unbounded
widening would only matter if the held-out pk sat beyond the 16384
cap of every rest span's neighbourhood, which is a pathological
shape the reference corpus never exhibits.

Why Milvus round-trip at all: span-anchored retrieval's recall of a
FIT's own chunk pk is empirical — spans and chunks live in different
embedding regimes — so the shortcut used in the chunk-anchored era
(reasoning about pairwise FIT distances only) is no longer sound.

Verdict bands are stated as fractions of 16 in the original design;
they scale linearly for any N.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from src.search.evidence import PrimaryKey

from .errors import AnchorRetrievalError
from .loader import FitChunk
from .retrieve import _similarity_to_distance, finalise_candidates
from .threshold import (
    _T_QUANTILE,
    quantile,
    span_loo_distances,
    span_to_chunk_distances,
)

log = logging.getLogger(__name__)

Verdict = Literal["HEALTHY", "FLAGGED", "FAILED"]

HEALTHY_FRACTION = 14.0 / 16.0
HARD_FAIL_FRACTION = 8.0 / 16.0

_MILVUS_PAGE_CAP = 16384
# Max query vectors per ``client.search`` call. The pymilvus gRPC
# channel caps response payloads at 256 MiB; at limit=16384 and ~30 B
# per (pk, distance) hit, ~200 vectors saturates the cap. 100 keeps
# us under ~50 MB with room for variable-length PKs and scales to
# any N — the LOO call splits ``span_vectors`` into chunks of this
# size and concatenates the responses.
_LOO_VECTOR_CHUNK = 100
_DENSE_FIELD = "dense_embedding"
_DENSE_PARAM: dict[str, Any] = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class LooPerFit:
    fit_pk: PrimaryKey
    fit_chunk_id: str
    recovered: bool
    recalibrated_T: float
    rank_of_own_pk: int | None
    distance_of_own_pk: float | None


@dataclass(frozen=True)
class RecoveryResult:
    recovered: int
    total: int
    verdict: Verdict
    per_fit: list[LooPerFit]
    missed_fits: list[LooPerFit]


def _classify(recovered: int, total: int) -> Verdict:
    fraction = recovered / total
    if fraction >= HEALTHY_FRACTION:
        return "HEALTHY"
    if fraction < HARD_FAIL_FRACTION:
        return "FAILED"
    return "FLAGGED"


def _emit(progress: ProgressCallback | None, msg: str) -> None:
    log.info(msg)
    if progress is not None:
        try:
            progress(msg)
        except Exception as exc:  # noqa: BLE001
            log.debug("progress callback raised: %s", exc)


def loo_fit_recovery(
    fits: list[FitChunk],
    *,
    collection: str,
    client: MilvusClient,
    max_k: int,
    # ``batch_size`` is accepted for signature symmetry with the main
    # pass but is not used — a batched ``client.search`` has no paging
    # concept; ``limit`` alone bounds the response size.
    batch_size: int | None = None,
    progress: ProgressCallback | None = None,
) -> RecoveryResult:
    """Run the batched Milvus round-trip LOO over every FIT.

    One ``client.search`` over the flattened ``N * (N-1)`` span
    vectors replaces N·(N-1) iterator opens. ``output_fields=[]`` is
    mandatory — at production N the full-entity payload overflows
    pymilvus's 256 MB gRPC cap (see PR #17 RESOURCE_EXHAUSTED fix).

    ``limit`` is clamped to ``min(max_k, MAX_BATCH_SIZE=16384)``. That
    cap is correctness-neutral for LOO: if a held-out pk sits beyond
    the top-16384 of every rest span's neighbourhood filtered by T',
    it is genuinely a weak recovery signal — the verdict it produces
    (missed) matches the intent of the gate.
    """
    n = len(fits)
    if n < 2:
        raise ValueError(f"LOO recovery needs at least 2 FITs; got {n}")

    loo_limit = min(max_k, _MILVUS_PAGE_CAP)

    # δ_j is structural (span-to-own-chunk distance per FIT) — doesn't
    # depend on which FIT is held out. Compute once over all N.
    full_deltas = span_to_chunk_distances(
        [f.span_embedding for f in fits],
        [f.chunk_embedding for f in fits],
    )

    rests: list[list[FitChunk]] = []
    T_rests: list[float] = []
    rest_deltas_per_slice: list[list[float]] = []
    span_vectors: list[list[float]] = []
    offsets: list[tuple[int, int]] = []

    for i in range(n):
        rest = fits[:i] + fits[i + 1 :]
        rest_span_loo = span_loo_distances([f.span_embedding for f in rest])
        T_rest = quantile(rest_span_loo, _T_QUANTILE)
        rest_deltas = [full_deltas[j] for j in range(n) if j != i]

        rests.append(rest)
        T_rests.append(T_rest)
        rest_deltas_per_slice.append(rest_deltas)
        start = len(span_vectors)
        span_vectors.extend(f.span_embedding for f in rest)
        offsets.append((start, len(span_vectors)))

    n_chunks = (len(span_vectors) + _LOO_VECTOR_CHUNK - 1) // _LOO_VECTOR_CHUNK
    _emit(
        progress,
        f"LOO: batched search over {len(span_vectors)} vectors "
        f"({n} slices × {n - 1}), limit={loo_limit}, "
        f"{n_chunks} chunk(s) of ≤{_LOO_VECTOR_CHUNK}...",
    )

    batched_hits: list[Any] = []
    for chunk_start in range(0, len(span_vectors), _LOO_VECTOR_CHUNK):
        chunk = span_vectors[chunk_start : chunk_start + _LOO_VECTOR_CHUNK]
        kwargs: dict[str, Any] = {
            "collection_name": collection,
            "data": chunk,
            "anns_field": _DENSE_FIELD,
            "search_params": _DENSE_PARAM,
            "limit": loo_limit,
            "output_fields": [],
        }
        try:
            chunk_hits = client.search(**kwargs)
        except MilvusException as exc:
            raise AnchorRetrievalError(
                f"LOO batched search failed on collection {collection!r} "
                f"(chunk offset {chunk_start}): {exc}"
            ) from exc
        batched_hits.extend(chunk_hits)

    per_fit: list[LooPerFit] = []

    for i, held_out in enumerate(fits):
        rest = rests[i]
        T_rest = T_rests[i]
        rest_deltas = rest_deltas_per_slice[i]
        start, end = offsets[i]
        hits_i = batched_hits[start:end]

        by_pk: dict[PrimaryKey, dict[str, Any]] = {}
        for fit, delta_j, hits in zip(rest, rest_deltas, hits_i):
            T_prime_j = T_rest + delta_j
            for hit in hits:
                d = _similarity_to_distance(float(hit["distance"]))
                if d > T_prime_j:
                    # Per-rest-FIT T' filter — each rest anchor carries
                    # its own δ_j, so each slice admits hits relative to
                    # that anchor's own threshold.
                    continue
                pk: PrimaryKey = hit["id"]
                existing = by_pk.get(pk)
                if existing is None or d < existing["nearest_fit_distance"]:
                    by_pk[pk] = {
                        "pk": pk,
                        "nearest_fit_pk": fit.pk,
                        "nearest_fit_label": fit.chunk_id,
                        "nearest_fit_distance": d,
                    }

        # fit_pks is the rest set only — held-out must stay eligible.
        # LOO recovery intentionally tests under f=1 semantics (own pk
        # appears in *any* rest FIT's neighbourhood) regardless of
        # ``harvest.anchor_frequency_gate``: the verdict measures
        # calibration health, decoupled from the cohort-tightening gate
        # the main pass applies (issue #22).
        candidates = finalise_candidates(
            by_pk,
            fit_pks={f.pk for f in rest},
        )
        own = next((c for c in candidates if c.pk == held_out.pk), None)
        if own is None:
            per_fit.append(
                LooPerFit(
                    fit_pk=held_out.pk,
                    fit_chunk_id=held_out.chunk_id,
                    recovered=False,
                    recalibrated_T=T_rest,
                    rank_of_own_pk=None,
                    distance_of_own_pk=None,
                )
            )
            _emit(progress, f"LOO {i + 1}/{n}: {held_out.chunk_id} MISSED")
        else:
            rank = candidates.index(own) + 1
            per_fit.append(
                LooPerFit(
                    fit_pk=held_out.pk,
                    fit_chunk_id=held_out.chunk_id,
                    recovered=True,
                    recalibrated_T=T_rest,
                    rank_of_own_pk=rank,
                    distance_of_own_pk=own.nearest_fit_distance,
                )
            )
            _emit(
                progress,
                f"LOO {i + 1}/{n}: {held_out.chunk_id} recovered "
                f"(rank {rank}, d={own.nearest_fit_distance:.4f})",
            )

    recovered = sum(1 for p in per_fit if p.recovered)
    verdict = _classify(recovered, n)
    _emit(progress, f"LOO complete: {recovered}/{n} recovered ({verdict})")
    return RecoveryResult(
        recovered=recovered,
        total=n,
        verdict=verdict,
        per_fit=per_fit,
        missed_fits=[p for p in per_fit if not p.recovered],
    )


def count_not_fit_intrusions(
    candidate_pks,
    not_fit_pks: frozenset,
) -> int:
    """Soft secondary gate — how many retained candidates were
    rated NOT_FIT during Phase 1."""
    return sum(1 for pk in candidate_pks if pk in not_fit_pks)
