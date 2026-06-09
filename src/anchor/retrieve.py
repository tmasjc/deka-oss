"""Span-anchored dense k-NN over N FIT spans against the session's
Milvus collection (resolved from the scope's ``milvus_collection``
and carried on the session's ``SearchConfig``).

For each FIT span vector ``s_i``, open a pymilvus ``search_iterator``
and page through the corpus until the page's last hit crosses the
calibrated threshold ``T'``. Union the N neighbourhoods, keeping each
candidate's nearest-FIT distance (min over FITs).

Stopping rule (distance-monotonic):

* Milvus returns COSINE similarities descending, so ``d = 1 - sim``
  is ascending across pages. Once a page's last hit has ``d > T'``,
  every subsequent hit is also ``> T'``. Close the iterator.

Safety cap:

* ``max_k`` bounds cumulative hits per FIT. Hitting the cap with the
  last hit still ``≤ T'`` means the concept has more neighbours than
  the cap allowed — the FIT is recorded in ``per_fit_budget_exhausted``
  and surfaced loudly, but the partial result is kept (this is strictly
  more informative than the pre-iterator fixed-K saturation model and
  never silently undercounts).

Memory:

* Filter-at-ingest — a hit with ``d > T'`` never enters ``by_pk``.
  Keeps peak memory proportional to ``output_count`` rather than
  ``N · max_k``.

Output fields:

* Iterator pages request ``output_fields=[]``. Phase 2 is PK-only —
  downstream consumers re-fetch entity payload from Milvus by pk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from src.search.evidence import PrimaryKey

from .errors import AnchorRetrievalError
from .loader import FitChunk

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 2000
DEFAULT_MAX_K = 200_000

ProgressCallback = Callable[[str], None]

_DENSE_FIELD = "dense_embedding"
_DENSE_PARAM: dict[str, Any] = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}


@dataclass(frozen=True)
class AnchorCandidate:
    pk: PrimaryKey
    nearest_fit_pk: PrimaryKey
    nearest_fit_label: str
    nearest_fit_distance: float
    # Set of FIT pks whose calibrated radius admitted this chunk
    # (i.e., FITs i where d(chunk, span_i) ≤ T'_i / T'_out).
    # ``len(qualifying_fit_pks) == qualifying_fit_count`` always.
    # Sorted for deterministic JSON output. Empty for LOO-internal
    # candidates which never carry the anchor-frequency-gate metadata.
    qualifying_fit_pks: tuple[PrimaryKey, ...] = ()
    qualifying_fit_count: int = 0


@dataclass(frozen=True)
class PerFitPages:
    fit_pk: PrimaryKey
    fit_chunk_id: str
    pages_fetched: int
    total_hits: int
    final_kth_distance: float
    budget_exhausted: bool


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[AnchorCandidate]
    n_raw_hits: int
    n_unique: int
    batch_size: int
    max_k: int
    per_fit_pages: list[PerFitPages]
    # Pre-exclusion set of pks that passed the T' filter across all
    # FITs. Includes FIT-own pks when they passed their own filter.
    # The runner uses this to test cohort consistency ("did FIT i's
    # own chunk survive the retrieval?") without depending on the
    # FIT-pk exclusion rule that shapes ``candidates``.
    retained_pks: frozenset[PrimaryKey] = frozenset()
    # Per-pk count of distinct FIT anchors whose calibrated radius
    # admitted that pk. Includes FIT-own pks. Drives the anchor-
    # frequency gate post-filter and cohort-consistency reporting in
    # the runner. Empty dict when the result was produced by LOO.
    qualifying_count_by_pk: dict[PrimaryKey, int] = field(default_factory=dict)


def _similarity_to_distance(sim: float) -> float:
    d = 1.0 - sim
    return 0.0 if d < 0.0 else d


def _iterate_fit(
    client: MilvusClient,
    *,
    span_vector: list[float],
    collection: str,
    T_prime: float,
    batch_size: int,
    max_k: int,
    fit: FitChunk,
) -> tuple[dict[PrimaryKey, dict[str, Any]], PerFitPages, int]:
    """Page through the corpus for one FIT span.

    Returns ``(local_by_pk, per_fit_pages, n_raw_hits)``.

    * ``local_by_pk`` holds only hits that passed ``T'`` (filter-at-
      ingest). The caller merges this into the cross-FIT ``by_pk`` map
      with min-distance-wins.
    * ``per_fit_pages`` records pages/total/final_kth/budget_exhausted
      for this FIT.
    * ``n_raw_hits`` is the count of pages-worth of hits before the T'
      filter — fed back into ``RetrievalResult.n_raw_hits`` for audit.
    """
    kwargs: dict[str, Any] = {
        "collection_name": collection,
        "data": [span_vector],
        "anns_field": _DENSE_FIELD,
        "batch_size": batch_size,
        "search_params": _DENSE_PARAM,
        "output_fields": [],
    }

    try:
        iterator = client.search_iterator(**kwargs)
    except MilvusException as exc:
        raise AnchorRetrievalError(
            f"search_iterator open failed on {collection!r}: {exc}"
        ) from exc

    local: dict[PrimaryKey, dict[str, Any]] = {}
    pages_fetched = 0
    total_hits = 0
    last_d = 0.0
    stop_on_T_prime = False
    n_raw = 0

    try:
        while total_hits < max_k:
            try:
                page = iterator.next()
            except MilvusException as exc:
                raise AnchorRetrievalError(
                    f"search_iterator.next() failed on {collection!r}: {exc}"
                ) from exc
            if not page:
                # Natural exhaustion of the ANN index — no more hits.
                last_d = float("inf")
                break
            pages_fetched += 1
            for hit in page:
                n_raw += 1
                total_hits += 1
                d = _similarity_to_distance(float(hit["distance"]))
                last_d = d
                if d <= T_prime:
                    pk: PrimaryKey = hit["id"]
                    existing = local.get(pk)
                    if existing is None or d < existing["nearest_fit_distance"]:
                        local[pk] = {
                            "pk": pk,
                            "nearest_fit_pk": fit.pk,
                            "nearest_fit_label": fit.chunk_id,
                            "nearest_fit_distance": d,
                        }
                if total_hits >= max_k:
                    break
            if last_d > T_prime:
                stop_on_T_prime = True
                break
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                log.debug("search_iterator.close() raised: %s", exc)

    budget_exhausted = total_hits >= max_k and not stop_on_T_prime and last_d <= T_prime
    pages = PerFitPages(
        fit_pk=fit.pk,
        fit_chunk_id=fit.chunk_id,
        pages_fetched=pages_fetched,
        total_hits=total_hits,
        final_kth_distance=last_d,
        budget_exhausted=budget_exhausted,
    )
    return local, pages, n_raw


def _emit(progress: ProgressCallback | None, msg: str) -> None:
    log.info(msg)
    if progress is not None:
        try:
            progress(msg)
        except Exception as exc:  # noqa: BLE001
            log.debug("progress callback raised: %s", exc)


def retrieve_anchored(
    fits: list[FitChunk],
    *,
    T_primes: list[float],
    collection: str,
    client: MilvusClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_k: int = DEFAULT_MAX_K,
    progress: ProgressCallback | None = None,
) -> RetrievalResult:
    """Run span-anchored widening k-NN against ``collection``.

    One pymilvus ``search_iterator`` per FIT, sequential. Each iterator
    pages until the page's last hit has ``distance > T'_i`` (natural
    convergence, where T'_i is the FIT's own per-anchor threshold) or
    ``total_hits ≥ max_k`` (budget exhausted).

    ``T_primes`` must be parallel to ``fits`` — ``T_primes[i]`` is the
    pass threshold for ``fits[i]``.

    ``client`` is required; callers own its lifecycle.

    ``progress`` (optional) receives per-FIT status lines as the main
    pass advances — the TUI/CLI surfaces these so the user sees which
    FIT is currently paging instead of a silent 15-minute wait.

    Budget exhaustion is recorded in
    ``result.per_fit_pages[i].budget_exhausted`` and never aborts the
    run — the partial result is always returned.
    """
    if not fits:
        raise ValueError("retrieve_anchored requires at least one FIT")
    if len(T_primes) != len(fits):
        raise ValueError(
            f"T_primes length ({len(T_primes)}) must match fits length ({len(fits)})"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1 (got {batch_size})")
    if max_k < batch_size:
        raise ValueError(f"max_k ({max_k}) must be ≥ batch_size ({batch_size})")

    n = len(fits)
    fit_pks = {f.pk for f in fits}
    by_pk: dict[PrimaryKey, dict[str, Any]] = {}
    qualifying: dict[PrimaryKey, set[PrimaryKey]] = {}
    per_fit_pages: list[PerFitPages] = []
    n_raw_total = 0

    for idx, (fit, T_prime_i) in enumerate(zip(fits, T_primes), start=1):
        _emit(progress, f"Main {idx}/{n}: {fit.chunk_id} iterating...")
        local, pages, n_raw = _iterate_fit(
            client,
            span_vector=fit.span_embedding,
            collection=collection,
            T_prime=T_prime_i,
            batch_size=batch_size,
            max_k=max_k,
            fit=fit,
        )
        n_raw_total += n_raw
        per_fit_pages.append(pages)
        for pk, entry in local.items():
            existing = by_pk.get(pk)
            if (
                existing is None
                or entry["nearest_fit_distance"] < existing["nearest_fit_distance"]
            ):
                by_pk[pk] = entry
            # ``local`` already had the T' filter applied at ingest, so
            # every key here is a chunk this FIT's radius admitted.
            qual_set = qualifying.get(pk)
            if qual_set is None:
                qualifying[pk] = {fit.pk}
            else:
                qual_set.add(fit.pk)
        summary = (
            f"Main {idx}/{n}: {fit.chunk_id} done "
            f"(pages={pages.pages_fetched}, hits={pages.total_hits})"
        )
        if pages.budget_exhausted:
            summary += " ⚠ BUDGET EXHAUSTED"
        _emit(progress, summary)

    exhausted = [p for p in per_fit_pages if p.budget_exhausted]
    if exhausted:
        log.warning(
            "Phase 2 budget exhausted on %d FIT(s): %s — raise harvest.max_k",
            len(exhausted),
            ", ".join(p.fit_chunk_id for p in exhausted),
        )
        _emit(
            progress,
            f"Main pass: BUDGET EXHAUSTED on {len(exhausted)}/{n} FIT(s)",
        )

    candidates = finalise_candidates(by_pk, fit_pks=fit_pks, qualifying=qualifying)
    qualifying_count_by_pk = {pk: len(s) for pk, s in qualifying.items()}
    _emit(
        progress,
        f"Main pass complete: {len(candidates)} retained "
        f"(raw hits={n_raw_total}, unique={len(by_pk)})",
    )
    return RetrievalResult(
        candidates=candidates,
        n_raw_hits=n_raw_total,
        n_unique=len(by_pk),
        batch_size=batch_size,
        max_k=max_k,
        per_fit_pages=per_fit_pages,
        retained_pks=frozenset(by_pk.keys()),
        qualifying_count_by_pk=qualifying_count_by_pk,
    )


def finalise_candidates(
    by_pk: dict[PrimaryKey, dict[str, Any]],
    *,
    fit_pks: set[PrimaryKey],
    qualifying: dict[PrimaryKey, set[PrimaryKey]] | None = None,
) -> list[AnchorCandidate]:
    """Project the by-pk map into sorted candidate records.

    ``by_pk`` already had the T' filter applied at ingest, so this only
    drops FIT-own pks and sorts. Distance-ascending output is the
    canonical replay order.

    ``qualifying`` is the per-pk set of FIT pks whose radius admitted
    that chunk; when supplied, each emitted ``AnchorCandidate`` carries
    its qualifying-anchor set (sorted) and count for the anchor-
    frequency gate. LOO callers omit it and fall back to empty
    metadata.
    """
    qualifying = qualifying or {}
    out: list[AnchorCandidate] = []
    for pk, entry in by_pk.items():
        if pk in fit_pks:
            continue
        qual_set = qualifying.get(pk, set())
        qual_sorted: tuple[PrimaryKey, ...] = tuple(sorted(qual_set))
        out.append(
            AnchorCandidate(
                pk=entry["pk"],
                nearest_fit_pk=entry["nearest_fit_pk"],
                nearest_fit_label=entry["nearest_fit_label"],
                nearest_fit_distance=entry["nearest_fit_distance"],
                qualifying_fit_pks=qual_sorted,
                qualifying_fit_count=len(qual_sorted),
            )
        )
    out.sort(key=lambda c: c.nearest_fit_distance)
    return out
