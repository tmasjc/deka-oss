"""Unit tests for src.anchor.retrieve — iterator widening + T' filter."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable

import pytest

from src.anchor.errors import AnchorRetrievalError
from src.anchor.loader import FitChunk
from src.anchor.retrieve import (
    AnchorCandidate,
    RetrievalResult,
    retrieve_anchored,
)


# ------------------------------- helpers -------------------------------


def _fit(pk: str, span_vec: list[float]) -> FitChunk:
    # Loader builds FitChunk with entity fields — we keep them for
    # construction but retrieve.py never reads them anymore.
    return FitChunk(
        pk=pk,
        chunk_id=f"cid-{pk}",
        chunk_content="",
        sample_id="",
        counselor_id="",
        term="",
        span_text=f"span-{pk}",
        span_line_indices=[0],
        span_embedding=span_vec,
        chunk_embedding=span_vec,
    )


def _hit(pk: str, sim: float) -> dict:
    """PK-only hit — iterator pages request ``output_fields=[]``, so
    entity dicts are absent."""
    return {"id": pk, "distance": sim}


class _FakeIterator:
    """Serves scripted pages, one ``next()`` per list entry."""

    def __init__(self, pages: list[list[dict]]):
        self._pages = list(pages)
        self.closed = False
        self.next_calls = 0

    def next(self) -> list[dict]:  # noqa: A003  (mirrors pymilvus API)
        self.next_calls += 1
        if not self._pages:
            return []
        return self._pages.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeMilvus:
    """Captures search_iterator kwargs; serves pages via ``page_fn``.

    ``page_fn(span_vector, collection, batch_size, filter) -> list[list[dict]]``
    """

    def __init__(self, page_fn: Callable[..., list[list[dict]]]):
        self._page_fn = page_fn
        self.calls: list[dict[str, Any]] = []
        self.iterators: list[_FakeIterator] = []

    def search_iterator(
        self,
        *,
        collection_name: str,
        data,
        anns_field: str,
        batch_size: int,
        search_params,
        output_fields,
        filter: str | None = None,
    ):
        assert anns_field == "dense_embedding"
        assert len(data) == 1, "search_iterator opens one iterator per vector"
        self.calls.append(
            {
                "collection": collection_name,
                "batch_size": batch_size,
                "output_fields": list(output_fields),
                "filter": filter,
            }
        )
        pages = self._page_fn(data[0], collection_name, batch_size, filter)
        it = _FakeIterator(pages)
        self.iterators.append(it)
        return it

    def close(self):
        pass


# ---------------------------- shape tests -----------------------------


def test_anchor_candidate_is_pk_only():
    """Phase 2 output is PK-only — candidates must not carry entity
    fields that downstream would re-fetch from Milvus anyway. The
    qualifying-anchor metadata added by the anchor-frequency gate is
    pk/count-only, not entity payload."""
    names = {f.name for f in fields(AnchorCandidate)}
    assert names == {
        "pk",
        "nearest_fit_pk",
        "nearest_fit_label",
        "nearest_fit_distance",
        "qualifying_fit_pks",
        "qualifying_fit_count",
    }


def test_iterator_output_fields_are_empty():
    """Iterator pages never ask for entity payload."""
    fits = [_fit("fA", [1.0, 0.0])]

    def pages(q, coll, bs, flt):
        return [[]]

    client = _FakeMilvus(pages)
    retrieve_anchored(
        fits, T_primes=[0.5], collection="c", client=client,
        batch_size=500, max_k=500,
    )
    assert client.calls[0]["output_fields"] == []


def test_T_primes_length_must_match_fits():
    fits = [_fit("fA", [1.0, 0.0]), _fit("fB", [0.0, 1.0])]
    client = _FakeMilvus(lambda *_a, **_k: [])
    with pytest.raises(ValueError, match="T_primes length"):
        retrieve_anchored(
            fits, T_primes=[0.5], collection="c", client=client,
            batch_size=10, max_k=10,
        )


# -------------------------- filter-at-ingest ---------------------------


def test_retrieve_filters_by_T_prime():
    fits = [
        _fit("fA", [1.0, 0.0]),
        _fit("fB", [0.0, 1.0]),
    ]

    def pages(q, coll, bs, flt):
        if q == [1.0, 0.0]:
            # One page: cand1 and cand2 inside T', cand3 outside.
            # The page's last hit (cand3) crosses T' → iterator stops.
            return [
                [
                    _hit("cand1", sim=0.95),  # d=0.05
                    _hit("cand2", sim=0.80),  # d=0.20
                    _hit("cand3", sim=0.50),  # d=0.50 > T'=0.25
                ]
            ]
        return [[_hit("cand4", sim=0.90)]]  # d=0.10

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.25, 0.25], collection="c", client=client,
        batch_size=500, max_k=500,
    )

    assert isinstance(result, RetrievalResult)
    retained_pks = {c.pk for c in result.candidates}
    assert retained_pks == {"cand1", "cand2", "cand4"}
    distances = [c.nearest_fit_distance for c in result.candidates]
    assert distances == sorted(distances)
    # No FIT hit its budget cap.
    assert not any(p.budget_exhausted for p in result.per_fit_pages)


def test_filter_at_ingest_never_materialises_over_threshold():
    """A hit with d > T' must not appear in by_pk — this is the memory
    optimization that lets max_k=200k be safe."""
    fits = [_fit("fA", [1.0, 0.0])]

    def pages(q, coll, bs, flt):
        return [
            [_hit("keep", sim=0.99)],       # d=0.01
            [_hit("drop", sim=0.01)],       # d=0.99  > T'
        ]

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.10], collection="c", client=client,
        batch_size=10, max_k=100,
    )

    assert [c.pk for c in result.candidates] == ["keep"]


def test_retrieve_union_dedupe_keeps_minimum():
    fits = [_fit("fA", [1.0, 0.0]), _fit("fB", [0.0, 1.0])]

    def pages(q, coll, bs, flt):
        if q == [1.0, 0.0]:
            return [[_hit("shared", sim=0.80)]]  # d=0.20
        return [[_hit("shared", sim=0.95)]]      # d=0.05

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.30, 0.30], collection="c", client=client,
        batch_size=10, max_k=10,
    )

    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.pk == "shared"
    assert c.nearest_fit_distance == pytest.approx(0.05)
    assert c.nearest_fit_pk == "fB"
    assert c.nearest_fit_label == "cid-fB"


def test_per_fit_T_prime_applied_independently():
    """Each FIT's iterator uses its own T'_i. fA has a tight T'=0.05
    (cand admitted at d=0.20 would be rejected); fB has a loose
    T'=0.50 (same d admitted). The FIT whose threshold admits wins."""
    fits = [_fit("fA", [1.0, 0.0]), _fit("fB", [0.0, 1.0])]

    def pages(q, coll, bs, flt):
        # Both FITs see the same candidate at d=0.20.
        return [[_hit("cand", sim=0.80)]]

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.05, 0.50], collection="c", client=client,
        batch_size=10, max_k=10,
    )

    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.pk == "cand"
    assert c.nearest_fit_pk == "fB"  # only fB's T' admitted it


def test_attribution_is_min_raw_distance():
    """When multiple FITs admit the same chunk under their own T'_i,
    the winner is the FIT with the smallest *raw* d, regardless of
    which one's T' was tighter or looser."""
    fits = [_fit("fA", [1.0, 0.0]), _fit("fB", [0.0, 1.0])]

    def pages(q, coll, bs, flt):
        if q == [1.0, 0.0]:
            return [[_hit("shared", sim=0.70)]]  # d=0.30
        return [[_hit("shared", sim=0.95)]]      # d=0.05

    client = _FakeMilvus(pages)
    # fA has a very loose T'; fB has a modest T'. Both admit the chunk.
    # The winner is fB (smaller raw distance), even though fA's T'
    # admits with more slack.
    result = retrieve_anchored(
        fits, T_primes=[0.99, 0.10], collection="c", client=client,
        batch_size=10, max_k=10,
    )
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.pk == "shared"
    assert c.nearest_fit_pk == "fB"
    assert c.nearest_fit_distance == pytest.approx(0.05)


def test_anchor_candidate_records_qualifying_fit_set():
    """A chunk admitted by 2 of 3 FITs records both qualifying FIT pks
    (sorted) plus the count. The third FIT's iterator misses the chunk
    so it must NOT appear in the qualifying set, even if its T' would
    have admitted similar distances."""
    fits = [
        _fit("fA", [1.0, 0.0]),
        _fit("fB", [0.0, 1.0]),
        _fit("fC", [0.0, 0.0]),
    ]

    def pages(q, coll, bs, flt):
        if q == [1.0, 0.0]:
            return [[_hit("shared", sim=0.95)]]      # admitted by fA
        if q == [0.0, 1.0]:
            return [[_hit("shared", sim=0.90)]]      # admitted by fB
        return [[_hit("only-fC-private", sim=0.99)]]  # fC misses "shared"

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.20, 0.20, 0.20], collection="c", client=client,
        batch_size=10, max_k=10,
    )

    by_pk = {c.pk: c for c in result.candidates}
    shared = by_pk["shared"]
    assert shared.qualifying_fit_count == 2
    assert shared.qualifying_fit_pks == ("fA", "fB")  # sorted

    # Single-anchor chunk records exactly its admitting FIT.
    only_fc = by_pk["only-fC-private"]
    assert only_fc.qualifying_fit_count == 1
    assert only_fc.qualifying_fit_pks == ("fC",)

    # qualifying_count_by_pk on the result is consistent and includes
    # only post-T'-filter pks.
    assert result.qualifying_count_by_pk["shared"] == 2
    assert result.qualifying_count_by_pk["only-fC-private"] == 1


def test_retrieve_excludes_fit_own_pks():
    fits = [_fit("fA", [1.0, 0.0])]

    def pages(q, coll, bs, flt):
        return [[_hit("fA", sim=1.0), _hit("other", sim=0.9)]]

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.5], collection="c", client=client,
        batch_size=10, max_k=10,
    )
    assert {c.pk for c in result.candidates} == {"other"}


# --------------------------- stopping rule -----------------------------


def test_stopping_rule_halts_on_page_crossing_T_prime():
    """Page 1 all ≤ T'; page 2's last hit > T' → loop stops before
    page 3."""
    fits = [_fit("fA", [1.0, 0.0])]

    def pages(q, coll, bs, flt):
        return [
            [_hit("p1a", sim=0.99), _hit("p1b", sim=0.95)],  # d 0.01, 0.05
            [_hit("p2a", sim=0.90), _hit("p2b", sim=0.20)],  # d 0.10, 0.80 → cross
            [_hit("p3a", sim=0.99)],  # must never be seen
        ]

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.15], collection="c", client=client,
        batch_size=10, max_k=100,
    )

    pks = {c.pk for c in result.candidates}
    assert pks == {"p1a", "p1b", "p2a"}  # p2b crosses; p3* unseen
    it = client.iterators[0]
    # Page 1 + page 2, no page-3 fetch.
    assert it.next_calls == 2
    assert it.closed is True

    fit_pages = result.per_fit_pages[0]
    assert fit_pages.pages_fetched == 2
    assert fit_pages.total_hits == 4
    assert fit_pages.budget_exhausted is False
    assert fit_pages.final_kth_distance > 0.15


def test_stopping_rule_natural_exhaustion():
    """Iterator returns an empty page before T' crosses — natural
    corpus exhaustion. final_kth_distance → inf → not exhausted."""
    fits = [_fit("fA", [1.0, 0.0])]

    def pages(q, coll, bs, flt):
        return [[_hit("only", sim=0.99)]]  # then empty

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.20], collection="c", client=client,
        batch_size=10, max_k=100,
    )
    assert not result.per_fit_pages[0].budget_exhausted
    # Iterator called once for the page, once more returning [] to
    # trigger natural-exhaustion break.
    assert client.iterators[0].next_calls == 2


# ----------------------------- max_k cap -------------------------------


def test_max_k_halts_loop_and_flags_budget_exhausted():
    """Infinite pages all at d ≤ T': halt at max_k and flag exhausted."""
    fits = [_fit("fA", [1.0, 0.0])]

    # Unique pk across pages so we can verify total_hits clamps at
    # max_k (10 hits/page × 3 pages = 30 total, but max_k=25 must
    # break inside page 3 at the 25th hit).
    counter = {"n": 0}

    def pages(q, coll, bs, flt):
        out = []
        for _p in range(10):
            page = []
            for _i in range(bs):
                counter["n"] += 1
                page.append(_hit(f"p{counter['n']}", sim=0.99))
            out.append(page)
        return out

    client = _FakeMilvus(pages)
    result = retrieve_anchored(
        fits, T_primes=[0.50], collection="c", client=client,
        batch_size=10, max_k=25,
    )

    fit_pages = result.per_fit_pages[0]
    assert fit_pages.budget_exhausted is True
    assert fit_pages.total_hits == 25
    assert fit_pages.final_kth_distance <= 0.50
    # finalise_candidates still runs — no exception.
    assert len(result.candidates) == 25
    assert client.iterators[0].closed is True


def test_max_k_below_batch_size_rejected():
    fits = [_fit("fA", [1.0, 0.0])]
    client = _FakeMilvus(lambda *_a, **_k: [])

    with pytest.raises(ValueError, match="max_k"):
        retrieve_anchored(
            fits, T_primes=[0.5], collection="c", client=client,
            batch_size=1000, max_k=500,
        )


# ------------------------- error propagation ---------------------------


def test_milvus_error_wrapped_as_anchor_retrieval():
    """Iterator opens via pymilvus — if the server rejects the call,
    we surface it as AnchorRetrievalError."""
    from pymilvus.exceptions import MilvusException

    fits = [_fit("fA", [1.0, 0.0])]

    class _BoomClient:
        def search_iterator(self, **kwargs):
            raise MilvusException(0, "connection refused")

        def close(self):
            pass

    with pytest.raises(AnchorRetrievalError, match="search_iterator"):
        retrieve_anchored(
            fits, T_primes=[0.5], collection="c", client=_BoomClient(),
            batch_size=10, max_k=100,
        )
