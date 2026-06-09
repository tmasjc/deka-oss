"""Unit tests for src.anchor.validate — batched LOO gate."""

from __future__ import annotations

from typing import Any

import pytest

from src.anchor.errors import AnchorRetrievalError
from src.anchor.loader import FitChunk
from src.anchor.validate import (
    RecoveryResult,
    count_not_fit_intrusions,
    loo_fit_recovery,
)


def _fit(pk: str, span: list[float], chunk: list[float]) -> FitChunk:
    return FitChunk(
        pk=pk,
        chunk_id=f"cid-{pk}",
        chunk_content="",
        sample_id="",
        counselor_id="",
        term="",
        span_text=f"span-{pk}",
        span_line_indices=[0],
        span_embedding=span,
        chunk_embedding=chunk,
    )


def _hit(pk: str, sim: float) -> dict:
    return {"id": pk, "distance": sim}


class _ScriptedClient:
    """Returns neighbor lists scripted by (query_vec -> hits)."""

    def __init__(self, script: dict[tuple[float, ...], list[dict[str, Any]]]):
        self._script = script
        self.call_count = 0
        self.last_kwargs: dict[str, Any] | None = None

    def search(
        self,
        *,
        data,
        collection_name,
        anns_field,
        search_params,
        limit,
        output_fields,
        filter=None,
    ):
        self.call_count += 1
        self.last_kwargs = {
            "collection_name": collection_name,
            "anns_field": anns_field,
            "limit": limit,
            "output_fields": list(output_fields),
            "filter": filter,
            "n_queries": len(data),
        }
        return [self._script[tuple(v)][:limit] for v in data]

    def close(self):
        pass


def test_loo_recovery_healthy_when_all_recovered():
    # Three FITs clustered tightly — every LOO slice returns all 3 FIT
    # pks inside any recalibrated T'.
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.95, 0.31], [0.95, 0.31]),
        _fit("pk-C", [0.85, 0.53], [0.85, 0.53]),
    ]
    script = {
        tuple(f.span_embedding): [
            _hit("pk-A", 0.99),
            _hit("pk-B", 0.99),
            _hit("pk-C", 0.99),
        ]
        for f in fits
    }
    client = _ScriptedClient(script)

    result = loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=100,
    )

    assert isinstance(result, RecoveryResult)
    assert result.recovered == 3
    assert result.total == 3
    assert result.verdict == "HEALTHY"
    assert all(p.recovered for p in result.per_fit)
    # LOO row reports session-level T on rest (not T + δ).
    assert all(p.recalibrated_T > 0 for p in result.per_fit)
    assert all(not hasattr(p, "recalibrated_T_prime") for p in result.per_fit)


def test_loo_recovery_failed_when_none_recovered():
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]
    script = {tuple(f.span_embedding): [_hit("unrelated", 0.95)] for f in fits}
    client = _ScriptedClient(script)

    result = loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=100,
    )
    assert result.recovered == 0
    assert result.verdict == "FAILED"
    assert [p.fit_pk for p in result.missed_fits] == ["pk-A", "pk-B", "pk-C"]
    assert [p.fit_chunk_id for p in result.missed_fits] == [
        "cid-pk-A",
        "cid-pk-B",
        "cid-pk-C",
    ]
    assert not hasattr(result, "missed_fit_chunk_ids")


def test_loo_uses_per_rest_fit_T_prime_filter():
    """Held-out chunk is admitted by only one rest FIT (the one with
    a large δ_j). Earlier global-δ behaviour would admit or reject
    uniformly per slice; per-FIT δ admits the held-out iff at least
    one rest FIT's T'_rest_j covers the distance."""
    # Construct 3 FITs where FIT A's chunk sits close to FIT B's span
    # but far from FIT C's span. If held out A, we need B's admission
    # (B has tight δ_B = small) or C's admission (C has large δ_C).
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.95, 0.31], [0.95, 0.31]),
        _fit("pk-C", [0.85, 0.53], [0.85, 0.53]),
    ]
    script = {
        tuple(f.span_embedding): [
            _hit("pk-A", 0.99),
            _hit("pk-B", 0.99),
            _hit("pk-C", 0.99),
        ]
        for f in fits
    }
    client = _ScriptedClient(script)

    result = loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=100,
    )
    # All three rest FITs admit the held-out at d≈0.01 under any
    # realistic T'_rest_j — this confirms the per-rest-FIT filter
    # doesn't break the simple recovery path.
    assert result.recovered == 3


def test_count_not_fit_intrusions():
    assert (
        count_not_fit_intrusions(
            candidate_pks=["a", "b", "c"],
            not_fit_pks=frozenset({"b", "z"}),
        )
        == 1
    )


def test_loo_uses_single_batched_search():
    """Batched LOO = one client.search over N·(N-1) vectors. Anything
    more round-trippy is the old per-slice iterator regression."""
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]
    script = {tuple(f.span_embedding): [_hit(f.pk, 0.99)] for f in fits}
    client = _ScriptedClient(script)

    loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=100,
    )
    assert client.call_count == 1
    # N=3 hold-outs × N-1=2 rest spans = 6 flattened vectors.
    assert client.last_kwargs["n_queries"] == 3 * 2


def test_loo_output_fields_are_empty():
    """LOO must request output_fields=[] — at production N the full
    payload overflows pymilvus's 256 MB gRPC cap (PR #17 regression)."""
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]
    script = {tuple(f.span_embedding): [_hit(f.pk, 0.99)] for f in fits}
    client = _ScriptedClient(script)
    loo_fit_recovery(fits, collection="c", client=client, max_k=100)
    assert client.last_kwargs["output_fields"] == []


def test_loo_limit_clamped_to_milvus_page_cap():
    """max_k bigger than Milvus's MAX_BATCH_SIZE → LOO limit clamped
    to 16384 (the server rejects anything above)."""
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]
    script = {tuple(f.span_embedding): [_hit(f.pk, 0.99)] for f in fits}
    client = _ScriptedClient(script)
    loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=200_000,
    )
    assert client.last_kwargs["limit"] == 16384


def test_loo_limit_uses_max_k_when_below_cap():
    """Explicit --max-k below 16384 wins (smaller budget)."""
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]
    script = {tuple(f.span_embedding): [_hit(f.pk, 0.99)] for f in fits}
    client = _ScriptedClient(script)
    loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=500,
    )
    assert client.last_kwargs["limit"] == 500


def test_loo_emits_progress_per_slice():
    """Progress callback fires at batch open + one entry per slice."""
    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]
    script = {
        tuple(f.span_embedding): [
            _hit("pk-A", 0.99),
            _hit("pk-B", 0.99),
            _hit("pk-C", 0.99),
        ]
        for f in fits
    }
    client = _ScriptedClient(script)
    msgs: list[str] = []
    loo_fit_recovery(
        fits,
        collection="c",
        client=client,
        max_k=100,
        progress=msgs.append,
    )
    assert any("batched search" in m for m in msgs)
    # One per-slice message per hold-out + final summary.
    slice_msgs = [m for m in msgs if m.startswith("LOO ")]
    assert len(slice_msgs) >= len(fits)
    assert any("complete" in m.lower() for m in msgs)


def test_loo_below_min_fits_raises():
    with pytest.raises(ValueError, match="at least 2"):
        loo_fit_recovery(
            [_fit("solo", [1.0, 0.0], [1.0, 0.0])],
            collection="c",
            client=_ScriptedClient({}),
            max_k=100,
        )


def test_loo_chunks_vectors_to_avoid_grpc_size_cap(monkeypatch):
    """Production sessions with ~24 FITs build 552-vector LOO calls
    that overflow pymilvus's 256 MB gRPC cap. The implementation
    splits ``span_vectors`` into ``_LOO_VECTOR_CHUNK``-sized batches
    and concatenates the responses so indexing via ``offsets`` still
    lines up after the split."""
    import src.anchor.validate as v_mod

    # Force chunking by shrinking the cap. N=5 → 5×4=20 vectors,
    # chunk=4 → 5 calls. Each call must receive exactly its slice.
    monkeypatch.setattr(v_mod, "_LOO_VECTOR_CHUNK", 4)

    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.95, 0.31], [0.95, 0.31]),
        _fit("pk-C", [0.85, 0.53], [0.85, 0.53]),
        _fit("pk-D", [0.70, 0.71], [0.70, 0.71]),
        _fit("pk-E", [0.50, 0.86], [0.50, 0.86]),
    ]
    # Every span recovers every fit pk so every hold-out succeeds —
    # we're testing chunking arithmetic, not the recovery verdict.
    script = {tuple(f.span_embedding): [_hit(g.pk, 0.99) for g in fits] for f in fits}

    class _RecordingClient:
        def __init__(self, script):
            self._script = script
            self.calls: list[int] = []

        def search(self, *, data, **kwargs):
            self.calls.append(len(data))
            return [self._script[tuple(v)][: kwargs["limit"]] for v in data]

        def close(self):
            pass

    client = _RecordingClient(script)
    result = loo_fit_recovery(fits, collection="c", client=client, max_k=100)

    # 5 fits × 4 rest = 20 vectors, chunk_size=4 → 5 calls of size 4.
    assert client.calls == [4, 4, 4, 4, 4]
    # Recovery semantics unchanged across the split.
    assert result.recovered == 5
    assert result.total == 5


def test_loo_milvus_error_wrapped():
    from pymilvus.exceptions import MilvusException

    fits = [
        _fit("pk-A", [1.0, 0.0], [1.0, 0.0]),
        _fit("pk-B", [0.9, 0.1], [0.9, 0.1]),
        _fit("pk-C", [0.8, 0.2], [0.8, 0.2]),
    ]

    class _BoomClient:
        def search(self, **kwargs):
            raise MilvusException(0, "timeout")

        def close(self):
            pass

    with pytest.raises(AnchorRetrievalError, match="LOO batched search"):
        loo_fit_recovery(
            fits,
            collection="c",
            client=_BoomClient(),
            max_k=100,
        )
