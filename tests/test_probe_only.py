"""Tests for ``probe_only`` and ``run_search`` probe re-use parity."""

from __future__ import annotations

from typing import Any

import pytest

from src.search import (
    ProbeResult,
    SearchConfig,
    probe_only,
    run_search,
)
from src.search import search as search_module


class _RecordingRanker:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeClient:
    def __init__(
        self,
        per_path_results: dict[str, list[tuple[int, float, dict[str, Any]]]],
        fused: list[tuple[int, dict[str, Any]]],
    ) -> None:
        self.per_path_results = per_path_results
        self.fused = fused
        self.search_calls: list[dict[str, Any]] = []
        self.hybrid_calls: list[dict[str, Any]] = []

    def search(
        self,
        *,
        collection_name: str,
        data: Any,
        anns_field: str,
        search_params: dict[str, Any],
        limit: int,
        output_fields: list[str],
        filter: str = "",
    ) -> list[list[dict[str, Any]]]:
        self.search_calls.append(
            {
                "anns_field": anns_field,
                "limit": limit,
                "filter": filter,
            }
        )
        hits = [
            {"id": pk, "distance": score, "entity": entity}
            for pk, score, entity in self.per_path_results.get(anns_field, [])
        ]
        return [hits]

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[Any],
        ranker: Any,
        limit: int,
        output_fields: list[str],
    ) -> list[list[dict[str, Any]]]:
        self.hybrid_calls.append(
            {
                "ranker": ranker,
                "limit": limit,
                "num_reqs": len(reqs),
                "req_exprs": [getattr(r, "expr", None) for r in reqs],
            }
        )
        hits = [
            {"id": pk, "distance": 0.0, "entity": entity} for pk, entity in self.fused
        ]
        return [hits]

    def close(self) -> None:
        pass


def _make_config(**overrides: Any) -> SearchConfig:
    base: dict[str, Any] = {
        "rrf_k": 60,
        "per_path_limit": 5,
        "top_k": 3,
        "embed_url": "http://fake",
        "milvus_uri": "http://fake",
        "collection": "fake",
    }
    base.update(overrides)
    return SearchConfig(**base)


def _fake_embedding(sparse: dict[str, float] | None = None):
    if sparse is None:
        sparse = {"1": 0.5}

    def _impl(sentences: list[str], embed_url: str, *, timeout: int = 30) -> dict:
        return {
            "dense_embeddings": [[0.1, 0.2, 0.3] for _ in sentences],
            "sparse_embeddings": [sparse for _ in sentences],
        }

    return _impl


@pytest.fixture
def patch_milvus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)


class TestProbeOnly:
    def test_returns_per_path_stats(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        per_path = {
            "dense_embedding": [(1, 0.9, {}), (2, 0.8, {}), (3, 0.7, {})],
            "sparse_embedding": [(1, 0.5, {}), (2, 0.4, {})],
        }
        client = _FakeClient(per_path_results=per_path, fused=[])

        result = probe_only("query", _make_config(), client=client)

        assert isinstance(result, ProbeResult)
        assert result.query == "query"
        assert result.stats_by_path["dense"]["hit_count"] == 3
        assert result.stats_by_path["dense"]["score_max"] == 0.9
        assert result.stats_by_path["dense"]["score_min"] == 0.7
        assert result.stats_by_path["sparse"]["hit_count"] == 2
        assert result.stats_by_path["dense"]["top3_pks"] == [1, 2, 3]
        # Embed diagnostics carry through
        assert result.embed_diagnostics["dense_dim"] == 3
        assert result.embed_diagnostics["sparse_nnz"] == 1
        # Hybrid_search must NOT have been called by probe_only.
        assert client.hybrid_calls == []

    def test_injects_filter_into_all_probes(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        """When seen_pks is supplied, every per-path probe must carry the
        ``id not in [...]`` filter at the Milvus layer."""
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        client = _FakeClient(per_path_results=per_path, fused=[])

        seen = frozenset({"x", "y", "z"})
        result = probe_only("query", _make_config(), client=client, seen_pks=seen)

        assert len(client.search_calls) == 2
        for call in client.search_calls:
            expr = call["filter"]
            assert expr.startswith("id not in [")
            assert '"x"' in expr and '"y"' in expr and '"z"' in expr
        for path in ("dense", "sparse"):
            assert result.stats_by_path[path]["filtered_by_seen"] == 3

    def test_skips_sparse_when_embedding_empty(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding(sparse={}))
        client = _FakeClient(
            per_path_results={"dense_embedding": [(1, 0.9, {})]},
            fused=[],
        )

        result = probe_only("query", _make_config(), client=client)

        assert result.stats_by_path["sparse"]["skipped"] is True
        assert result.stats_by_path["sparse"]["hit_count"] == 0
        # Sparse field should not have been queried.
        probed_fields = {call["anns_field"] for call in client.search_calls}
        assert "sparse_embedding" not in probed_fields


class TestRunSearchProbeReuse:
    def test_run_search_with_probe_skips_embed_and_probes(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        per_path = {
            "dense_embedding": [(1, 0.9, {}), (2, 0.8, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [
            (
                1,
                {
                    "sample_id": "S1",
                    "counselor_id": "T1",
                    "term": "T",
                    "chunk_content": (
                        "老师：你好，今天我们想聊一下孩子最近的学习情况好吗？\n"
                        "家长：好的，老师，非常感谢您一直以来的关心和指导。"
                    ),
                    "chunk_id": 1,
                },
            )
        ]
        cfg = _make_config()

        # First, run probe with one client.
        probe_client = _FakeClient(per_path_results=per_path, fused=[])
        probe = probe_only("q", cfg, client=probe_client)

        # Now hand the probe to run_search with a fresh client.
        run_client = _FakeClient(per_path_results=per_path, fused=fused)
        # Reset embedding to fail loudly if called again.
        called: dict[str, int] = {"embed": 0}

        def _no_embed(*args: Any, **kwargs: Any) -> Any:
            called["embed"] += 1
            raise AssertionError(
                "get_embeddings should not be called when probe reused"
            )

        monkeypatch.setattr(search_module, "get_embeddings", _no_embed)

        table = run_search("q", cfg, client=run_client, probe=probe)

        assert called["embed"] == 0
        # No per-path search calls on the run_client — those are reused from the probe.
        assert run_client.search_calls == []
        # But fused hybrid_search was called.
        assert len(run_client.hybrid_calls) == 1
        # Probe stats land in the diagnostics.
        assert table.search_diagnostics is not None
        assert table.search_diagnostics["probes"] == probe.stats_by_path
        assert table.search_diagnostics["embedding"]["reused_from_probe"] is True

    def test_run_search_rejects_mismatched_probe_query(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config()
        probe_client = _FakeClient(
            per_path_results={
                "dense_embedding": [(1, 0.9, {})],
                "sparse_embedding": [],
            },
            fused=[],
        )
        probe = probe_only("query A", cfg, client=probe_client)

        with pytest.raises(ValueError, match="does not match"):
            run_search(
                "query B",
                cfg,
                client=_FakeClient(per_path_results={}, fused=[]),
                probe=probe,
            )

    def test_run_search_without_probe_runs_legacy_path(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        # Smoke test: the legacy code path (no probe arg) still runs.
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config()
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [
            (
                1,
                {
                    "sample_id": "S1",
                    "counselor_id": "T1",
                    "term": "T",
                    "chunk_content": (
                        "老师：你好，今天我们想聊一下孩子最近的学习情况好吗？\n"
                        "家长：好的，老师，非常感谢您一直以来的关心和指导。"
                    ),
                    "chunk_id": 1,
                },
            )
        ]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        table = run_search("q", cfg, client=client)

        assert len(table.rows) == 1
        # Both per-path probes ran via run_search.
        assert len(client.search_calls) == 2
        assert table.search_diagnostics is not None
        assert "probes" in table.search_diagnostics
        assert "reused_from_probe" not in table.search_diagnostics["embedding"]


class TestAutoRetry:
    """The min_survivors auto-retry doubles per_path_limit until met or 3 attempts."""

    def _multiline(self, idx: int) -> dict[str, Any]:
        # Content must clear the 50-char short-chunk gate after .strip().
        return {
            "sample_id": f"S{idx}",
            "counselor_id": "T",
            "term": "T",
            "chunk_content": (
                f"老师：你好，今天我们想聊一下孩子最近的学习情况好吗？(line {idx})\n"
                "家长：好的，老师，非常感谢您一直以来的关心和指导。"
            ),
            "chunk_id": idx,
        }

    def test_no_retry_when_first_attempt_meets_floor(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config(min_survivors=1)
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [(1, self._multiline(1))]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        table = run_search("q", cfg, client=client)

        assert table.search_diagnostics["auto_retry"]["triggered"] is False
        assert len(table.search_diagnostics["auto_retry"]["attempts"]) == 1
        assert (
            table.search_diagnostics["auto_retry"]["final_per_path_limit"]
            == cfg.per_path_limit
        )
        # Single fused call.
        assert len(client.hybrid_calls) == 1

    def test_retry_doubles_per_path_limit_until_met(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        # Fake client returns a single (too-short) fused row that always
        # gets filtered, until per_path_limit reaches 20 — at that point it
        # returns long rows that survive the short-chunk gate.
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config(per_path_limit=5, top_k=3, min_survivors=1)

        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }

        class _LimitGatedClient(_FakeClient):
            def hybrid_search(self, **kwargs: Any) -> Any:
                self.hybrid_calls.append(kwargs)
                # Use the inner per-path search limit as a proxy for
                # current_per_path_limit on this attempt.
                limit_seen = self.search_calls[-1]["limit"] if self.search_calls else 0
                if limit_seen >= 20:
                    return [
                        [
                            {"id": pk, "distance": 0.0, "entity": entity}
                            for pk, entity in self.fused
                        ]
                    ]
                # Too-short content gets filtered by the gate.
                return [
                    [
                        {
                            "id": 99,
                            "distance": 0.0,
                            "entity": {
                                "sample_id": "S99",
                                "counselor_id": "T",
                                "term": "T",
                                "chunk_content": "x",
                                "chunk_id": 99,
                            },
                        }
                    ]
                ]

        client = _LimitGatedClient(
            per_path_results=per_path,
            fused=[(1, self._multiline(1)), (2, self._multiline(2))],
        )

        table = run_search("q", cfg, client=client)

        ar = table.search_diagnostics["auto_retry"]
        assert ar["triggered"] is True
        # Attempt 1 (5), attempt 2 (10), attempt 3 (20) — survivors meet on 3.
        assert [a["per_path_limit"] for a in ar["attempts"]] == [5, 10, 20]
        assert ar["final_per_path_limit"] == 20
        assert ar["met"] is True
        assert table.config.per_path_limit == 20

    def test_retry_caps_at_three_attempts_even_when_unmet(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        # Always returns too-short content; retry exhausts after 3 attempts.
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config(per_path_limit=5, top_k=3, min_survivors=2)

        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [
            (
                1,
                {
                    "sample_id": "S1",
                    "counselor_id": "T",
                    "term": "T",
                    "chunk_content": "x",  # always too short, always filtered
                    "chunk_id": 1,
                },
            )
        ]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        table = run_search("q", cfg, client=client)

        ar = table.search_diagnostics["auto_retry"]
        assert ar["triggered"] is True
        assert len(ar["attempts"]) == 3
        assert ar["met"] is False
        assert ar["final_per_path_limit"] == 20  # 5 -> 10 -> 20
        assert len(table.rows) == 0

    def test_retry_emits_searching_retry_stage(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config(per_path_limit=5, top_k=3, min_survivors=2)
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [
            (
                1,
                {
                    "sample_id": "S1",
                    "counselor_id": "T",
                    "term": "T",
                    "chunk_content": "x",
                    "chunk_id": 1,
                },
            )
        ]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        stages: list[tuple[str, dict[str, Any]]] = []
        run_search(
            "q",
            cfg,
            client=client,
            on_stage=lambda s, info: stages.append((s, info)),
        )

        retry_stages = [s for s in stages if s[0] == "searching_retry"]
        # Two retries (attempts 2 and 3) emit the stage.
        assert len(retry_stages) == 2
        assert retry_stages[0][1]["attempt"] == 2
        assert retry_stages[0][1]["per_path_limit"] == 10
        assert retry_stages[1][1]["attempt"] == 3
        assert retry_stages[1][1]["per_path_limit"] == 20

    def test_fused_limit_floor_is_two_top_k(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        """When per_path_limit < 2*top_k the floor (2*top_k) wins."""
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config(per_path_limit=5, top_k=3, min_survivors=1)
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [(1, self._multiline(1))]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        run_search("q", cfg, client=client)

        # max(top_k * 2 = 6, per_path_limit = 5) = 6
        assert client.hybrid_calls[0]["limit"] == 6

    def test_fused_limit_scales_with_per_path_limit(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        """When per_path_limit > 2*top_k it drives the fused limit.

        Regression for runs/e557f005a0b3: a hard 2*top_k cap on the fused
        step pinned survivors flat across retries even as per_path_limit
        doubled. Each retry must now ask the fused step for more candidates.
        """
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        # top_k=3, so 2*top_k=6. per_path_limit=20 → fused_limit=20.
        cfg = _make_config(per_path_limit=20, top_k=3, min_survivors=1)
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        fused = [(1, self._multiline(1))]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        run_search("q", cfg, client=client)

        assert client.hybrid_calls[0]["limit"] == 20

    def test_fused_limit_grows_each_retry(
        self, monkeypatch: pytest.MonkeyPatch, patch_milvus: None
    ) -> None:
        """The fused limit must grow with each retry's per_path_limit bump."""
        monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
        cfg = _make_config(per_path_limit=20, top_k=3, min_survivors=2)
        per_path = {
            "dense_embedding": [(1, 0.9, {})],
            "sparse_embedding": [(1, 0.5, {})],
        }
        # All too short — every fused row is filtered, every attempt fails.
        fused = [
            (
                1,
                {
                    "sample_id": "S1",
                    "counselor_id": "T",
                    "term": "T",
                    "chunk_content": "x",
                    "chunk_id": 1,
                },
            )
        ]
        client = _FakeClient(per_path_results=per_path, fused=fused)

        run_search("q", cfg, client=client)

        # Three attempts: per_path_limit 20 → 40 → 80. fused_limit tracks
        # because top_k*2 = 6 < per_path_limit on every attempt.
        limits = [call["limit"] for call in client.hybrid_calls]
        assert limits == [20, 40, 80]
