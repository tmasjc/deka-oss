"""Unit + integration tests for ``src.search``.

Unit tests use a ``_FakeClient`` stub and never touch real services.
The single integration test is marked ``@pytest.mark.integration`` and
is deselected by default (see ``pyproject.toml``). Run with::

    uv run pytest tests/test_search.py -v
    uv run pytest tests/test_search.py -v -m integration
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest
import requests

from src.search import (
    CandidateRow,
    ConfigError,
    EmbeddingServiceError,
    EvidenceRow,
    EvidenceTable,
    SearchConfig,
    compute_breakdown,
    load_default_config,
    run_search,
)
from src.search import search as search_module
from src.search.embedding import sparse_to_milvus
from src.search.search import _build_chunk_id, _build_ranker, _build_requests


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingRanker:
    """Stand-in for RRFRanker that captures its args."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeClient:
    """Minimal Milvus-client stand-in covering search + hybrid_search.

    ``per_path_results`` maps ``anns_field`` → list of ``(pk, score,
    entity)`` tuples. ``fused`` is the list of tuples the fake
    ``hybrid_search`` returns.
    """

    def __init__(
        self,
        per_path_results: dict[str, list[tuple[int, float, dict[str, Any]]]],
        fused: list[tuple[int, dict[str, Any]]],
    ) -> None:
        self.per_path_results = per_path_results
        self.fused = fused
        self.hybrid_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.closed = False

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
                "collection_name": collection_name,
                "anns_field": anns_field,
                "search_params": search_params,
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
                "collection_name": collection_name,
                "ranker": ranker,
                "limit": limit,
                "output_fields": output_fields,
                "num_reqs": len(reqs),
                "req_exprs": [getattr(r, "expr", None) for r in reqs],
            }
        )
        hits = [
            {"id": pk, "distance": 0.0, "entity": entity} for pk, entity in self.fused
        ]
        return [hits]

    def close(self) -> None:
        self.closed = True


def _make_config(**overrides: Any) -> SearchConfig:
    base = {
        "rrf_k": 60,
        "per_path_limit": 5,
        "top_k": 3,
        "embed_url": "http://fake-embed",
        "milvus_uri": "http://fake-milvus",
        # ``collection`` is no longer a config-level concern — it's
        # injected per-session by ``with_overrides``. Tests that call
        # the search functions directly pass it explicitly here so the
        # fake Milvus client gets a stable name to match against.
        "collection": "fake_collection",
    }
    base.update(overrides)
    return SearchConfig(**base)


def _fake_embedding(dense_dim: int = 4, sparse: dict[str, float] | None = None):
    """Return a drop-in replacement for ``get_embeddings``."""

    if sparse is None:
        sparse = {"1": 0.5, "2": 0.25}

    def _impl(sentences: list[str], embed_url: str, *, timeout: int = 30) -> dict:
        return {
            "dense_embeddings": [[0.1] * dense_dim for _ in sentences],
            "sparse_embeddings": [sparse for _ in sentences],
        }

    return _impl


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


_SEARCH_KEYS = textwrap.dedent(
    """\
    rrf_k: 60
    per_path_limit: 10
    top_k: 5
    min_survivors: 3
    active_paths: [dense, sparse]
    embed_url: http://localhost:8000
    milvus_uri: http://localhost:19530
    http_timeout: 30
    """
)


def _write_yaml(
    path: Path,
    *,
    replace: dict[str, str] | None = None,
    extra: str = "",
    drop: set[str] | None = None,
) -> Path:
    """Write a unified YAML with a populated ``search:`` section.

    ``replace`` / ``drop`` / ``extra`` operate on the ``search`` section
    only so tests can inject malformed inputs without repeating the
    surrounding structure.
    """
    lines = _SEARCH_KEYS.splitlines()
    if drop:
        lines = [ln for ln in lines if not any(ln.startswith(f"{k}:") for k in drop)]
    if replace:
        for k, v in replace.items():
            lines = [f"{k}: {v}" if ln.startswith(f"{k}:") else ln for ln in lines]
    body = "search:\n" + "\n".join(f"  {ln}" if ln else "" for ln in lines)
    if extra:
        body += "\n" + "\n".join(f"  {ln}" for ln in extra.splitlines() if ln)
    path.write_text(body + "\n", encoding="utf-8")
    return path


_EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml.example"


def test_load_default_config_parses_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strip any environment overrides so we see the documented defaults.
    monkeypatch.delenv("DEKA_EMBED_URL", raising=False)
    monkeypatch.delenv("DEKA_MILVUS_URI", raising=False)

    cfg = load_default_config(_EXAMPLE_CONFIG)

    assert cfg.rrf_k == 60
    assert cfg.per_path_limit == 20
    assert cfg.top_k == 10
    assert cfg.min_survivors == 5
    assert cfg.embed_url == "http://localhost:8000"
    assert cfg.milvus_uri == "http://localhost:19530"
    # ``collection`` is intentionally blank in the loaded config —
    # session creation supplies it from the chosen scope.
    assert cfg.collection == ""


def test_load_default_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEKA_EMBED_URL", "http://embed.example")
    monkeypatch.setenv("DEKA_MILVUS_URI", "http://milvus.example")

    cfg = load_default_config(_EXAMPLE_CONFIG)

    assert cfg.embed_url == "http://embed.example"
    assert cfg.milvus_uri == "http://milvus.example"


def test_load_default_config_rejects_unknown_key(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "bad.yaml", extra="mystery_key: 42\n")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_default_config(bad)


def test_load_default_config_rejects_missing_key(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "bad.yaml", drop={"embed_url"})
    with pytest.raises(ConfigError, match="missing required keys"):
        load_default_config(bad)


def test_config_rejects_non_positive_rrf_k(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "bad_rrf.yaml", replace={"rrf_k": "0"})
    with pytest.raises(ConfigError, match="rrf_k"):
        load_default_config(bad)


def test_config_rejects_non_positive_min_survivors(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "bad_min.yaml", replace={"min_survivors": "0"})
    with pytest.raises(ConfigError, match="min_survivors"):
        load_default_config(bad)


def test_config_rejects_min_survivors_above_top_k(tmp_path: Path) -> None:
    bad = _write_yaml(
        tmp_path / "min_too_high.yaml",
        replace={"min_survivors": "10", "top_k": "5"},
    )
    with pytest.raises(ConfigError, match="cannot exceed"):
        load_default_config(bad)


def test_config_default_active_paths_all_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEKA_EMBED_URL", raising=False)
    cfg = load_default_config(_EXAMPLE_CONFIG)
    assert cfg.active_paths == frozenset({"dense", "sparse"})


def test_config_parses_active_paths_subset(tmp_path: Path) -> None:
    good = _write_yaml(tmp_path / "subset.yaml", replace={"active_paths": "[dense]"})
    cfg = load_default_config(good)
    assert cfg.active_paths == frozenset({"dense"})


def test_config_rejects_empty_active_paths(tmp_path: Path) -> None:
    bad = _write_yaml(tmp_path / "empty_paths.yaml", replace={"active_paths": "[]"})
    with pytest.raises(ConfigError, match="at least one path"):
        load_default_config(bad)


def test_config_rejects_unknown_path(tmp_path: Path) -> None:
    bad = _write_yaml(
        tmp_path / "unknown_path.yaml", replace={"active_paths": "[dense, bogus]"}
    )
    with pytest.raises(ConfigError, match="unknown path"):
        load_default_config(bad)


def test_config_rejects_legacy_bm25(tmp_path: Path) -> None:
    """BM25 was retired from Phase 1 after commit 7237be0; the YAML loader
    must reject it with the same 'unknown path' error as any other invalid
    name so users see a clear rejection instead of a silent acceptance."""
    bad = _write_yaml(
        tmp_path / "legacy_bm25.yaml",
        replace={"active_paths": "[dense, sparse, bm25]"},
    )
    with pytest.raises(ConfigError, match=r"unknown path.*bm25"):
        load_default_config(bad)


def test_config_rejects_duplicate_paths(tmp_path: Path) -> None:
    bad = _write_yaml(
        tmp_path / "dup_path.yaml",
        replace={"active_paths": "[dense, dense]"},
    )
    with pytest.raises(ConfigError, match="duplicates"):
        load_default_config(bad)


# ---------------------------------------------------------------------------
# embedding.py
# ---------------------------------------------------------------------------


def test_sparse_to_milvus_string_keys() -> None:
    out = sparse_to_milvus({"1": 0.5, "22": 0.1})

    assert out == {1: 0.5, 22: 0.1}
    assert all(isinstance(k, int) for k in out)
    assert all(isinstance(v, float) for v in out.values())


def test_sparse_to_milvus_empty() -> None:
    assert sparse_to_milvus({}) == {}


# ---------------------------------------------------------------------------
# ranker dispatch
# ---------------------------------------------------------------------------


def test_build_ranker_rrf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    cfg = _make_config(rrf_k=60)
    ranker = _build_ranker(cfg)

    assert isinstance(ranker, _RecordingRanker)
    assert ranker.kwargs == {"k": 60}


def test_build_requests_all_paths_when_all_active() -> None:
    reqs = _build_requests(
        dense_vec=[0.1, 0.2],
        sparse_vec={1: 0.5},
        limit=5,
        active_paths=frozenset({"dense", "sparse"}),
    )
    assert len(reqs) == 2
    fields = [r.anns_field for r in reqs]
    assert fields == ["dense_embedding", "sparse_embedding"]


def test_build_requests_filters_inactive_paths() -> None:
    reqs = _build_requests(
        dense_vec=[0.1, 0.2],
        sparse_vec={1: 0.5},
        limit=5,
        active_paths=frozenset({"dense"}),
    )
    assert len(reqs) == 1
    fields = [r.anns_field for r in reqs]
    assert fields == ["dense_embedding"]


def test_build_requests_single_active_path() -> None:
    reqs = _build_requests(
        dense_vec=[0.1, 0.2],
        sparse_vec={1: 0.5},
        limit=5,
        active_paths=frozenset({"dense"}),
    )
    assert len(reqs) == 1
    assert reqs[0].anns_field == "dense_embedding"


# ---------------------------------------------------------------------------
# chunk_id format
# ---------------------------------------------------------------------------


def test_chunk_id_uses_chunk_seq_when_available() -> None:
    # Live collection exposes an integer ``chunk_id`` field (the
    # per-transcript sequence); prefer it over the PK.
    assert _build_chunk_id("S1042", 16, "any-pk") == "S1042_C0000016"
    assert _build_chunk_id("S1042", 0, "any-pk") == "S1042_C0000000"
    assert _build_chunk_id("S1042", 99_999_999, "any-pk") == "S1042_C99999999"


def test_chunk_id_falls_back_to_int_pk() -> None:
    assert _build_chunk_id("S1042", None, 12345) == "S1042_C0012345"


def test_chunk_id_falls_back_to_string_pk() -> None:
    assert (
        _build_chunk_id("30006850", None, "37901377-30006850-16")
        == "30006850_C37901377-30006850-16"
    )


# ---------------------------------------------------------------------------
# run_search — provenance join
# ---------------------------------------------------------------------------


def test_provenance_join(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(per_path_limit=5, top_k=3)

    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    # Chunk 1001 comes back from both paths.
    # Chunk 1002 comes back from dense only.
    # Chunk 1003 comes back from sparse only.
    per_path = {
        "dense_embedding": [
            (1001, 0.91, {}),
            (1002, 0.88, {}),
            (
                1050,
                0.60,
                {
                    "sample_id": "S44",
                    "chunk_content": "老师：candidate\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                    "chunk_id": 50,
                },
            ),  # fused ranker will drop this one
        ],
        "sparse_embedding": [
            (1001, 0.75, {}),
            (1003, 0.55, {}),
        ],
    }

    fused_entities = [
        (
            1001,
            {
                "sample_id": "S42",
                "counselor_id": "T7",
                "term": "2025F",
                "chunk_content": "老师：chunk 1001\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                "chunk_id": 1,
            },
        ),
        (
            1002,
            {
                "sample_id": "S43",
                "counselor_id": "T7",
                "term": "2025F",
                "chunk_content": "老师：chunk 1002\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                "chunk_id": 2,
            },
        ),
        (
            1003,
            {
                "sample_id": "S99",
                "counselor_id": "T3",
                "term": "2025F",
                "chunk_content": "老师：chunk 1003\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                "chunk_id": 7,
            },
        ),
    ]

    client = _FakeClient(per_path_results=per_path, fused=fused_entities)

    table = run_search("fake query", cfg, client=client)

    assert [row.rank for row in table.rows] == [1, 2, 3]
    assert [row.pk for row in table.rows] == [1001, 1002, 1003]
    # Per-path candidates: dense's top-3 are [1001, 1002, 1050];
    # 1001/1002 are in the fused top-K so only 1050 should survive.
    assert [c.pk for c in table.per_path_candidates["dense"]] == [1050]
    assert table.per_path_candidates["dense"][0].rank_in_path == 3
    # sparse top-3 = [1001, 1003] — both in fusion → no candidates.
    assert table.per_path_candidates["sparse"] == []

    row1, row2, row3 = table.rows

    assert row1.chunk_id == "S42_C0000001"
    assert row1.source_paths == ["dense", "sparse"]
    assert row1.scores == {"dense": 0.91, "sparse": 0.75}

    assert row2.source_paths == ["dense"]
    assert row2.scores == {"dense": 0.88, "sparse": 0.0}

    assert row3.source_paths == ["sparse"]
    assert row3.scores == {"dense": 0.0, "sparse": 0.55}

    # run_search must NOT close a caller-injected client.
    assert client.closed is False
    # It called hybrid_search exactly once with two reqs (dense + sparse).
    assert len(client.hybrid_calls) == 1
    assert client.hybrid_calls[0]["num_reqs"] == 2
    assert client.hybrid_calls[0]["limit"] == cfg.top_k * 2
    # And two per-path probes at per_path_limit.
    assert len(client.search_calls) == 2
    assert all(call["limit"] == cfg.per_path_limit for call in client.search_calls)


# ---------------------------------------------------------------------------
# run_search — failure modes
# ---------------------------------------------------------------------------


def test_run_search_on_stage_reports_search_and_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_stage fires twice: searching_milvus at start, extracting_spans before the loop."""
    cfg = _make_config(per_path_limit=5, top_k=2)

    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(5001, 0.9, {})],
        "sparse_embedding": [(5001, 0.4, {})],
    }
    fused = [
        (
            5001,
            {
                "sample_id": "Sx",
                "counselor_id": "Tx",
                "term": "T",
                "chunk_content": "老师：test\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 42,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    calls: list[tuple[str, dict[str, Any]]] = []

    def _extract(chunk_content: str) -> tuple[list[int], str]:
        return [0], chunk_content.splitlines()[0]

    run_search(
        "q",
        cfg,
        client=client,
        extract_span=_extract,
        on_stage=lambda stage, info: calls.append((stage, info)),
    )

    stages = [s for s, _ in calls]
    assert stages[0] == "searching_milvus"
    assert "extracting_spans" in stages
    extracting_call = next(c for c in calls if c[0] == "extracting_spans")
    assert extracting_call[1]["total"] >= 1


def test_run_search_on_stage_skipped_when_no_extract_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without extract_span, only searching_milvus fires — no extracting_spans stage."""
    cfg = _make_config(per_path_limit=5, top_k=2)

    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(5001, 0.9, {})],
        "sparse_embedding": [(5001, 0.4, {})],
    }
    fused = [
        (
            5001,
            {
                "sample_id": "Sx",
                "counselor_id": "Tx",
                "term": "T",
                "chunk_content": "老师：test\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 42,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    stages: list[str] = []
    run_search("q", cfg, client=client, on_stage=lambda s, _i: stages.append(s))
    assert stages == ["searching_milvus"]


def test_run_search_handles_empty_sparse(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _make_config(per_path_limit=5, top_k=2)

    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding(sparse={}))
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(5001, 0.9, {})],
        "sparse_embedding": [],
    }
    fused = [
        (
            5001,
            {
                "sample_id": "Sx",
                "counselor_id": "Tx",
                "term": "T",
                "chunk_content": "老师：test\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 42,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    with caplog.at_level(logging.WARNING, logger="src.search.search"):
        table = run_search("q", cfg, client=client)

    assert len(table.rows) == 1
    assert table.rows[0].scores["sparse"] == 0.0
    assert table.rows[0].source_paths == ["dense"]
    # Sparse probe should have been skipped (empty sparse vec short-circuits it).
    probed_fields = {call["anns_field"] for call in client.search_calls}
    assert "sparse_embedding" not in probed_fields
    assert any("empty sparse" in rec.message for rec in caplog.records)


def test_run_search_injects_expr_into_all_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When seen_pks is supplied, every per-path search AND every fused
    AnnSearchRequest must carry the matching ``id not in [...]`` filter."""
    cfg = _make_config(per_path_limit=5, top_k=2)

    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(5001, 0.9, {})],
        "sparse_embedding": [(5001, 0.4, {})],
    }
    fused = [
        (
            5001,
            {
                "sample_id": "Sx",
                "counselor_id": "Tx",
                "term": "T",
                "chunk_content": "老师：test\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 42,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    seen = frozenset({"pk-a", "pk-b"})
    table = run_search("q", cfg, client=client, seen_pks=seen)

    # Every per-path search.filter must carry the expr.
    assert len(client.search_calls) == 2
    for call in client.search_calls:
        expr = call["filter"]
        assert expr.startswith("id not in [")
        assert '"pk-a"' in expr and '"pk-b"' in expr
    # Every AnnSearchRequest in the fused call must carry the same expr.
    assert len(client.hybrid_calls) == 1
    exprs = client.hybrid_calls[0]["req_exprs"]
    assert len(exprs) == 2
    assert all(e and e.startswith("id not in [") for e in exprs)
    # Diagnostics record the seen-set size.
    assert table.search_diagnostics is not None
    assert table.search_diagnostics["seen_set_size"] == 2
    # And the per-path probe stats track the filter count for downstream
    # rendering (so reflection can distinguish exhausted from dead paths).
    for path in ("dense", "sparse"):
        assert table.search_diagnostics["probes"][path]["filtered_by_seen"] == 2


def test_run_search_no_seen_set_passes_no_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: default seen_pks=None must pass no filter (empty
    string for client.search, None for AnnSearchRequest.expr)."""
    cfg = _make_config(per_path_limit=5, top_k=2)

    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(9001, 0.9, {})],
        "sparse_embedding": [(9001, 0.4, {})],
    }
    fused = [
        (
            9001,
            {
                "sample_id": "S",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": "老师：test\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    for call in client.search_calls:
        assert call["filter"] == ""
    assert all(e is None for e in client.hybrid_calls[0]["req_exprs"])
    assert table.search_diagnostics is not None
    assert table.search_diagnostics["seen_set_size"] == 0
    for path in ("dense", "sparse"):
        assert table.search_diagnostics["probes"][path]["filtered_by_seen"] == 0


def test_run_search_embedding_service_down(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()

    def _boom(sentences: list[str], embed_url: str, *, timeout: int = 30) -> dict:
        raise EmbeddingServiceError("connection refused")

    monkeypatch.setattr(search_module, "get_embeddings", _boom)

    with pytest.raises(EmbeddingServiceError, match="connection refused"):
        run_search("q", cfg, client=_FakeClient(per_path_results={}, fused=[]))


def test_get_embeddings_wraps_request_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapping layer at embedding.get_embeddings translates RequestException."""

    from src.search import embedding as embedding_module

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(embedding_module.requests, "post", _raise)

    with pytest.raises(EmbeddingServiceError, match="no route"):
        embedding_module.get_embeddings(["q"], "http://unreachable", timeout=5)


# ---------------------------------------------------------------------------
# compute_breakdown
# ---------------------------------------------------------------------------


def _row(
    rank: int,
    paths: list[str],
    rating: str | None,
) -> EvidenceRow:
    return EvidenceRow(
        rank=rank,
        pk=rank * 100,
        chunk_id=f"S1_C{rank * 100:07d}",
        chunk_content="x",
        sample_id="S1",
        counselor_id="T1",
        term="T",
        source_paths=list(paths),  # type: ignore[arg-type]
        scores={"dense": 0.0, "sparse": 0.0},
        rating=rating,  # type: ignore[arg-type]
    )


def test_compute_breakdown_tally() -> None:
    cfg = _make_config()
    table = EvidenceTable(
        query="q",
        config=cfg,
        rows=[
            _row(1, ["dense"], "FIT"),
            _row(2, ["dense"], "NOT_FIT"),
            _row(3, ["sparse"], "FIT"),
            _row(5, ["dense", "sparse"], "FIT"),
            _row(6, ["dense", "sparse"], "FIT"),
        ],
    )

    breakdown = compute_breakdown(table)

    assert breakdown["dense_only"] == {
        "total": 2,
        "fit": 1,
        "not_fit": 1,
        "discard": 0,
    }
    assert breakdown["sparse_only"] == {
        "total": 1,
        "fit": 1,
        "not_fit": 0,
        "discard": 0,
    }
    assert breakdown["multi_path"] == {
        "total": 2,
        "fit": 2,
        "not_fit": 0,
        "discard": 0,
    }


def test_compute_breakdown_counts_discard() -> None:
    """DISCARD-rated chunks tally into a third bucket and don't leak
    into the fit/not_fit counts (issue #46)."""
    cfg = _make_config()
    table = EvidenceTable(
        query="q",
        config=cfg,
        rows=[
            _row(1, ["dense"], "FIT"),
            _row(2, ["dense"], "DISCARD"),
            _row(3, ["sparse"], "NOT_FIT"),
            _row(4, ["sparse"], "DISCARD"),
            _row(5, ["dense", "sparse"], "DISCARD"),
        ],
    )

    breakdown = compute_breakdown(table)

    assert breakdown["dense_only"] == {
        "total": 2,
        "fit": 1,
        "not_fit": 0,
        "discard": 1,
    }
    assert breakdown["sparse_only"] == {
        "total": 2,
        "fit": 0,
        "not_fit": 1,
        "discard": 1,
    }
    assert breakdown["multi_path"] == {
        "total": 1,
        "fit": 0,
        "not_fit": 0,
        "discard": 1,
    }


def test_compute_breakdown_rejects_unrated() -> None:
    cfg = _make_config()
    table = EvidenceTable(
        query="q",
        config=cfg,
        rows=[
            _row(1, ["dense"], "FIT"),
            _row(2, ["sparse"], None),  # unrated
        ],
    )

    with pytest.raises(ValueError, match="unrated"):
        compute_breakdown(table)


def test_candidate_rating_helpers_and_all_rated() -> None:
    cfg = _make_config()
    table = EvidenceTable(
        query="q",
        config=cfg,
        rows=[_row(1, ["dense"], "FIT")],
        per_path_candidates={
            "dense": [],
            "sparse": [
                CandidateRow(
                    path="sparse",
                    rank_in_path=1,
                    pk=2001,
                    chunk_id="S1_C0002001",
                    chunk_content="content",
                    sample_id="S1",
                    counselor_id="T1",
                    term="T",
                    score=0.42,
                )
            ],
        },
    )

    # Default all_rated() ignores per-path candidates — outside audit
    # mode they are materialised but not surfaced for rating, so an
    # unrated candidate must not block turn completion.
    assert table.all_rated() is True

    # Audit-mode opt-in: include_candidates=True requires every
    # candidate to be rated as well.
    assert table.all_rated(include_candidates=True) is False
    table.set_candidate_rating("sparse", 1, "FIT")
    assert table.per_path_candidates["sparse"][0].rating == "FIT"
    assert table.all_rated(include_candidates=True) is True

    # Unknown candidate rank raises.
    with pytest.raises(KeyError):
        table.set_candidate_rating("sparse", 99, "FIT")

    # all_candidates flattens dense → sparse.
    flat = table.all_candidates()
    assert [c.path for c in flat] == ["sparse"]


def test_run_search_excludes_fused_pks_from_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path's top-3 PKs that survive into the fused top-K must not appear as
    candidates — those are already rated as part of the main evidence table."""
    cfg = _make_config(per_path_limit=5, top_k=2)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    # dense returns 3 candidates; only top-2 (2001, 2002) survive fusion.
    # 2003 is a path-only candidate that should appear in per_path_candidates.
    per_path = {
        "dense_embedding": [
            (
                2001,
                0.95,
                {
                    "sample_id": "S1",
                    "chunk_content": "老师：fused-1\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                    "chunk_id": 1,
                },
            ),
            (
                2002,
                0.85,
                {
                    "sample_id": "S2",
                    "chunk_content": "老师：fused-2\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                    "chunk_id": 2,
                },
            ),
            (
                2003,
                0.65,
                {
                    "sample_id": "S3",
                    "chunk_content": "老师：candidate\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                    "chunk_id": 3,
                },
            ),
        ],
        "sparse_embedding": [],
    }
    fused = [
        (
            2001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：fused-1\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                "chunk_id": 1,
            },
        ),
        (
            2002,
            {
                "sample_id": "S2",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：fused-2\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？",
                "chunk_id": 2,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    assert [r.pk for r in table.rows] == [2001, 2002]
    cands = table.per_path_candidates["dense"]
    assert len(cands) == 1
    assert cands[0].pk == 2003
    assert cands[0].rank_in_path == 3
    assert cands[0].path == "dense"
    assert cands[0].score == 0.65
    assert (
        cands[0].chunk_content
        == "老师：candidate\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。"
    )


def test_run_search_dense_short_hits_in_raw_fused_still_surface_as_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When dense's high-scoring top-3 are short enough to be dropped by Gate 1,
    they must still surface as per-path candidates. Reproduces the live-run
    bug where ``fused_pks`` is computed from raw fused (pre-gate) so the
    candidate builder treats dense's top-3 as 'already in fused' even though
    Gate 1 will drop them moments later — leaving the user with no view of
    dense's contribution at all."""
    cfg = _make_config(per_path_limit=5, top_k=2)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    short_a = "家长：这个价格对我们来说有点贵了承受不起"  # 20 chars
    short_b = "家长：但是别家机构现在好像便宜不少呀这让我怎么选"  # 24 chars
    short_c = "家长：报名之后真的会有效果吗我心里还有点没底"  # 22 chars
    long_sparse = (
        "老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？"
        "方便后续制定计划，今天的电话先聊到这里。"
    )

    per_path = {
        "dense_embedding": [
            (4001, 0.95, {"sample_id": "Sa", "chunk_content": short_a, "chunk_id": 1}),
            (4002, 0.90, {"sample_id": "Sb", "chunk_content": short_b, "chunk_id": 2}),
            (4003, 0.85, {"sample_id": "Sc", "chunk_content": short_c, "chunk_id": 3}),
        ],
        "sparse_embedding": [
            (
                4100,
                0.40,
                {"sample_id": "Sd", "chunk_content": long_sparse, "chunk_id": 4},
            ),
        ],
    }
    # Raw fused includes dense's short top-3 (RRF places them near the top
    # because their dense rank is high). Gate 1 will drop them.
    fused = [
        (
            4001,
            {
                "sample_id": "Sa",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": short_a,
                "chunk_id": 1,
            },
        ),
        (
            4002,
            {
                "sample_id": "Sb",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": short_b,
                "chunk_id": 2,
            },
        ),
        (
            4003,
            {
                "sample_id": "Sc",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": short_c,
                "chunk_id": 3,
            },
        ),
        (
            4100,
            {
                "sample_id": "Sd",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": long_sparse,
                "chunk_id": 4,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    # Only the long sparse hit survives Gate 1 in the fused view.
    assert [r.pk for r in table.rows] == [4100]

    # Dense's gated-out top-3 must surface as candidates so the user can
    # rate them — that's the candidate view's whole job.
    dense_cands = table.per_path_candidates["dense"]
    assert [c.pk for c in dense_cands] == [4001, 4002, 4003]
    assert [c.rank_in_path for c in dense_cands] == [1, 2, 3]


def test_run_search_short_chunk_gate_skips_per_path_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-path candidates must surface even when their chunks are below the
    50-char min — the candidate view exists so the user can see what fusion
    lost, and silently filtering short candidates makes a path that returns
    only short hits (common for dense on focused queries) invisible."""
    cfg = _make_config(per_path_limit=5, top_k=2)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    short_a = "家长：这个价格对我们来说有点贵了承受不起"  # 20 chars, well below 50
    short_b = "家长：但是别家机构现在好像便宜不少呀这让我怎么选"  # 24 chars
    short_c = "家长：报名之后真的会有效果吗我心里还有点没底"  # 22 chars
    long_fused = (
        "老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？"
        "方便后续制定计划，今天的电话先聊到这里。"
    )

    per_path = {
        "dense_embedding": [
            (3001, 0.95, {"sample_id": "Sa", "chunk_content": short_a, "chunk_id": 1}),
            (3002, 0.90, {"sample_id": "Sb", "chunk_content": short_b, "chunk_id": 2}),
            (3003, 0.85, {"sample_id": "Sc", "chunk_content": short_c, "chunk_id": 3}),
        ],
        "sparse_embedding": [
            (
                3100,
                0.40,
                {"sample_id": "Sd", "chunk_content": long_fused, "chunk_id": 4},
            ),
        ],
    }
    fused = [
        (
            3100,
            {
                "sample_id": "Sd",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": long_fused,
                "chunk_id": 4,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    # Fused row: only the long sparse hit survives the gate.
    assert [r.pk for r in table.rows] == [3100]

    # The dense top-3 are all short and all NOT in the fused set, so they
    # MUST surface as per-path candidates (the gate doesn't apply here).
    dense_cands = table.per_path_candidates["dense"]
    assert [c.pk for c in dense_cands] == [3001, 3002, 3003]
    assert [c.rank_in_path for c in dense_cands] == [1, 2, 3]
    assert dense_cands[0].chunk_content == short_a

    # ``filtered_short_chunk`` only counts fused-row drops now — there were
    # no short fused rows, so the counter is zero even though the per-path
    # candidates carry sub-50-char content.
    assert table.filtered_short_chunk == 0


def test_run_search_short_chunk_count_excludes_top_k_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``filtered_short_chunk`` must report the gate's drop
    count, not the combined drop from the gate plus the ``[:top_k]``
    slice that follows. With ``top_k=3`` and ``per_path_limit=5`` the
    over-fetched fused pool is 6; if 2 rows are short, 4 survive the
    gate and 3 reach the display — the metric must be 2, not the
    pre-fix 3 (= 6 fused − 3 displayed)."""
    cfg = _make_config(per_path_limit=5, top_k=3)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    long_tail = (
        "方便后续制定计划，今天的电话先聊到这里，期待和您下次再交流。"
        "孩子在课程上的进步我们看在眼里，下学期的安排也想跟您聊聊。"
    )
    long_a = "老师：A " + long_tail
    long_b = "老师：B " + long_tail
    long_c = "老师：C " + long_tail
    long_d = "老师：D " + long_tail
    short_e = "家长：嗯。"
    short_f = "家长：好的。"
    # Sanity-check the fixture matches the gate's invariant: 50 chars is
    # the production cut-off (search.py::_MIN_CHUNK_CHARS).
    assert all(len(s.strip()) >= 50 for s in (long_a, long_b, long_c, long_d))
    assert all(len(s.strip()) < 50 for s in (short_e, short_f))

    per_path = {
        "dense_embedding": [
            (5000, 0.99, {"sample_id": "Sa", "chunk_content": long_a, "chunk_id": 1}),
            (5001, 0.95, {"sample_id": "Sb", "chunk_content": long_b, "chunk_id": 2}),
            (5002, 0.90, {"sample_id": "Sc", "chunk_content": long_c, "chunk_id": 3}),
        ],
        "sparse_embedding": [
            (5003, 0.80, {"sample_id": "Sd", "chunk_content": long_d, "chunk_id": 4}),
            (5004, 0.70, {"sample_id": "Se", "chunk_content": short_e, "chunk_id": 5}),
            (5005, 0.60, {"sample_id": "Sf", "chunk_content": short_f, "chunk_id": 6}),
        ],
    }
    # Fused pool of 6: 4 long + 2 short. After Gate 1 → 4. After [:3] → 3.
    fused = [
        (
            5000,
            {
                "sample_id": "Sa",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": long_a,
                "chunk_id": 1,
            },
        ),
        (
            5001,
            {
                "sample_id": "Sb",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": long_b,
                "chunk_id": 2,
            },
        ),
        (
            5004,
            {
                "sample_id": "Se",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": short_e,
                "chunk_id": 5,
            },
        ),
        (
            5002,
            {
                "sample_id": "Sc",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": long_c,
                "chunk_id": 3,
            },
        ),
        (
            5005,
            {
                "sample_id": "Sf",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": short_f,
                "chunk_id": 6,
            },
        ),
        (
            5003,
            {
                "sample_id": "Sd",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": long_d,
                "chunk_id": 4,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    # Top-3 displayed are the long survivors in fused order.
    assert [r.pk for r in table.rows] == [5000, 5001, 5002]
    # The metric counts only the gate's drops (2 short rows), not the
    # 3 healthy rows that fell off the [:top_k] slice.
    assert table.filtered_short_chunk == 2


def test_run_search_computes_drop_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """search_diagnostics['drop_previews'] has one entry per active path
    with the simulated fused top-K excluding that path."""
    cfg = _make_config(per_path_limit=5, top_k=3)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    # Each PK gets a distinct sample_id so the simulator's sample_id
    # dedup gate doesn't collapse the previews (which would mask the
    # RRF-ordering signal this test is checking).
    long_chunk = (
        "老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？"
        "方便后续制定计划，今天的电话先聊到这里，期待和您下次再交流。"
    )
    entity_a = {"sample_id": "Sa", "chunk_content": "a\n" + long_chunk}
    entity_b = {"sample_id": "Sb", "chunk_content": "b\n" + long_chunk}
    entity_c = {"sample_id": "Sc", "chunk_content": "c\n" + long_chunk}

    # Dense-unique: 4000 (rank 1 dense only).
    # Shared across both paths: 4001 (rank 1 in dense + sparse).
    # Sparse-unique: 4002.
    per_path = {
        "dense_embedding": [
            (4000, 0.95, entity_a),
            (4001, 0.60, entity_b),
        ],
        "sparse_embedding": [
            (4001, 0.90, entity_b),
            (4002, 0.50, entity_c),
        ],
    }
    fused = [
        (
            4001,
            {
                "sample_id": "Sb",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": "b\n" + long_chunk,
                "chunk_id": 1,
            },
        ),
        (
            4000,
            {
                "sample_id": "Sa",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": "a\n" + long_chunk,
                "chunk_id": 0,
            },
        ),
        (
            4002,
            {
                "sample_id": "Sc",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": "c\n" + long_chunk,
                "chunk_id": 2,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    assert table.search_diagnostics is not None
    previews = table.search_diagnostics["drop_previews"]
    # One preview per currently-active path.
    assert set(previews.keys()) == {"dense", "sparse"}
    # Dropping dense: only sparse remains -> 4001, 4002.
    # 4000 is dense-only and must NOT appear.
    assert 4000 not in previews["dense"]
    assert previews["dense"][0] == 4001
    # Dropping sparse: only dense remains -> 4000, 4001.
    assert 4000 in previews["sparse"]
    assert 4002 not in previews["sparse"]
    # top_k respected.
    assert all(len(v) <= cfg.top_k for v in previews.values())


def test_run_search_drop_previews_skipped_for_single_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When <=1 active path, nothing meaningful to drop -> empty dict."""
    cfg = _make_config(
        per_path_limit=5,
        top_k=2,
        active_paths=frozenset({"dense"}),
    )
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(5001, 0.9, {})],
        "sparse_embedding": [(5001, 0.5, {})],
    }
    fused = [
        (
            5001,
            {
                "sample_id": "S",
                "counselor_id": "T",
                "term": "Tm",
                "chunk_content": "c\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)
    assert table.search_diagnostics is not None
    assert table.search_diagnostics["drop_previews"] == {}


def test_drop_previews_apply_post_rrf_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulator applies the same short-chunk + sample_id gates the live
    pipeline applies, so the diff against the actual top-K isolates the
    dropped-path effect.

    Setup: dense+sparse return the same three
    chunks, two of which would fail the gates (one too short, one a
    sample_id duplicate). The actual top-K and the "drop sparse" simulated
    top-K must be identical — otherwise the preview falsely reports
    losses/gains driven entirely by gate filtering.
    """
    cfg = _make_config(per_path_limit=5, top_k=3)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    long = (
        "老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？"
        "方便后续制定计划，今天的电话先聊到这里，期待和您下次再交流。"
    )
    keeper = {"sample_id": "Sa", "chunk_content": "kept\n" + long}
    duplicate = {"sample_id": "Sa", "chunk_content": "dup\n" + long}  # same sample_id
    too_short = {"sample_id": "Sb", "chunk_content": "短"}  # below 50-char gate

    per_path = {
        "dense_embedding": [
            (6000, 0.95, keeper),
            (6001, 0.80, duplicate),
            (6002, 0.60, too_short),
        ],
        "sparse_embedding": [
            (6000, 0.95, keeper),
            (6001, 0.80, duplicate),
            (6002, 0.60, too_short),
        ],
    }
    fused = [
        (6000, {**keeper, "counselor_id": "T", "term": "Tm", "chunk_id": 0}),
        (6001, {**duplicate, "counselor_id": "T", "term": "Tm", "chunk_id": 1}),
        (6002, {**too_short, "counselor_id": "T", "term": "Tm", "chunk_id": 2}),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    actual_pks = [row.pk for row in table.rows]
    # Live pipeline keeps 6000 only: 6001 is a sample_id dup, 6002 fails
    # the short-chunk gate.
    assert actual_pks == [6000]

    previews = table.search_diagnostics["drop_previews"]
    # Both active paths return the same 3 chunks; the simulator should
    # apply the same gates to its candidate pool, leaving 6000 as the
    # only chunk past the dedup gate.
    assert previews["dense"] == [6000] or previews["dense"] == []
    assert previews["sparse"] == [6000] or previews["sparse"] == []
    # The duplicate / short chunks must not appear in the simulated lists,
    # otherwise the diff would falsely flag them as gained.
    for path, pks in previews.items():
        assert 6001 not in pks, f"sample_id duplicate leaked into {path} preview"
        assert 6002 not in pks, f"short chunk leaked into {path} preview"


def test_run_search_with_inactive_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """When sparse is inactive, the fused call gets only 1 req and
    source_paths strips sparse even if the probe found the chunk."""
    cfg = _make_config(
        per_path_limit=5,
        top_k=2,
        active_paths=frozenset({"dense"}),
    )
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    # Chunk 3001 is found by dense + sparse (sparse probe still runs for
    # diagnostics); sparse must not appear in source_paths since it is
    # not in active_paths.
    per_path = {
        "dense_embedding": [(3001, 0.91, {})],
        "sparse_embedding": [(3001, 0.55, {})],
    }
    fused = [
        (
            3001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：test\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    # Fused call receives only 1 req — sparse is inactive.
    assert len(client.hybrid_calls) == 1
    assert client.hybrid_calls[0]["num_reqs"] == 1

    assert len(table.rows) == 1
    row = table.rows[0]
    # sparse was probed (diagnostic) but is filtered out of source_paths.
    assert row.source_paths == ["dense"]
    # The raw probe score is preserved for visibility.
    assert row.scores["sparse"] == 0.55

    # Diagnostics record active_paths.
    assert table.search_diagnostics is not None
    assert table.search_diagnostics["active_paths"] == ["dense"]


def test_run_search_invokes_extract_span_on_rows_and_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``extract_span`` is supplied, every fused row and every per-path
    candidate gets span fields populated and the diagnostics block records
    the call counts + length distribution."""
    cfg = _make_config(per_path_limit=5, top_k=2)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [
            (2001, 0.9, {}),
            (2002, 0.8, {}),
            (
                2003,
                0.7,
                {
                    "sample_id": "S3",
                    "chunk_content": "老师：candidate\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                    "chunk_id": 3,
                },
            ),
        ],
        "sparse_embedding": [],
    }
    fused = [
        (
            2001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：a\n家长：b\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        ),
        (
            2002,
            {
                "sample_id": "S2",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：c\n家长：d\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 2,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    calls: list[str] = []

    def extract(content: str) -> tuple[list[int], str]:
        calls.append(content)
        lines = content.split("\n")
        return [0], lines[0]

    table = run_search("q", cfg, client=client, extract_span=extract)

    # Every fused row carries the span.
    assert len(table.rows) == 2
    for row in table.rows:
        assert row.span_line_indices == [0]
        assert row.span_text == row.chunk_content.split("\n")[0]

    # Per-path candidate also extracted.
    dense_cand = table.per_path_candidates["dense"][0]
    assert dense_cand.span_line_indices == [0]
    assert dense_cand.span_text == "老师：candidate"

    # One call per rated chunk (2 rows + 1 candidate).
    assert len(calls) == 3

    # Diagnostics block matches.
    diag = table.search_diagnostics["span_extraction"]
    assert diag["chunks_extracted"] == 3
    assert diag["empty_span_count"] == 0
    assert diag["span_length_distribution"] == {"0": 0, "1": 3, "2": 0, "3": 0}


def test_run_search_empty_span_counted_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty-span returns count toward empty_span_count and the length[0] bucket."""
    cfg = _make_config(per_path_limit=5, top_k=1)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(3001, 0.9, {})],
        "sparse_embedding": [],
    }
    fused = [
        (
            3001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：a\n家长：b\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search(
        "q",
        cfg,
        client=client,
        extract_span=lambda _content: ([], ""),
    )

    assert table.rows[0].span_line_indices == []
    assert table.rows[0].span_text == ""
    diag = table.search_diagnostics["span_extraction"]
    assert diag["empty_span_count"] == 1
    assert diag["span_length_distribution"]["0"] == 1


def test_run_search_histogram_grows_for_dual_vendor_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for issue #54: a dual-vendor ``DualSpanExtractor``
    can return a 4–6 line union of two single-vendor ≤3-line spans,
    so ``span_length_distribution`` must accept bucket keys past the
    historical 0–3 cap rather than ``KeyError``-ing on first hit.
    The 0–3 buckets stay pre-seeded at zero (existing shape contract);
    new buckets appear lazily only when actually populated.
    """
    cfg = _make_config(per_path_limit=5, top_k=1)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(4001, 0.9, {})],
        "sparse_embedding": [],
    }
    fused = [
        (
            4001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                # Six-line chunk so [0,1,2,3] and [0,1,2,3,4,5] are
                # both legitimate dual-vendor union shapes.
                "chunk_content": "l0\nl1\nl2\nl3\nl4\nl5",
                "chunk_id": 1,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    # Simulate a dual-vendor union of length 4 — the exact crash
    # case from the local-session repro (KeyError: '4' at
    # search.py:502 pre-fix).
    table = run_search(
        "q",
        cfg,
        client=client,
        extract_span=lambda _content: ([0, 1, 2, 3], "l0\nl1\nl2\nl3"),
    )

    diag = table.search_diagnostics["span_extraction"]
    dist = diag["span_length_distribution"]
    assert dist["4"] == 1
    # Pre-seeded zero buckets remain present for shape stability.
    assert dist["0"] == 0
    assert dist["1"] == 0
    assert dist["2"] == 0
    assert dist["3"] == 0
    # Never-hit lengths are absent (lazy creation, not pre-seeded).
    assert "5" not in dist
    assert "6" not in dist


def test_run_search_drops_rows_when_extractor_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closure that returns None signals "extractor failed on this chunk"
    and run_search drops the row / candidate entirely rather than leaving
    it half-evaluated. The drop count surfaces on the evidence table."""
    cfg = _make_config(per_path_limit=5, top_k=3)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [
            (7001, 0.9, {}),
            (7002, 0.8, {}),
            (
                7003,
                0.7,
                {
                    "sample_id": "S3",
                    "chunk_content": "老师：candidate\n家长：ok\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                    "chunk_id": 3,
                },
            ),
        ],
        "sparse_embedding": [],
    }
    fused = [
        (
            7001,
            {
                "sample_id": "S1",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": "老师：a\n家长：b\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        ),
        (
            7002,
            {
                "sample_id": "S2",
                "counselor_id": "T",
                "term": "T",
                "chunk_content": "老师：drop me\n家长：y\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 2,
            },
        ),
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    def extract(content: str):
        if "drop me" in content:
            return None
        return [0], content.split("\n")[0]

    table = run_search("q", cfg, client=client, extract_span=extract)

    # Row 7002 dropped; the surviving row was re-ranked to rank 1.
    assert [r.pk for r in table.rows] == [7001]
    assert table.rows[0].rank == 1
    # Per-path candidate 7003 is fine (content doesn't say "drop me").
    assert len(table.per_path_candidates["dense"]) == 1
    assert table.dropped_by_extractor == 1
    diag = table.search_diagnostics["span_extraction"]
    assert diag["dropped_by_extractor"] == 1
    # Surviving rows/candidates were counted by the diagnostics block.
    assert diag["chunks_extracted"] == 2


def test_run_search_enriches_span_block_with_callable_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the ``extract_span`` callable exposes model / prompt_version /
    latency_ms_total / cache_hits as attributes, run_search folds them
    into ``search_diagnostics['span_extraction']`` so the sidecar
    logger surfaces the proposal's full observability block."""
    cfg = _make_config(per_path_limit=5, top_k=1)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(6001, 0.9, {})],
        "sparse_embedding": [],
    }
    fused = [
        (
            6001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：a\n家长：b\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    class _StubCallable:
        model = "mock/model"
        prompt_version = "v7"
        latency_ms_total = 12.5
        cache_hits = 2

        def __call__(self, content: str) -> tuple[list[int], str]:
            return [0], content.split("\n")[0]

    table = run_search("q", cfg, client=client, extract_span=_StubCallable())
    block = table.search_diagnostics["span_extraction"]
    assert block["model"] == "mock/model"
    assert block["prompt_version"] == "v7"
    assert block["latency_ms_total"] == 12.5
    assert block["cache_hits"] == 2


def test_run_search_skips_extractor_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy path: no extractor -> no span_extraction diagnostics, defaults on rows."""
    cfg = _make_config(per_path_limit=5, top_k=1)
    monkeypatch.setattr(search_module, "get_embeddings", _fake_embedding())
    monkeypatch.setattr(search_module, "RRFRanker", _RecordingRanker)

    per_path = {
        "dense_embedding": [(4001, 0.9, {})],
        "sparse_embedding": [],
    }
    fused = [
        (
            4001,
            {
                "sample_id": "S1",
                "counselor_id": "T1",
                "term": "T",
                "chunk_content": "老师：a\n家长：b\n老师：那我们再聊聊孩子最近的学习情况、兴趣爱好以及未来的学业规划好吗？方便后续制定计划。",
                "chunk_id": 1,
            },
        )
    ]
    client = _FakeClient(per_path_results=per_path, fused=fused)

    table = run_search("q", cfg, client=client)

    assert "span_extraction" not in table.search_diagnostics
    assert table.rows[0].span_line_indices == []
    assert table.rows[0].span_text == ""


def test_evidence_table_set_rating_unknown_rank() -> None:
    cfg = _make_config()
    table = EvidenceTable(
        query="q",
        config=cfg,
        rows=[_row(1, ["dense"], None)],
    )

    table.set_rating(1, "FIT")
    assert table.rows[0].rating == "FIT"

    with pytest.raises(KeyError):
        table.set_rating(99, "FIT")


def test_evidence_table_set_rating_discard() -> None:
    """DISCARD is a valid rating value end-to-end (issue #46)."""
    cfg = _make_config()
    table = EvidenceTable(
        query="q",
        config=cfg,
        rows=[_row(1, ["dense"], None)],
        per_path_candidates={
            "dense": [],
            "sparse": [
                CandidateRow(
                    path="sparse",
                    rank_in_path=1,
                    pk=42,
                    chunk_id="S1_C0000042",
                    chunk_content="garbled STT",
                    sample_id="S1",
                    counselor_id="T1",
                    term="T",
                    score=0.1,
                )
            ],
        },
    )

    table.set_rating(1, "DISCARD")
    assert table.rows[0].rating == "DISCARD"

    table.set_candidate_rating("sparse", 1, "DISCARD")
    assert table.per_path_candidates["sparse"][0].rating == "DISCARD"

    # Both row and candidate count as "rated" — all_rated() passes.
    assert table.all_rated() is True
    assert table.all_rated(include_candidates=True) is True


# ---------------------------------------------------------------------------
# Integration test — live services
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_search_live() -> None:
    """End-to-end against real BGE-M3 + Milvus. Skipped by default."""

    cfg = load_default_config()
    table = run_search("家长对价格的犹豫和老师的应对话术", cfg)

    assert len(table.rows) == cfg.top_k
    assert all(row.chunk_content for row in table.rows)
    assert all(len(row.source_paths) >= 1 for row in table.rows)
    # At least one row should carry non-zero scores on the path(s) it came from.
    assert any(any(score > 0 for score in row.scores.values()) for row in table.rows)
