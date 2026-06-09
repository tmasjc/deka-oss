"""Tests for src.refine.derive._apply_example_caps — the FIT / NOT_FIT
cap that runs before the rubric meta-prompt is rendered."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.refine import derive as derive_mod
from src.refine.config import RefineConfig
from src.refine.derive import _apply_example_caps
from src.refine.errors import RefineError
from src.refine.load_session import (
    FitExemplar,
    NotFitExemplar,
    Phase3SessionInputs,
)
from src.search.config import SearchConfig


def _search_config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=10,
        embed_url="http://embed.test",
        milvus_uri="http://milvus.test",
        http_timeout=5,
    )


def _cfg(*, max_fit: int, max_not_fit: int) -> RefineConfig:
    return RefineConfig(
        enabled=True,
        sample_size=2,
        n_bins=2,
        seed=0,
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        max_fit_examples=max_fit,
        max_not_fit_examples=max_not_fit,
        derive_model="d",
        derive_base_url="x",
        derive_temperature=0.2,
        judge_model="j",
        judge_base_url="x",
        judge_concurrency=2,
        judge_qps_limit=1,
        judge_tpm_limit=1,
        judge_timeout_seconds=10,
        judge_max_retries=0,
        api_key_env="X",
        auto_drop_known_intruders=True,
    )


def _fit(i: int) -> FitExemplar:
    return FitExemplar(
        pk=i,
        chunk_id=f"chunk-{i}",
        span_text=f"span text {i}",
        span_line_indices=[1],
    )


def _not_fit(i: int) -> NotFitExemplar:
    return NotFitExemplar(
        pk=i,
        chunk_id=f"chunk-{i}",
        chunk_content=f"chunk content {i}",
    )


def _inputs(n_fit: int, n_not_fit: int) -> Phase3SessionInputs:
    return Phase3SessionInputs(
        session_id="s",
        query="q",
        fits=[_fit(i) for i in range(1, n_fit + 1)],
        not_fits=[_not_fit(i) for i in range(101, 101 + n_not_fit)],
        rated_pks=frozenset(),
        reflection_diagnoses=[],
    )


def test_no_embed_call_when_within_caps(monkeypatch):
    """Pool sizes within cap → no embed request, inputs returned as-is."""

    calls: list[list[str]] = []

    def boom(*args, **kwargs):
        calls.append(list(args))
        raise AssertionError("get_embeddings must not be called")

    monkeypatch.setattr(derive_mod, "get_embeddings", boom)
    cfg = _cfg(max_fit=6, max_not_fit=6)
    inputs = _inputs(n_fit=3, n_not_fit=2)
    out = _apply_example_caps(inputs, cfg, _search_config())
    assert out is inputs
    assert calls == []


def test_caps_fits_when_pool_exceeds_cap(monkeypatch):
    """FIT pool of 10, cap of 4 → embed called once with the FIT spans
    and the returned pool has exactly 4 entries."""

    captured: dict[str, object] = {}

    def fake_embed(sentences, embed_url, *, timeout):
        captured["sentences"] = list(sentences)
        captured["embed_url"] = embed_url
        captured["timeout"] = timeout
        # Vectors spread on the unit circle so farthest-first picks
        # distinct items.
        import math

        return {
            "dense": [
                [math.cos(i * 0.6), math.sin(i * 0.6)] for i in range(len(sentences))
            ]
        }

    monkeypatch.setattr(derive_mod, "get_embeddings", fake_embed)
    cfg = _cfg(max_fit=4, max_not_fit=6)
    inputs = _inputs(n_fit=10, n_not_fit=2)
    out = _apply_example_caps(inputs, cfg, _search_config())
    assert len(out.fits) == 4
    # NOT_FITs untouched (within cap → no second embed call).
    assert len(out.not_fits) == 2
    assert captured["sentences"] == [f"span text {i}" for i in range(1, 11)]
    assert captured["embed_url"] == "http://embed.test"
    assert captured["timeout"] == 5


def test_caps_not_fits_when_pool_exceeds_cap(monkeypatch):
    """NOT_FIT pool of 8, cap of 3 → embed call uses chunk_content text
    and the returned pool has exactly 3 entries."""

    captured: dict[str, object] = {}

    def fake_embed(sentences, embed_url, *, timeout):
        captured["sentences"] = list(sentences)
        import math

        return {
            "dense": [
                [math.cos(i * 0.4), math.sin(i * 0.4)] for i in range(len(sentences))
            ]
        }

    monkeypatch.setattr(derive_mod, "get_embeddings", fake_embed)
    cfg = _cfg(max_fit=6, max_not_fit=3)
    inputs = _inputs(n_fit=2, n_not_fit=8)
    out = _apply_example_caps(inputs, cfg, _search_config())
    assert len(out.fits) == 2
    assert len(out.not_fits) == 3
    assert captured["sentences"] == [f"chunk content {i}" for i in range(101, 109)]


def test_embed_service_error_raises_refine_error(monkeypatch):
    """Failures from the embed service must surface as RefineError so
    the operator sees a Phase-3-side message, not a transport error."""

    from src.search.errors import EmbeddingServiceError

    def fake_embed(*args, **kwargs):
        raise EmbeddingServiceError("connection refused")

    monkeypatch.setattr(derive_mod, "get_embeddings", fake_embed)
    cfg = _cfg(max_fit=2, max_not_fit=6)
    inputs = _inputs(n_fit=5, n_not_fit=1)
    with pytest.raises(RefineError, match="connection refused"):
        _apply_example_caps(inputs, cfg, _search_config())


def test_accepts_bge_m3_dense_embeddings_key(monkeypatch):
    """Live BGE-M3 /embed-all returns ``dense_embeddings``; legacy/mocks
    use ``dense``. The selector must accept either, matching the anchor
    loader's contract (src/anchor/loader.py:466-470)."""

    def fake_embed(sentences, *args, **kwargs):
        import math

        return {
            "dense_embeddings": [
                [math.cos(i * 0.6), math.sin(i * 0.6)] for i in range(len(sentences))
            ]
        }

    monkeypatch.setattr(derive_mod, "get_embeddings", fake_embed)
    cfg = _cfg(max_fit=4, max_not_fit=6)
    inputs = _inputs(n_fit=10, n_not_fit=2)
    out = _apply_example_caps(inputs, cfg, _search_config())
    assert len(out.fits) == 4


def test_malformed_dense_response_raises_refine_error(monkeypatch):
    """The selection algorithm needs one vector per text. A short or
    non-list response is a service bug; surface it as RefineError."""

    def fake_embed(sentences, *args, **kwargs):
        return {"dense": [[1.0, 0.0]]}  # only one vector for >1 sentence

    monkeypatch.setattr(derive_mod, "get_embeddings", fake_embed)
    cfg = _cfg(max_fit=2, max_not_fit=6)
    inputs = _inputs(n_fit=5, n_not_fit=1)
    with pytest.raises(RefineError, match="malformed"):
        _apply_example_caps(inputs, cfg, _search_config())


def test_loads_default_search_config_when_none_passed(monkeypatch):
    """search_config=None → call site loads the default. A no-cap path
    skips that load entirely, but the cap path needs the default."""

    loaded = {"called": 0}

    def fake_load_default():
        loaded["called"] += 1
        return _search_config()

    def fake_embed(sentences, *args, **kwargs):
        import math

        return {
            "dense": [
                [math.cos(i * 0.5), math.sin(i * 0.5)] for i in range(len(sentences))
            ]
        }

    monkeypatch.setattr(derive_mod, "load_default_config", fake_load_default)
    monkeypatch.setattr(derive_mod, "get_embeddings", fake_embed)

    # No-cap path: default loader must NOT be called.
    cfg = _cfg(max_fit=10, max_not_fit=10)
    inputs = _inputs(n_fit=2, n_not_fit=2)
    _apply_example_caps(inputs, cfg, None)
    assert loaded["called"] == 0

    # Cap path: default loader called once for the whole call.
    cfg = _cfg(max_fit=2, max_not_fit=10)
    inputs = _inputs(n_fit=5, n_not_fit=2)
    _apply_example_caps(inputs, cfg, None)
    assert loaded["called"] == 1


def test_replace_preserves_other_fields():
    """Capping only mutates the two list fields — everything else is
    pass-through."""

    inputs = _inputs(n_fit=2, n_not_fit=2)
    inputs = replace(inputs, reflection_diagnoses=["foo"], rated_pks=frozenset({1, 2}))
    cfg = _cfg(max_fit=6, max_not_fit=6)
    out = _apply_example_caps(inputs, cfg, _search_config())
    assert out.reflection_diagnoses == ["foo"]
    assert out.rated_pks == frozenset({1, 2})
    assert out.session_id == inputs.session_id
    assert out.query == inputs.query
