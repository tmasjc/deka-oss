"""Unit tests for src.anchor.loader — span-anchored input loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.anchor.errors import AnchorLoadError
from src.anchor.loader import (
    AnchorInputs,
    load_anchor_inputs,
)


class _FakeMilvus:
    """Minimal MilvusClient replacement for loader tests."""

    def __init__(
        self,
        *,
        chunk_embeddings: dict[str, list[float]],
        index_info: dict[str, Any] | None = None,
    ) -> None:
        self._chunk_embeddings = chunk_embeddings
        self._index_info = index_info or {
            "index_type": "HNSW",
            "params": {"ef": 64, "M": 16},
        }
        self.calls: list[dict[str, Any]] = []

    def query(self, **kwargs):
        self.calls.append({"op": "query", **kwargs})
        # Filter pks out of `filter` expr — simple string extraction
        # is enough for tests.
        expr = kwargs.get("filter", "")
        rows = []
        for pk, vec in self._chunk_embeddings.items():
            if f'"{pk}"' in expr or str(pk) in expr:
                rows.append({"id": pk, "dense_embedding": vec})
        return rows

    def list_indexes(self, collection_name: str) -> list[str]:
        # Production pymilvus returns a list of index names; production
        # ``_read_index_info`` resolves the dense_embedding index via this
        # call before describing it.
        return ["dense_embedding", "sparse_embedding"]

    def describe_index(self, collection_name: str, index_name: str):
        # No default on ``index_name`` — pymilvus's real signature
        # requires it, so a regression that drops the kwarg fails here
        # with a TypeError instead of silently passing.
        # pymilvus returns a list of dicts or a single dict depending on
        # version; returning the dict is enough for our reader.
        return self._index_info

    def close(self) -> None:  # noqa: D401 - mock
        pass


def _fake_embedder(mapping: dict[str, list[float]], *, model_id: str = "bge-m3"):
    """Fake embed-service client returning canned vectors."""

    def _call(span_texts: list[str], embed_url: str, *, timeout: int):
        assert embed_url == "http://localhost:8000"
        return {
            "dense": [mapping[t] for t in span_texts],
            "model_id": model_id,
        }

    return _call


def _fake_model_id_reader(model_id: str):
    def _call(embed_url: str, *, timeout: int = 5) -> str:
        assert embed_url == "http://localhost:8000"
        return model_id

    return _call


def test_load_anchor_inputs_happy_path(session_dir: Path, monkeypatch):
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 1.0, 0.0],
    }
    from src.anchor import loader

    monkeypatch.setattr(loader, "_get_embeddings", _fake_embedder(spans))
    monkeypatch.setattr(loader, "_read_embed_model_id", _fake_model_id_reader("bge-m3"))

    client = _FakeMilvus(chunk_embeddings=chunks)

    inputs = load_anchor_inputs(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        allow_unconverged=True,
    )

    assert isinstance(inputs, AnchorInputs)
    assert inputs.session_id == "SESS-TEST"
    assert inputs.collection == "parent_transcripts"
    assert inputs.embed_url == "http://localhost:8000"
    assert inputs.embed_model_id == "bge-m3"
    # Regression for B6.1: ``_read_index_info`` must propagate the actual
    # Milvus index type/params (not the UNKNOWN fallback) when
    # ``list_indexes`` + ``describe_index`` are available with the real
    # pymilvus signature.
    assert inputs.milvus_index_type == "HNSW"
    assert inputs.milvus_index_params == {"ef": 64, "M": 16}
    assert {f.pk for f in inputs.fits} == {"pk-A", "pk-B", "pk-D"}
    by_pk = {f.pk: f for f in inputs.fits}
    assert by_pk["pk-A"].span_text == "span for A"
    assert by_pk["pk-A"].span_line_indices == [0, 1]
    assert by_pk["pk-A"].span_embedding == [1.0, 0.0, 0.0, 0.0]
    assert by_pk["pk-A"].chunk_embedding == [1.0, 0.0, 0.0, 0.0]
    assert "pk-C" in inputs.not_fit_pks
    # The fixture session has no DISCARD rows, so the set is empty —
    # but the field must exist (issue #46) so the runner can filter on it.
    assert inputs.discard_pks == frozenset()


def test_load_anchor_inputs_populates_discard_pks(
    tmp_path: Path, monkeypatch, session_dir: Path
):
    """A Phase-1 DISCARD row surfaces as ``inputs.discard_pks`` so the
    Phase-2 runner can drop it from the harvest output (issue #46)."""
    import json

    # Patch the fixture progress log to include a DISCARD row.
    progress = session_dir / "SESS-TEST.jsonl"
    lines = progress.read_text(encoding="utf-8").splitlines()
    turn1 = json.loads(lines[0])
    turn1["results"].append(
        {
            "rank": 4,
            "pk": "pk-DISC",
            "chunk_id": "S1_C0000004",
            "chunk_content": "garbled STT",
            "sample_id": "S1",
            "counselor_id": "C-1",
            "term": "2025F",
            "rating": "DISCARD",
            "source_paths": ["dense"],
            "scores": {"dense": 0.3, "sparse": 0.0},
            "span_text": "",
            "span_line_indices": [],
        }
    )
    lines[0] = json.dumps(turn1)
    progress.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 1.0, 0.0],
    }
    from src.anchor import loader

    monkeypatch.setattr(loader, "_get_embeddings", _fake_embedder(spans))
    monkeypatch.setattr(loader, "_read_embed_model_id", _fake_model_id_reader("bge-m3"))

    client = _FakeMilvus(chunk_embeddings=chunks)
    inputs = load_anchor_inputs(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        allow_unconverged=True,
    )

    assert inputs.discard_pks == frozenset({"pk-DISC"})
    # DISCARD never bleeds into the FIT cohort or the NOT_FIT pool.
    assert "pk-DISC" not in {f.pk for f in inputs.fits}
    assert "pk-DISC" not in inputs.not_fit_pks


def test_load_raises_when_no_fits(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "EMPTY.jsonl").write_text(
        json.dumps(
            {
                "turn": 1,
                "timestamp": "2026-04-23T00:00:00Z",
                "session_id": "EMPTY",
                "query": "q",
                "config": {
                    "rrf_k": 60,
                    "per_path_limit": 20,
                    "top_k": 10,
                    "active_paths": ["dense"],
                },
                "results": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Replay loader also requires a details sidecar.
    (runs / "EMPTY.details.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(AnchorLoadError, match="no FIT"):
        load_anchor_inputs("EMPTY", runs_dir=runs)


def test_load_drops_fits_with_empty_span_text(session_dir: Path, monkeypatch, caplog):
    """Missing-span rows are dropped with a warning, not an abort."""
    log = session_dir / "SESS-TEST.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    # pk-A now has no span; pk-B keeps its span. Turn-2 pk-D also has span.
    row["results"][0]["span_text"] = ""
    lines[0] = json.dumps(row)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from src.anchor import loader

    # pk-A is dropped pre-embed, so the embedder only sees the
    # remaining two spans.
    monkeypatch.setattr(
        loader,
        "_get_embeddings",
        _fake_embedder(
            {
                "span for B": [0.9, 0.1, 0.0, 0.0],
                "span for D": [0.0, 0.0, 1.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(loader, "_read_embed_model_id", _fake_model_id_reader("bge-m3"))
    client = _FakeMilvus(
        chunk_embeddings={
            "pk-B": [0.9, 0.1, 0.0, 0.0],
            "pk-D": [0.0, 0.0, 1.0, 0.0],
        }
    )

    with caplog.at_level("WARNING", logger="src.anchor.loader"):
        inputs = load_anchor_inputs(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            allow_unconverged=True,
        )
    assert {f.pk for f in inputs.fits} == {"pk-B", "pk-D"}
    assert any("empty span fields" in rec.message for rec in caplog.records)


def test_load_raises_when_dropping_empty_spans_leaves_fewer_than_two(
    session_dir: Path, monkeypatch
):
    """Dropping to <2 usable FITs is a hard error (calibration floor)."""
    log = session_dir / "SESS-TEST.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    # Blank both turn-1 FITs (pk-A, pk-B). Only turn-2 pk-D survives.
    row = json.loads(lines[0])
    row["results"][0]["span_text"] = ""
    row["results"][1]["span_text"] = ""
    lines[0] = json.dumps(row)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AnchorLoadError, match="at least 2"):
        load_anchor_inputs("SESS-TEST", runs_dir=session_dir, allow_unconverged=True)


def test_load_raises_on_model_mismatch(session_dir: Path, monkeypatch):
    from src.anchor import loader

    monkeypatch.setattr(loader, "_get_embeddings", _fake_embedder({}))
    # Embed service reports a different model than what details sidecar
    # would record (we store model id on first load — see loader
    # logic).
    monkeypatch.setattr(
        loader, "_read_embed_model_id", _fake_model_id_reader("other-model")
    )

    # First, bake an expected model id into details by writing a
    # details sidecar with `embed_model_id` field on the search block.
    details = session_dir / "SESS-TEST.details.jsonl"
    rows = [json.loads(ln) for ln in details.read_text(encoding="utf-8").splitlines()]
    rows[0]["search"]["embed_model_id"] = "bge-m3"
    details.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    with pytest.raises(AnchorLoadError, match="model.*drift|mismatch"):
        load_anchor_inputs(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=_FakeMilvus(chunk_embeddings={}),
            allow_unconverged=True,
        )


def test_load_raises_on_missing_chunk_embedding(session_dir: Path, monkeypatch):
    from src.anchor import loader

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
    }
    monkeypatch.setattr(loader, "_get_embeddings", _fake_embedder(spans))
    monkeypatch.setattr(loader, "_read_embed_model_id", _fake_model_id_reader("bge-m3"))

    # Milvus returns only pk-A and pk-B; pk-D is missing.
    client = _FakeMilvus(
        chunk_embeddings={
            "pk-A": [1.0, 0.0, 0.0, 0.0],
            "pk-B": [0.9, 0.1, 0.0, 0.0],
        }
    )

    with pytest.raises(AnchorLoadError, match="pk-D"):
        load_anchor_inputs(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            allow_unconverged=True,
        )


class TestReadIndexInfo:
    """Direct tests for ``_read_index_info`` covering the fallback paths.

    The happy-path is asserted via ``test_load_anchor_inputs_happy_path``;
    these cases exercise the warning-and-fallback branches that exist
    so a reproducibility miss never aborts the run.
    """

    def test_returns_index_type_and_params_for_dense_embedding(self) -> None:
        from src.anchor.loader import _read_index_info

        class _Strict:
            def list_indexes(self, collection_name: str) -> list[str]:
                return ["sparse_embedding", "dense_embedding"]

            def describe_index(self, collection_name: str, index_name: str):
                assert index_name == "dense_embedding", (
                    "must prefer dense_embedding when present"
                )
                return {"index_type": "AUTOINDEX", "params": {"M": 8}}

        index_type, params = _read_index_info(_Strict(), "any")
        assert index_type == "AUTOINDEX"
        assert params == {"M": 8}

    def test_falls_back_to_first_index_when_dense_absent(self) -> None:
        from src.anchor.loader import _read_index_info

        class _NoDense:
            def list_indexes(self, collection_name: str) -> list[str]:
                return ["custom_idx_a", "custom_idx_b"]

            def describe_index(self, collection_name: str, index_name: str):
                assert index_name == "custom_idx_a"
                return {"index_type": "IVF_FLAT", "params": {"nlist": 1024}}

        index_type, params = _read_index_info(_NoDense(), "any")
        assert index_type == "IVF_FLAT"
        assert params == {"nlist": 1024}

    def test_returns_unknown_when_list_indexes_raises(self) -> None:
        from src.anchor.loader import _read_index_info

        class _Broken:
            def list_indexes(self, collection_name: str) -> list[str]:
                raise RuntimeError("rpc failed")

            def describe_index(self, collection_name: str, index_name: str):
                raise AssertionError("must not be called")

        index_type, params = _read_index_info(_Broken(), "any")
        assert index_type == "UNKNOWN"
        assert params == {}

    def test_returns_unknown_when_no_indexes_listed(self) -> None:
        from src.anchor.loader import _read_index_info

        class _Empty:
            def list_indexes(self, collection_name: str) -> list[str]:
                return []

            def describe_index(self, collection_name: str, index_name: str):
                raise AssertionError("must not be called")

        index_type, params = _read_index_info(_Empty(), "any")
        assert index_type == "UNKNOWN"
        assert params == {}

    def test_returns_unknown_when_client_lacks_methods(self) -> None:
        from src.anchor.loader import _read_index_info

        class _Bare:
            pass

        index_type, params = _read_index_info(_Bare(), "any")
        assert index_type == "UNKNOWN"
        assert params == {}
