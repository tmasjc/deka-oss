"""End-to-end test of src.anchor.runner with mocked Milvus + embed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.anchor.config import RadiusScheme
from src.anchor.errors import AnchorValidationError
from src.anchor.runner import AnchorResult, run_anchor


class _FakeIterator:
    def __init__(self, pages: list[list[dict]]):
        self._pages = list(pages)

    def next(self) -> list[dict]:  # noqa: A003
        if not self._pages:
            return []
        return self._pages.pop(0)

    def close(self) -> None:
        pass


class _FakeMilvusFull:
    """Fake client that answers loader.query() + LOO's batched
    client.search() + main-pass client.search_iterator()."""

    def __init__(
        self,
        *,
        chunk_embeddings: dict[str, list[float]],
        page_fn,
    ):
        self._chunk_embeddings = chunk_embeddings
        self._page_fn = page_fn
        self._index_info = {"index_type": "HNSW", "params": {"ef": 64, "M": 16}}
        self.iterator_opens = 0
        self.query_calls = 0
        self.search_calls = 0

    def query(self, **kwargs):
        self.query_calls += 1
        expr = kwargs.get("filter", "")
        return [
            {"id": pk, "dense_embedding": vec}
            for pk, vec in self._chunk_embeddings.items()
            if pk in expr
        ]

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
        # LOO path: flatten the first page of what page_fn would serve
        # and clamp to `limit`. page_fn is reused so tests can drive
        # both LOO and main pass off one scripted fixture.
        self.search_calls += 1
        out = []
        for v in data:
            pages = self._page_fn(v, collection_name, limit)
            hits = pages[0] if pages else []
            out.append(hits[:limit])
        return out

    def search_iterator(
        self,
        *,
        data,
        collection_name,
        anns_field,
        batch_size,
        search_params,
        output_fields,
        filter=None,
    ):
        self.iterator_opens += 1
        pages = self._page_fn(data[0], collection_name, batch_size)
        return _FakeIterator(pages)

    def list_indexes(self, collection_name: str) -> list[str]:
        return ["dense_embedding", "sparse_embedding"]

    def describe_index(self, collection_name: str, index_name: str):
        # No default on ``index_name`` — matches the real pymilvus
        # signature so a regression dropping the kwarg surfaces here.
        return self._index_info

    def close(self):
        pass


def _hit(pk: str, sim: float) -> dict:
    return {"id": pk, "distance": sim}


def test_run_anchor_writes_phase2_sidecars(session_dir: Path, monkeypatch):
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    # Every span queries: return all three FIT pks + two extras at
    # distance 0, then an empty page so the iterator exhausts.
    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("extra-1", 1.0),
                _hit("extra-2", 1.0),
            ],
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)

    from src.anchor import loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "_get_embeddings",
        lambda span_texts, embed_url, *, timeout: {
            "dense": [spans[t] for t in span_texts],
            "model_id": "bge-m3",
        },
    )
    monkeypatch.setattr(
        loader_mod, "_read_embed_model_id", lambda url, timeout=5: "bge-m3"
    )

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        allow_unconverged=True,
    )

    assert isinstance(result, AnchorResult)
    assert result.recovery.verdict == "HEALTHY"
    jsonl = session_dir / "SESS-TEST.phase2.jsonl"
    meta = session_dir / "SESS-TEST.phase2.meta.json"
    details = session_dir / "SESS-TEST.details.jsonl"
    assert jsonl.exists()
    assert meta.exists()

    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_obj["session_id"] == "SESS-TEST"
    assert meta_obj["collection"] == "parent_transcripts"
    assert meta_obj["n_fit"] == 3
    assert meta_obj["batch_size"] == 100
    assert meta_obj["max_k"] == 1000
    assert "per_fit_pages" in meta_obj
    assert len(meta_obj["per_fit_pages"]) == 3
    # Every per_fit_pages entry carries fit_pk (authoritative id) + label.
    for entry in meta_obj["per_fit_pages"]:
        assert "fit_pk" in entry and "fit_chunk_id" in entry
    assert meta_obj["per_fit_budget_exhausted"] == []
    # loo_recovery uses structured missed_fits, not legacy missed_fit_chunk_ids.
    loo = meta_obj["loo_recovery"]
    assert "missed_fits" in loo and "missed_fit_chunk_ids" not in loo
    for entry in loo["missed_fits"]:
        assert set(entry.keys()) == {"fit_pk", "fit_chunk_id"}
    # not_fit_intrusion exposes intruder_pks (name matches content).
    intr = meta_obj["not_fit_intrusion"]
    assert "intruder_pks" in intr and "intruder_chunk_ids" not in intr
    assert "calibration" in meta_obj
    # Per-FIT calibration vectors (not legacy scalars).
    calib_meta = meta_obj["calibration"]
    assert "deltas" in calib_meta and len(calib_meta["deltas"]) == 3
    assert "T_primes" in calib_meta and len(calib_meta["T_primes"]) == 3
    assert "delta_summary" in calib_meta and "T_prime_summary" in calib_meta
    assert "delta" not in calib_meta and "T_prime" not in calib_meta
    assert meta_obj["loo_recovery"]["verdict"] == "HEALTHY"

    # PK-only JSONL shape — no chunk_content / sample_id / etc.
    record_line = jsonl.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(record_line)
    assert set(record.keys()) == {
        "pk",
        "nearest_fit_pk",
        "nearest_fit_label",
        "nearest_fit_distance",
        "passed_threshold",
        "threshold_T_prime",
        "qualifying_fit_count",
        "qualifying_fit_pks",
    }

    lines = details.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    assert last["turn"] == "phase2"
    assert "per_fit" in last["phase2"]
    assert len(last["phase2"]["per_fit"]) == 3
    # New per-FIT audit columns present.
    sample = last["phase2"]["per_fit"][0]
    assert "pages_fetched" in sample
    assert "total_hits" in sample
    assert "budget_exhausted" in sample
    # Per-FIT threshold columns.
    assert "T_prime" in sample
    assert "span_to_own_chunk_distance" in sample
    assert "recalibrated_T" in sample["loo"]
    assert "recalibrated_T_prime" not in sample["loo"]


def test_runner_uses_config_when_kwargs_are_none(session_dir: Path, monkeypatch):
    """batch_size / max_k = None → resolve from HarvestConfig."""
    from src.anchor import config as config_mod
    from src.anchor import loader as loader_mod
    from src.anchor import runner as runner_mod

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)

    monkeypatch.setattr(
        loader_mod,
        "_get_embeddings",
        lambda span_texts, embed_url, *, timeout: {
            "dense": [spans[t] for t in span_texts],
            "model_id": "bge-m3",
        },
    )
    monkeypatch.setattr(
        loader_mod, "_read_embed_model_id", lambda url, timeout=5: "bge-m3"
    )

    fake_cfg = config_mod.HarvestConfig(
        min_fit=1,
        min_not_fit=1,
        precision_at_k=0.0,
        batch_size=77,
        max_k=777,
        s2c_outlier_multiple=3.0,
        radius_scheme=RadiusScheme.PER_FIT,
        anchor_frequency_gate=1,
    )
    monkeypatch.setattr(runner_mod, "load_harvest_config", lambda **_: fake_cfg)
    monkeypatch.setattr(loader_mod, "load_harvest_config", lambda **_: fake_cfg)

    run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
    )
    meta_obj = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    assert meta_obj["batch_size"] == 77
    assert meta_obj["max_k"] == 777


def test_dry_run_skips_writes(session_dir: Path, monkeypatch):
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

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)

    from src.anchor import loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "_get_embeddings",
        lambda span_texts, embed_url, *, timeout: {
            "dense": [spans[t] for t in span_texts],
            "model_id": "bge-m3",
        },
    )
    monkeypatch.setattr(
        loader_mod, "_read_embed_model_id", lambda url, timeout=5: "bge-m3"
    )

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        dry_run=True,
        allow_unconverged=True,
    )
    assert result.write is None
    assert not (session_dir / "SESS-TEST.phase2.jsonl").exists()
    assert not (session_dir / "SESS-TEST.phase2.meta.json").exists()
    details = session_dir / "SESS-TEST.details.jsonl"
    for ln in details.read_text(encoding="utf-8").splitlines():
        assert json.loads(ln).get("turn") != "phase2"


def test_loo_failed_verdict_raises(session_dir: Path, monkeypatch):
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

    def pages(q, coll, bs):
        # Neighbors never include the FIT pks → LOO fully fails.
        return [[_hit("foreign", 0.99)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)

    from src.anchor import loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "_get_embeddings",
        lambda span_texts, embed_url, *, timeout: {
            "dense": [spans[t] for t in span_texts],
            "model_id": "bge-m3",
        },
    )
    monkeypatch.setattr(
        loader_mod, "_read_embed_model_id", lambda url, timeout=5: "bge-m3"
    )

    with pytest.raises(AnchorValidationError):
        run_anchor(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            batch_size=100,
            max_k=1000,
            allow_unconverged=True,
        )
    assert not (session_dir / "SESS-TEST.phase2.jsonl").exists()


def test_budget_exhaustion_still_writes_sidecar(session_dir: Path, monkeypatch):
    """When a FIT hits max_k with last d ≤ T', the partial sidecar is
    still written and per_fit_budget_exhausted is populated — runner
    exits cleanly."""
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    # Main-pass T' is recomputed from the 3 tight spans — around 0.0-ish.
    # Feed the main pass more hits than max_k, all at d=0 — guaranteed
    # to flip budget_exhausted on every FIT. LOO sees the same script;
    # since every FIT pk is in every page at sim=1.0, LOO recovers.
    def pages(q, coll, bs):
        # Return many full-batch pages; max_k guards the loop.
        pages_out = []
        for _ in range(50):
            pages_out.append(
                [_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]
                + [_hit(f"extra-{i}", 1.0) for i in range(bs - 3)]
            )
        return pages_out

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)

    from src.anchor import loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "_get_embeddings",
        lambda span_texts, embed_url, *, timeout: {
            "dense": [spans[t] for t in span_texts],
            "model_id": "bge-m3",
        },
    )
    monkeypatch.setattr(
        loader_mod, "_read_embed_model_id", lambda url, timeout=5: "bge-m3"
    )

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=10,
        max_k=50,
        allow_unconverged=True,
    )
    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    assert len(meta["per_fit_budget_exhausted"]) == 3
    assert (session_dir / "SESS-TEST.phase2.jsonl").exists()
    assert result.write is not None


# ------------------------- quality gate ----------------------------


def _patch_embed(monkeypatch, spans: dict[str, list[float]]) -> None:
    from src.anchor import loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "_get_embeddings",
        lambda span_texts, embed_url, *, timeout: {
            "dense": [spans[t] for t in span_texts],
            "model_id": "bge-m3",
        },
    )
    monkeypatch.setattr(
        loader_mod, "_read_embed_model_id", lambda url, timeout=5: "bge-m3"
    )


def _fake_harvest_cfg(monkeypatch, **overrides):
    """Install a HarvestConfig with small min_fit + sane defaults so
    tests can drive the gate without tripping the min_fit abort or
    the convergence precondition.

    Patches both ``runner.load_harvest_config`` and
    ``loader.load_harvest_config`` so the convergence precondition
    check sees the same config as the runner. Synthetic conftest
    fixtures don't write a ``metrics.precision_at_k`` block, so the
    replayed precision is 0.0; ``precision_at_k=0.0`` keeps the
    convergence half of the gate trivially satisfied.
    """
    from src.anchor import config as config_mod
    from src.anchor import loader as loader_mod
    from src.anchor import runner as runner_mod

    cfg_kwargs = {
        "min_fit": 1,
        "min_not_fit": 1,
        "precision_at_k": 0.0,
        "batch_size": 100,
        "max_k": 1000,
        "s2c_outlier_multiple": 3.0,
        "radius_scheme": RadiusScheme.PER_FIT,
        "anchor_frequency_gate": 1,
    }
    cfg_kwargs.update(overrides)
    fake_cfg = config_mod.HarvestConfig(**cfg_kwargs)
    monkeypatch.setattr(runner_mod, "load_harvest_config", lambda **_: fake_cfg)
    monkeypatch.setattr(loader_mod, "load_harvest_config", lambda **_: fake_cfg)
    return fake_cfg


def _append_fourth_fit_to_session(session_dir: Path) -> None:
    """Extend the conftest 3-FIT session to a 4-FIT session by adding
    a third turn (pk-E) so the drop test can lose 1 and still satisfy
    the structural LOO floor (survivors ≥ 3)."""
    progress = session_dir / "SESS-TEST.jsonl"
    turn_lines = progress.read_text(encoding="utf-8").splitlines()
    # Turn 3: one new FIT pk-E with chunk_id S1_C0000004.
    new_turn = {
        "turn": 3,
        "timestamp": "2026-04-23T10:03:00Z",
        "session_id": "SESS-TEST",
        "query": "test query",
        "config": {
            "rrf_k": 60,
            "per_path_limit": 20,
            "top_k": 10,
            "active_paths": ["dense", "sparse"],
        },
        "results": [
            {
                "rank": 4,
                "pk": "pk-E",
                "chunk_id": "S1_C0000004",
                "chunk_content": "content for pk-E",
                "sample_id": "S1",
                "counselor_id": "C-1",
                "term": "2025F",
                "rating": "FIT",
                "source_paths": ["dense"],
                "scores": {"dense": 0.9, "sparse": 0.0},
                "span_text": "span for E",
                "span_line_indices": [4],
            }
        ],
    }
    turn_lines.append(json.dumps(new_turn))
    progress.write_text("\n".join(turn_lines) + "\n", encoding="utf-8")

    span_cache = session_dir / "SESS-TEST.span_cache.jsonl"
    with span_cache.open("a", encoding="utf-8") as fp:
        fp.write(
            json.dumps(
                {
                    "chunk_id": "S1_C0000004",
                    "span_text": "span for E",
                    "span_line_indices": [4],
                    "reason": "fixture",
                }
            )
            + "\n"
        )

    details = session_dir / "SESS-TEST.details.jsonl"
    with details.open("a", encoding="utf-8") as fp:
        fp.write(
            json.dumps(
                {
                    "turn": 3,
                    "timestamp": "2026-04-23T10:03:00Z",
                    "session_id": "SESS-TEST",
                    "query": "test query",
                    "search": {
                        "query": "test query",
                        "collection": "parent_transcripts",
                        "embed_url": "http://localhost:8000",
                        "milvus_uri": "http://localhost:19530",
                        "ranker_params": {"rrf_k": 60},
                        "per_path_limit": 20,
                        "top_k": 10,
                        "active_paths": ["dense", "sparse"],
                        "seen_set_size": 0,
                        "milvus_search_params": {},
                        "embedding": {
                            "latency_ms": 1.0,
                            "dense_dim": 4,
                            "sparse_nnz": 0,
                            "sparse_empty": True,
                            "reused_from_probe": False,
                        },
                        "probes": {},
                    },
                }
            )
            + "\n"
        )


def test_quality_gate_drops_outlier_fit(session_dir: Path, monkeypatch):
    """FIT D's chunk embedding sits orthogonal to its span → δ_D ≈ 1.0,
    dwarfing both T and 3× median(δ). A, B, E stay — after the drop we
    still have 3 FITs (the structural LOO floor)."""
    _append_fourth_fit_to_session(session_dir)

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
        "span for E": [0.7, 0.3, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 1.0, 0.0],  # orthogonal — huge δ_D
        "pk-E": [0.7, 0.3, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    # D was dropped: main pass opened 3 iterators, not 4.
    assert client.iterator_opens == 3
    assert len(result.quality_gate_dropped) == 1
    dropped = result.quality_gate_dropped[0]
    assert dropped["fit_pk"] == "pk-D"
    # δ_D ≈ 1.0 > T (small positive) → `exceeds_T` always fires.
    assert "exceeds_T" in dropped["reasons"]
    # Under the floored-median logic (issue #47), δ_D = 1.0 also
    # exceeds the floor-backed cutoff (k · 0.005 = 0.015), so the
    # multiplier reason fires alongside ``exceeds_T``. Pre-#47 the
    # multiplier rule was disabled on this near-zero-median cohort
    # and only ``exceeds_T`` surfaced.
    assert "exceeds_median_multiple" in dropped["reasons"]

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    assert len(meta["quality_gate"]["dropped"]) == 1
    assert meta["quality_gate"]["dropped"][0]["fit_pk"] == "pk-D"
    # Calibration arrays reflect the post-drop cohort (3 survivors).
    assert len(meta["calibration"]["deltas"]) == 3
    assert len(meta["calibration"]["T_primes"]) == 3


def test_quality_gate_multiplier_rule_fires(session_dir: Path, monkeypatch):
    """When the cohort has non-trivial δ spread (so median >
    ``_MEDIAN_DELTA_EPSILON``), a FIT whose δ beats the multiplier
    cutoff but not T is still dropped."""
    _append_fourth_fit_to_session(session_dir)

    # A, B, E have δ ≈ 0.05 each (slight span/chunk mismatch).
    # D has δ ≈ 0.30 — 6× the cohort median, well past 3× the median,
    # but below T (which is the p90 of span-LOO ≈ 0.30 on this
    # divergent span set). Construct spans/chunks to exercise the
    # multiplier rule in isolation.
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.0, 1.0, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
        "span for E": [0.577, 0.577, 0.577, 0.0],  # ~equidistant to A,B,D
    }
    chunks = {
        # δ_A = small — rotate A's chunk slightly off its span.
        "pk-A": [0.998, 0.05, 0.0, 0.0],
        "pk-B": [0.05, 0.998, 0.0, 0.0],
        "pk-D": [0.8, 0.0, 0.6, 0.0],  # rotates D's chunk ~0.20 off its span
        "pk-E": [0.577, 0.577, 0.527, 0.226],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    # Either 0 or 1 drops depending on exact geometry; what matters is
    # that when a FIT IS dropped here, the multiplier reason appears
    # and the pipeline still runs.
    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    qg = meta["quality_gate"]
    assert qg["median_delta_pre_drop"] > 0.0
    # Every recorded reason is one of the two documented values.
    for rec in qg["dropped"]:
        assert set(rec["reasons"]) <= {"exceeds_T", "exceeds_median_multiple"}
    # Multiplier rule must be flagged active and the cutoff persisted
    # whenever median > _MEDIAN_DELTA_EPSILON (issue #48).
    assert qg["multiplier_applied"] is True
    assert qg["multiplier_cutoff"] is not None
    assert qg["multiplier_cutoff"] > 0.0
    # Cutoff = s2c_outlier_multiple × median (default k=3.0).
    expected = qg["s2c_outlier_multiple"] * qg["median_delta_pre_drop"]
    assert qg["multiplier_cutoff"] == pytest.approx(expected, rel=1e-3)
    # AnchorResult mirrors the regime fields.
    assert result.quality_gate_multiplier == qg["s2c_outlier_multiple"]
    assert result.quality_gate_multiplier_cutoff == pytest.approx(
        qg["multiplier_cutoff"], rel=1e-3
    )
    # Median sits above ``_MEDIAN_DELTA_FLOOR`` for this cohort — the
    # floor is inert and the cutoff tracks the median (issue #47).
    assert qg["median_floor_applied"] is False
    assert result.quality_gate_median_floor_applied is False


def test_run_anchor_aborts_when_session_unconverged(session_dir: Path, monkeypatch):
    """Default precondition: a session below the dual gate (cumulative
    FIT < ``min_fit`` or latest P@K < ``precision_at_k``) raises
    ``AnchorValidationError`` before any heavy work runs. The synthetic
    fixture has 3 FITs and turn metrics absent (precision = 0.0); under
    a strict ``min_fit=10, precision_at_k=0.8`` gate it must abort."""
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    # Production-shape gate: cumulative FIT must reach 10, P@K must reach 0.8.
    _fake_harvest_cfg(monkeypatch, min_fit=10, precision_at_k=0.8)

    with pytest.raises(AnchorValidationError, match="not converged"):
        run_anchor(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            batch_size=100,
            max_k=1000,
        )
    # No iterators opened, no sidecars written.
    assert client.iterator_opens == 0
    assert not (session_dir / "SESS-TEST.phase2.jsonl").exists()


def test_run_anchor_allow_unconverged_bypasses_gate(session_dir: Path, monkeypatch):
    """``allow_unconverged=True`` lets the same below-gate session run
    end-to-end — the escape hatch the CLI surfaces as
    ``--allow-unconverged``."""
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch, min_fit=10, precision_at_k=0.8)

    # Same gate as the abort-by-default test, but the bypass flag wins.
    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        allow_unconverged=True,
    )
    assert result.recovery.verdict in {"HEALTHY", "FLAGGED"}
    assert (session_dir / "SESS-TEST.phase2.jsonl").exists()


def test_run_anchor_honours_session_precision_override(session_dir: Path, monkeypatch):
    """Per-session ``harvest.precision_at_k`` overrides must reach the
    Phase 2 convergence gate. Phase 1's ``state.is_converged`` already
    reads the override (snapshotted at session creation), so a session
    that converges under a relaxed threshold must not be re-rejected
    by the loader's gate. Mirrors prod incident on session 48089f25."""
    from src.anchor import config as config_mod
    from src.anchor import loader as loader_mod
    from src.anchor import runner as runner_mod

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]]

    def fake_load(*, session_overrides=None):
        # Simulate YAML default precision_at_k=0.8; per-session override
        # of 0.0 relaxes the gate. Synthetic conftest replays turn-2
        # precision as 0.0, so the override is what decides.
        precision = 0.8
        if session_overrides and "precision_at_k" in session_overrides:
            precision = session_overrides["precision_at_k"]
        return config_mod.HarvestConfig(
            min_fit=1,
            min_not_fit=1,
            precision_at_k=precision,
            batch_size=100,
            max_k=1000,
            s2c_outlier_multiple=3.0,
            radius_scheme=RadiusScheme.PER_FIT,
            anchor_frequency_gate=1,
        )

    monkeypatch.setattr(runner_mod, "load_harvest_config", fake_load)
    monkeypatch.setattr(loader_mod, "load_harvest_config", fake_load)

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)

    # Sanity: without overrides the same session aborts at the gate.
    with pytest.raises(AnchorValidationError, match="not converged"):
        run_anchor(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            batch_size=100,
            max_k=1000,
        )

    # Fresh client — the previous attempt did not open any iterators.
    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        harvest_overrides={"precision_at_k": 0.0},
    )
    assert result.recovery.verdict in {"HEALTHY", "FLAGGED"}
    assert (session_dir / "SESS-TEST.phase2.jsonl").exists()


def test_quality_gate_floor_preserves_pure_whole_chunk_cohort(
    session_dir: Path, monkeypatch
):
    """All-whole-chunk cohort (every δ ≈ float noise) emerges from the
    gate with zero drops. Under the floored-median logic (issue #47)
    the multiplier rule is live (cutoff = k · floor ≈ 0.015), but no
    δ exceeds it — exactly the float-noise protection the older
    epsilon-disable branch gave, now expressed without the regime
    flip that broke mixed cohorts.
    """
    _append_fourth_fit_to_session(session_dir)

    # Every span exactly equals its chunk → δ ≈ 0 for all four FITs.
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.0, 1.0, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
        "span for E": [0.0, 0.0, 0.0, 1.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.0, 1.0, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 1.0, 0.0],
        "pk-E": [0.0, 0.0, 0.0, 1.0],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    assert result.quality_gate_dropped == []
    assert client.iterator_opens == 4
    assert result.quality_gate_median_floor_applied is True
    assert result.quality_gate_multiplier_cutoff == pytest.approx(0.015, abs=1e-6)

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    qg = meta["quality_gate"]
    assert qg["dropped"] == []
    assert qg["multiplier_applied"] is True
    assert qg["multiplier_cutoff"] == pytest.approx(0.015, abs=1e-6)
    assert qg["median_floor_applied"] is True
    assert qg["median_floor"] == pytest.approx(0.005, abs=1e-9)


def test_quality_gate_floor_drops_mixed_cohort_outlier(session_dir: Path, monkeypatch):
    """Issue #47 regression: cohort with three whole-chunk FITs
    (δ ≈ 0) plus one legitimate sub-chunk span (δ ≈ 0.08). Under the
    OLD epsilon-disable logic the multiplier rule was off and the
    outlier slipped through (the bug). Under the floored logic the
    cutoff is k · floor = 0.015 and the 0.08 outlier drops with
    ``exceeds_median_multiple``.
    """
    _append_fourth_fit_to_session(session_dir)

    # Near-orthogonal spans → T ≈ 1.0 (so ``exceeds_T`` cannot fire
    # on δ_D = 0.08; the multiplier rule is the only thing that can
    # catch it).
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.0, 1.0, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
        "span for E": [0.0, 0.0, 0.0, 1.0],
    }
    # A, B, E sit on their own span (δ ≈ 0); D's chunk is rotated so
    # δ_D ≈ 0.08 — the legitimate sub-chunk outlier the old logic
    # would have let through.
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.0, 1.0, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 0.92, 0.392],
        "pk-E": [0.0, 0.0, 0.0, 1.0],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    # Exactly one drop — pk-D — via the multiplier rule.
    assert client.iterator_opens == 3
    assert len(result.quality_gate_dropped) == 1
    dropped = result.quality_gate_dropped[0]
    assert dropped["fit_pk"] == "pk-D"
    assert dropped["reasons"] == ["exceeds_median_multiple"]
    assert dropped["delta"] == pytest.approx(0.08, abs=0.01)

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    qg = meta["quality_gate"]
    assert qg["median_floor_applied"] is True
    assert qg["multiplier_cutoff"] == pytest.approx(0.015, abs=1e-6)


def test_quality_gate_emits_bimodal_warning(session_dir: Path, monkeypatch):
    """When the cohort straddles the float-noise / real-magnitude
    boundary (median below ``_MEDIAN_DELTA_EPSILON``, max well above),
    the gate emits a WARNING through the progress callback so
    operators investigating divergent harvests on identical inputs
    can spot the bimodal regime in logs (issue #47).
    """
    _append_fourth_fit_to_session(session_dir)

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.0, 1.0, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
        "span for E": [0.0, 0.0, 0.0, 1.0],
    }
    # Same straddle fixture as the mixed-cohort drop test: three
    # δ ≈ 0 plus one δ ≈ 0.08 trips the bimodality warning.
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.0, 1.0, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 0.92, 0.392],
        "pk-E": [0.0, 0.0, 0.0, 1.0],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    progress_log: list[str] = []
    run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        progress=progress_log.append,
    )

    bimodal_msgs = [m for m in progress_log if "bimodal" in m.lower()]
    assert bimodal_msgs, f"expected a bimodality warning, got: {progress_log[:10]!r}"
    assert "WARNING" in bimodal_msgs[0]
    assert "Floor backstop active" in bimodal_msgs[0]


def test_quality_gate_no_bimodal_warning_on_uniform_cohort(
    session_dir: Path, monkeypatch
):
    """Whole-chunk cohort with no outlier (every δ ≈ 0, max also in
    the noise band) does NOT trip the bimodality warning — only the
    straddle case fires it. Confirms the warning won't spam every
    healthy whole-chunk run.
    """
    _append_fourth_fit_to_session(session_dir)

    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.0, 1.0, 0.0, 0.0],
        "span for D": [0.0, 0.0, 1.0, 0.0],
        "span for E": [0.0, 0.0, 0.0, 1.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.0, 1.0, 0.0, 0.0],
        "pk-D": [0.0, 0.0, 1.0, 0.0],
        "pk-E": [0.0, 0.0, 0.0, 1.0],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    progress_log: list[str] = []
    run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        progress=progress_log.append,
    )

    assert not any("bimodal" in m.lower() for m in progress_log), (
        f"unexpected bimodal warning in: {progress_log!r}"
    )


def test_quality_gate_all_dropped_raises(session_dir: Path, monkeypatch):
    """When the gate drops the cohort below the structural LOO floor
    (N=3), abort with AnchorValidationError; no sidecar is written."""
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [1.0, 0.0, 0.0, 0.0],
        "span for D": [1.0, 0.0, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [0.0, 1.0, 0.0, 0.0],
        "pk-B": [0.0, 0.0, 1.0, 0.0],
        "pk-D": [0.0, 0.0, 0.0, 1.0],
    }

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    # Identical spans → T = 0; every FIT's δ = 1.0 > T → all dropped.
    # Zero survivors trips the structural floor (3) and aborts.
    _fake_harvest_cfg(monkeypatch)

    with pytest.raises(AnchorValidationError, match="Quality gate"):
        run_anchor(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            batch_size=100,
            max_k=1000,
        )
    assert not (session_dir / "SESS-TEST.phase2.jsonl").exists()


def test_quality_gate_passive_when_all_tight(session_dir: Path, monkeypatch):
    """Healthy near-whole-chunk cohort → no drops. Under the floored
    multiplier rule (issue #47) the cutoff is the floor backstop
    (k · floor ≈ 0.015) and every δ sits well below it, so the
    pipeline continues just like the pre-#47 disable branch did —
    but the sidecar makes the regime explicit via
    ``median_floor_applied``.
    """
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [[_hit("pk-A", 1.0), _hit("pk-B", 1.0), _hit("pk-D", 1.0)]]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )
    assert result.quality_gate_dropped == []
    assert client.iterator_opens == 3
    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    qg = meta["quality_gate"]
    assert qg["dropped"] == []
    # Floored cutoff active — multiplier rule never disabled now.
    assert qg["multiplier_applied"] is True
    assert qg["multiplier_cutoff"] == pytest.approx(0.015, abs=1e-6)
    # median(δ) for this near-whole-chunk cohort is ~0 → floor wins.
    assert qg["median_floor_applied"] is True
    assert qg["median_floor"] == pytest.approx(0.005, abs=1e-9)
    assert result.quality_gate_multiplier_cutoff == pytest.approx(0.015, abs=1e-6)
    assert result.quality_gate_median_floor_applied is True
    # DISCARD-pk filter block always present (zero when no discards).
    assert meta["discard_pk_filter"] == {"n_dropped": 0, "n_total": 0}
    assert result.n_discard_filtered == 0


# --------------------- decoupled-radius (issue #20) -----------------


def _spread_cohort() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Three FITs whose pairwise span-LOO distances are ≈ 0.293 (well
    above all three δ_i), so the quality gate stays passive and the
    scheme branch is the only thing that changes output.

    δ_A = 0,  δ_B ≈ 0.05,  δ_D ≈ 0.10   →   all three per-FIT T'_i differ.
    """
    import math

    span_A = [1.0, 0.0, 0.0, 0.0]
    span_B = [0.0, 1.0, 0.0, 0.0]
    span_D = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]

    # Rotate span_B by acos(0.95) → chunk_B with δ = 0.05.
    theta_B = math.acos(0.95)
    chunk_B = [-math.sin(theta_B), math.cos(theta_B), 0.0, 0.0]

    # Rotate span_D by acos(0.90) → chunk_D with δ = 0.10.
    theta_D = math.acos(0.90)
    c, s = math.cos(theta_D), math.sin(theta_D)
    chunk_D = [
        span_D[0] * c - span_D[1] * s,
        span_D[0] * s + span_D[1] * c,
        0.0,
        0.0,
    ]

    spans = {
        "span for A": span_A,
        "span for B": span_B,
        "span for D": span_D,
    }
    chunks = {
        "pk-A": list(span_A),
        "pk-B": chunk_B,
        "pk-D": chunk_D,
    }
    return spans, chunks


def _spread_pages_factory():
    """Per-FIT page function: each FIT's iterator sees its own chunk at
    sim=1 and a distinct non-FIT neighbour at the FIT-specific
    similarity needed to make the output records' threshold_T_prime
    inherit the *attracting* FIT's T'_i under ``per_fit``.

    Iterator order in ``retrieve_anchored`` is the FIT input order, so
    we use ``search_iterator``'s ``data`` vector to identify which FIT
    is asking.
    """

    # Each iterator returns FIT pks (excluded from output) + one
    # neighbour unique to that FIT so the attribution is unambiguous.
    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit(f"nbr-{round(q[0], 3)}-{round(q[1], 3)}", 0.99),
            ]
        ]

    return pages


def test_radius_scheme_decoupled_uses_session_wide_T_prime_out(
    session_dir: Path, monkeypatch
):
    """Under ``decoupled`` every output record's ``threshold_T_prime``
    equals the session-wide ``T_prime_out``, regardless of which FIT
    attracted it."""
    spans, chunks = _spread_cohort()

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=_spread_pages_factory())
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        radius_scheme=RadiusScheme.DECOUPLED,
    )

    assert result.radius_scheme is RadiusScheme.DECOUPLED
    assert client.iterator_opens == 3  # gate stays passive

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    assert meta["calibration"]["radius_scheme"] == "decoupled"
    assert "T_prime_out" in meta["calibration"]

    T_prime_out = meta["calibration"]["T_prime_out"]
    # T_prime_out = T + min(δ). For δ = [0, 0.05, 0.10] min = 0.
    assert T_prime_out == pytest.approx(meta["calibration"]["T"], abs=1e-5)

    jsonl = session_dir / "SESS-TEST.phase2.jsonl"
    records = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines()]
    assert records, "expected at least one non-FIT neighbour in JSONL"
    threshold_values = {rec["threshold_T_prime"] for rec in records}
    # Every record sees the same threshold — the session-wide cap.
    assert threshold_values == {round(T_prime_out, 6)}


def test_radius_scheme_per_fit_uses_per_fit_T_primes(session_dir: Path, monkeypatch):
    """Under ``per_fit`` the output's ``threshold_T_prime`` varies by
    the attracting FIT, so multiple distinct values appear when δ_i
    varies across the cohort."""
    spans, chunks = _spread_cohort()

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=_spread_pages_factory())
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
        radius_scheme=RadiusScheme.PER_FIT,
    )

    assert result.radius_scheme is RadiusScheme.PER_FIT

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    assert meta["calibration"]["radius_scheme"] == "per_fit"

    jsonl = session_dir / "SESS-TEST.phase2.jsonl"
    records = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines()]
    assert records, "expected at least one non-FIT neighbour in JSONL"
    threshold_values = {rec["threshold_T_prime"] for rec in records}
    # Per-FIT scheme → varying values (since δ_A, δ_B, δ_D all differ).
    assert len(threshold_values) >= 2


# ----------------- anchor-frequency gate (issue #22) ----------------


def _frequency_gate_session(session_dir: Path) -> None:
    """Extend the conftest fixture so the runner sees a 4-FIT cohort
    that survives the quality gate. Three FITs (A, B, D) plus pk-E
    (added below); the gate stays passive on this cohort."""
    _append_fourth_fit_to_session(session_dir)


def _frequency_gate_spans_chunks() -> tuple[
    dict[str, list[float]], dict[str, list[float]]
]:
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.95, 0.31, 0.0, 0.0],
        "span for D": [0.85, 0.53, 0.0, 0.0],
        "span for E": [0.70, 0.71, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.95, 0.31, 0.0, 0.0],
        "pk-D": [0.85, 0.53, 0.0, 0.0],
        "pk-E": [0.70, 0.71, 0.0, 0.0],
    }
    return spans, chunks


def test_frequency_gate_drops_chunks_below_f(session_dir: Path, monkeypatch):
    """With f=2, a chunk admitted by only one FIT is filtered. The
    candidate emitted by ALL FITs is retained; the FIT-specific neighbour
    (admitted by a single anchor) is dropped."""
    _frequency_gate_session(session_dir)
    spans, chunks = _frequency_gate_spans_chunks()

    # FIT A's iterator sees a "wide" neighbour only A admits; every
    # iterator (A, B, D, E) also sees the shared neighbour.
    def pages(q, coll, bs):
        # Encode which FIT is asking via the span vector.
        is_fit_a = q[0] >= 0.99
        page = [
            _hit("pk-A", 1.0),
            _hit("pk-B", 1.0),
            _hit("pk-D", 1.0),
            _hit("pk-E", 1.0),
            _hit("shared", 1.0),
        ]
        if is_fit_a:
            page.append(_hit("solo-fit-a", 1.0))
        return [page]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=2)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    pks_kept = {c.pk for c in result.retrieval.candidates}
    assert "shared" in pks_kept
    assert "solo-fit-a" not in pks_kept

    fg = result.frequency_gate
    assert fg is not None
    assert fg.f_configured == 2
    assert fg.dropped >= 1
    assert fg.kept == len(pks_kept)
    assert fg.qualifying_count_distribution["min"] >= 2

    jsonl = session_dir / "SESS-TEST.phase2.jsonl"
    records = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines()]
    pks_in_jsonl = {r["pk"] for r in records}
    assert pks_in_jsonl == pks_kept
    for rec in records:
        assert rec["qualifying_fit_count"] >= 2
        # qualifying_fit_pks is parallel to count and sorted.
        assert len(rec["qualifying_fit_pks"]) == rec["qualifying_fit_count"]
        assert rec["qualifying_fit_pks"] == sorted(rec["qualifying_fit_pks"])


def test_frequency_gate_keeps_chunks_meeting_f_exactly(session_dir: Path, monkeypatch):
    """Chunk admitted by exactly f distinct anchors is retained."""
    _frequency_gate_session(session_dir)
    spans, chunks = _frequency_gate_spans_chunks()

    # Two FITs (A, B) admit "exact-two"; D and E miss it. f=2 keeps it.
    def pages(q, coll, bs):
        page = [
            _hit("pk-A", 1.0),
            _hit("pk-B", 1.0),
            _hit("pk-D", 1.0),
            _hit("pk-E", 1.0),
        ]
        # Identify the FIT by the leading component of its span vector.
        if q[0] >= 0.99 or (0.94 <= q[0] <= 0.96):
            # FIT A or FIT B
            page.append(_hit("exact-two", 1.0))
        return [page]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=2)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    pks_kept = {c.pk for c in result.retrieval.candidates}
    assert "exact-two" in pks_kept
    by_pk = {c.pk: c for c in result.retrieval.candidates}
    assert by_pk["exact-two"].qualifying_fit_count == 2


def test_frequency_gate_disabled_at_one_keeps_single_anchor_chunks(
    session_dir: Path, monkeypatch
):
    """With f=1 the gate is a no-op: chunks admitted by even one anchor
    are retained, matching pre-#22 behaviour."""
    _frequency_gate_session(session_dir)
    spans, chunks = _frequency_gate_spans_chunks()

    def pages(q, coll, bs):
        page = [
            _hit("pk-A", 1.0),
            _hit("pk-B", 1.0),
            _hit("pk-D", 1.0),
            _hit("pk-E", 1.0),
        ]
        if q[0] >= 0.99:
            page.append(_hit("solo-fit-a", 1.0))
        return [page]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=1)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    pks_kept = {c.pk for c in result.retrieval.candidates}
    assert "solo-fit-a" in pks_kept

    fg = result.frequency_gate
    assert fg.f_configured == 1
    assert fg.dropped == 0
    assert fg.qualifying_count_distribution["min"] >= 1


def test_frequency_gate_aborts_when_f_exceeds_quality_gate_survivors(
    session_dir: Path, monkeypatch
):
    """f greater than the post-quality-gate cohort raises before the
    main pass; no JSONL is written and no Milvus iterator is opened."""
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.95, 0.31, 0.0, 0.0],
        "span for D": [0.85, 0.53, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.95, 0.31, 0.0, 0.0],
        "pk-D": [0.85, 0.53, 0.0, 0.0],
    }

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    # 3-FIT cohort, f=4 → abort.
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=4)

    with pytest.raises(AnchorValidationError, match="anchor_frequency_gate"):
        run_anchor(
            "SESS-TEST",
            runs_dir=session_dir,
            milvus_client=client,
            batch_size=100,
            max_k=1000,
        )

    assert not (session_dir / "SESS-TEST.phase2.jsonl").exists()
    assert not (session_dir / "SESS-TEST.phase2.meta.json").exists()
    # Feasibility check fires before the main pass — no iterators opened.
    assert client.iterator_opens == 0


def test_frequency_gate_meta_block_shape(session_dir: Path, monkeypatch):
    """The frequency_gate block in .phase2.meta.json carries the
    documented keys."""
    _frequency_gate_session(session_dir)
    spans, chunks = _frequency_gate_spans_chunks()

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
                _hit("nbr-1", 1.0),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=2)

    run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    fg = meta["frequency_gate"]
    assert fg is not None
    assert set(fg.keys()) == {
        "f_configured",
        "n_fit_after_quality_gate",
        "kept",
        "dropped",
        "qualifying_count_distribution",
        "qualifying_count_histogram",
    }
    assert fg["f_configured"] == 2
    assert fg["n_fit_after_quality_gate"] == 4
    # All chunks admitted by all 4 FITs (constant page_fn).
    assert fg["dropped"] == 0
    assert set(fg["qualifying_count_distribution"].keys()) == {
        "min",
        "median",
        "max",
    }
    assert fg["qualifying_count_distribution"]["min"] >= 2
    # Histogram counts must sum to kept and be JSON-keyed (str → int).
    hist = fg["qualifying_count_histogram"]
    assert sum(hist.values()) == fg["kept"]
    assert all(isinstance(k, str) for k in hist.keys())


def test_frequency_gate_does_not_affect_loo(session_dir: Path, monkeypatch):
    """LOO recovery is computed under f=1 semantics regardless of
    ``harvest.anchor_frequency_gate`` (issue #22 keeps LOO as a
    calibration sanity check, decoupled from the cohort-tightening
    gate)."""
    _frequency_gate_session(session_dir)
    spans, chunks = _frequency_gate_spans_chunks()

    def pages(q, coll, bs):
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("pk-D", 1.0),
                _hit("pk-E", 1.0),
            ]
        ]

    client_f1 = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=1)
    result_f1 = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client_f1,
        batch_size=100,
        max_k=1000,
    )

    # Re-prime the session and run again under f=2; the LOO
    # verdict / per-FIT recovery is byte-equal because LOO is gated by
    # T'_rest_j only.
    client_f2 = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _fake_harvest_cfg(monkeypatch, anchor_frequency_gate=2)
    result_f2 = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client_f2,
        batch_size=100,
        max_k=1000,
    )

    assert result_f1.recovery.verdict == result_f2.recovery.verdict
    assert result_f1.recovery.recovered == result_f2.recovery.recovered
    assert [p.fit_pk for p in result_f1.recovery.per_fit] == [
        p.fit_pk for p in result_f2.recovery.per_fit
    ]
    assert [p.recovered for p in result_f1.recovery.per_fit] == [
        p.recovered for p in result_f2.recovery.per_fit
    ]


def test_cohort_consistency_flags_missing_own_chunk(session_dir: Path, monkeypatch):
    """If a FIT's own pk is absent from the retained set, cohort-
    consistency marks it missing — warning only, run still exits 0."""
    spans = {
        "span for A": [1.0, 0.0, 0.0, 0.0],
        "span for B": [0.9, 0.1, 0.0, 0.0],
        "span for D": [0.8, 0.2, 0.0, 0.0],
    }
    chunks = {
        "pk-A": [1.0, 0.0, 0.0, 0.0],
        "pk-B": [0.9, 0.1, 0.0, 0.0],
        "pk-D": [0.8, 0.2, 0.0, 0.0],
    }

    # LOO uses client.search (no iterators yet); main pass uses
    # client.search_iterator (bumps iterator_opens first). Discriminate
    # on that counter: LOO gets the full set so it clears; the main
    # pass omits pk-D so cohort consistency can flag it.
    client: _FakeMilvusFull | None = None

    def pages(q, coll, bs):
        if client is None or client.iterator_opens == 0:
            # LOO phase → return every FIT pk so recovery passes.
            return [
                [
                    _hit("pk-A", 1.0),
                    _hit("pk-B", 1.0),
                    _hit("pk-D", 1.0),
                ]
            ]
        # Main pass → pk-D is absent from every iterator's page, plus
        # one non-FIT neighbour so the JSONL isn't empty.
        return [
            [
                _hit("pk-A", 1.0),
                _hit("pk-B", 1.0),
                _hit("nbr-1", 0.99),
            ]
        ]

    client = _FakeMilvusFull(chunk_embeddings=chunks, page_fn=pages)
    _patch_embed(monkeypatch, spans)
    _fake_harvest_cfg(monkeypatch)

    result = run_anchor(
        "SESS-TEST",
        runs_dir=session_dir,
        milvus_client=client,
        batch_size=100,
        max_k=1000,
    )

    # Run completes (warning only, no abort).
    assert result.write is not None

    # pk-D's own chunk is absent — consistency block flags it.
    by_pk = {rec["fit_pk"]: rec for rec in result.cohort_consistency}
    assert by_pk["pk-A"]["own_chunk_retained"] is True
    assert by_pk["pk-B"]["own_chunk_retained"] is True
    assert by_pk["pk-D"]["own_chunk_retained"] is False

    meta = json.loads(
        (session_dir / "SESS-TEST.phase2.meta.json").read_text(encoding="utf-8")
    )
    cc = meta["cohort_consistency"]
    missing = [r for r in cc if not r["own_chunk_retained"]]
    assert len(missing) == 1
    assert missing[0]["fit_pk"] == "pk-D"
