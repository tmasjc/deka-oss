"""End-to-end tests for ``POST /api/session/<sid>/resume``.

These tests exercise the integration between the disk classifier,
the per-target hydrator, the SessionStore.attach helper, and the
ownership / 501-deferred-target surface of the route handler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.auth.config import User, UserRegistry
from src.search.adapt import AdaptedConfig
from src.search.config import SearchConfig
from src.search.evidence import EvidenceRow, EvidenceTable
from src.scopes import Scope, ScopeRegistry
from src.web_api import app as app_module
from src.web_api.app import create_app
from src.web_api.sessions import SessionStore


_ALICE_TOKEN = "alice-token-resume"
_BOB_TOKEN = "bob-token-resume"
_SCOPE = "Test Scope"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _users() -> UserRegistry:
    return UserRegistry(
        users=(
            User(id="alice", token_sha256=_sha(_ALICE_TOKEN), allowed_scopes=None),
            User(id="bob", token_sha256=_sha(_BOB_TOKEN), allowed_scopes=None),
        )
    )


def _scopes() -> ScopeRegistry:
    return ScopeRegistry(
        scopes=(
            Scope(name=_SCOPE, description="d", milvus_collection="c1", postgres_table="c1"),
        )
    )


def _base_config() -> SearchConfig:
    return SearchConfig(
        rrf_k=60,
        per_path_limit=20,
        top_k=3,
        embed_url="http://x",
        milvus_uri="http://x",
        collection="c1",
    )


def _fake_table(query: str, config: SearchConfig, *, prefix: str) -> EvidenceTable:
    rows = [
        EvidenceRow(
            rank=i,
            pk=f"{prefix}-pk-{i}",
            chunk_id=f"sample-{i}_C0000001",
            chunk_content=f"line {i}",
            sample_id=f"sample-{i}",
            counselor_id="c1",
            term="2025q1",
            source_paths=["dense"],
            scores={"dense": 0.5, "sparse": 0.0},
        )
        for i in range(1, 4)
    ]
    return EvidenceTable(
        query=query,
        config=config,
        rows=rows,
        per_path_candidates={"dense": [], "sparse": []},
    )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base_config = _base_config()

    def fake_probe(query, config, **_):  # noqa: ANN001
        class P:
            pass

        p = P()
        p.query = query
        p.config = config
        p.stats_by_path = {
            "dense": {"skipped": False, "hit_count": 3},
            "sparse": {"skipped": False, "hit_count": 0},
        }
        return p

    def fake_adapt(seed, probe):  # noqa: ANN001
        return AdaptedConfig(config=seed, rationale=[], flags=[])

    def fake_run_search(query, config, **_):  # noqa: ANN001
        return _fake_table(query, config, prefix="t1")

    monkeypatch.setattr(app_module, "probe_only", fake_probe)
    monkeypatch.setattr(app_module, "adapt_config", fake_adapt)
    monkeypatch.setattr(app_module, "run_search", fake_run_search)

    store = SessionStore(runs_dir=tmp_path, base_config=base_config)
    return create_app(
        store=store,
        scopes=_scopes(),
        users=_users(),
        session_secret="t",
    )


def _client(app, token: str) -> TestClient:
    c = TestClient(app)
    res = c.post("/api/auth/login", json={"token": token})
    assert res.status_code == 200
    return c


def _create_then_quit(client: TestClient, monkeypatch_thresholds: pytest.MonkeyPatch) -> str:
    """Drive a session to convergence + drop the in-memory context.

    Lower the convergence thresholds so a single FIT-rated turn
    crosses the dual gate; then DELETE drops the in-memory ctx
    while the on-disk sidecars (canonical jsonl + converged marker)
    remain. That's what the resume flow re-hydrates from.

    Thresholds are passed as a session override (same path the
    ``[Edit parameters]`` modal uses) so the locked-at-start
    convergence snapshot honours them — see
    :class:`src.session.state.ConvergenceThresholds`.
    """

    res = client.post(
        "/api/session",
        json={
            "query": "the query",
            "scope": _SCOPE,
            "overrides": {
                "harvest": {
                    "min_fit": 1,
                    "min_not_fit": 1,
                    "precision_at_k": 0.5,
                }
            },
        },
    )
    assert res.status_code == 200, res.text
    sid = res.json()["session_id"]

    # Rate two rows FIT + one NOT_FIT so the turn satisfies the dual
    # gate (≥1 FIT, ≥1 NOT_FIT) with P@K = 2/3 ≈ 0.67 ≥ 0.5.
    client.post(f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"})
    client.post(f"/api/session/{sid}/rate", json={"rank": 2, "rating": "FIT"})
    client.post(
        f"/api/session/{sid}/rate", json={"rank": 3, "rating": "NOT_FIT"}
    )
    # Advance one turn so the canonical jsonl gains its first row +
    # the converged marker. After this call the session is converged
    # and the harvest-confirm flow would normally fire.
    advance = client.post(f"/api/session/{sid}/turn/next")
    assert advance.status_code == 200, advance.text

    # Drop the in-memory context — the resume flow will re-hydrate
    # purely from disk.
    client.delete(f"/api/session/{sid}")
    return sid


# ---------------------------------------------------------------------------


def test_resume_idempotent_for_warm_session(app, monkeypatch) -> None:
    """A POST to /resume on a session that's still in memory just
    returns the current snapshot — no re-hydration."""
    alice = _client(app, _ALICE_TOKEN)
    res = alice.post(
        "/api/session",
        json={
            "query": "hello",
            "scope": _SCOPE,
            "overrides": {
                "harvest": {
                    "min_fit": 1,
                    "min_not_fit": 1,
                    "precision_at_k": 0.5,
                }
            },
        },
    )
    sid = res.json()["session_id"]

    again = alice.post(f"/api/session/{sid}/resume")
    assert again.status_code == 200
    assert again.json()["session_id"] == sid


def test_resume_post_tuning_rehydrates_from_disk(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    # The in-memory context is gone. /resume must rebuild from disk.
    res = alice.post(f"/api/session/{sid}/resume")
    assert res.status_code == 200, res.text
    snap = res.json()
    assert snap["session_id"] == sid
    assert snap["query"] == "the query"
    assert snap["scope"] == _SCOPE
    # The reconstructed state has at least one turn and reads as
    # converged (the convergence predicate re-derives from metrics).
    assert snap["turn_number"] >= 1
    assert snap.get("turn_complete") is True

    # Subsequent /api/session/<sid> calls find the now-warm context.
    follow = alice.get(f"/api/session/{sid}")
    assert follow.status_code == 200


def test_resume_unknown_session_returns_404(app) -> None:
    alice = _client(app, _ALICE_TOKEN)
    res = alice.post("/api/session/000000000000/resume")
    assert res.status_code == 404


def test_resume_other_users_session_returns_403(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    bob = _client(app, _BOB_TOKEN)
    res = bob.post(f"/api/session/{sid}/resume")
    assert res.status_code == 403


def test_resume_unauthenticated_returns_401(app) -> None:
    res = TestClient(app).post("/api/session/whatever/resume")
    assert res.status_code == 401


def _write_phase2_sidecars(
    user_dir: Path,
    sid: str,
    *,
    retained_pks: list[str],
    quality_gate: dict | None = None,
) -> None:
    """Write a minimal-but-valid phase2.meta.json + phase2.jsonl pair.

    Only the fields the resume reconstructor reads are populated; the
    rest are absent and the loader's ``.get(..., default)`` calls
    handle them. Mirrors the writer in ``src/anchor/writer.py``.

    ``quality_gate`` overrides the default gate block — pass a full
    dict to exercise specific gate regimes (e.g. multiplier_applied
    True/False, pre-PR-50 sidecars missing the new fields).
    """
    meta = {
        "session_id": sid,
        "query": "the query",
        "collection": "c1",
        "n_fit": 2,
        "calibration": {
            "T": 0.20,
            "deltas": [0.05, 0.10],
            "T_primes": [0.25, 0.30],
            "T_prime_out": 0.27,
            "radius_scheme": "per_fit",
            "delta_summary": {"min": 0.05, "median": 0.075, "max": 0.10},
            "T_prime_summary": {"min": 0.25, "median": 0.275, "max": 0.30},
            "span_loo_distances": [0.05, 0.10],
        },
        "loo_recovery": {
            "recovered": 2,
            "total": 2,
            "verdict": "HEALTHY",
            "missed_fits": [],
        },
        "not_fit_intrusion": {"passed": 0, "total": 0, "intruder_pks": []},
        "output_count": len(retained_pks),
        "batch_size": 2000,
        "max_k": 200000,
        "per_fit_pages": [],
        "per_fit_budget_exhausted": [],
        "cohort_consistency": [],
        "quality_gate": quality_gate
        or {
            "s2c_outlier_multiple": 5.0,
            "T_pre_drop": 0.20,
            "median_delta_pre_drop": 0.075,
            "dropped": [],
        },
        "frequency_gate": {
            "f_configured": 1,
            "n_fit_after_quality_gate": 2,
            "kept": len(retained_pks),
            "dropped": 0,
            "qualifying_count_distribution": {"min": 1, "median": 1, "max": 1},
        },
        "milvus_index_type": "HNSW",
        "milvus_index_params": {},
        "embed_model_id": "test-embed",
        "timings": {
            "load_ms": 1.0,
            "calibrate_ms": 1.0,
            "loo_ms": 1.0,
            "retrieve_ms": 1.0,
            "total_ms": 4.0,
        },
    }
    (user_dir / f"{sid}.phase2.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (user_dir / f"{sid}.phase2.jsonl").open(
        "w", encoding="utf-8"
    ) as fp:
        for i, pk in enumerate(retained_pks, start=1):
            fp.write(
                json.dumps(
                    {
                        "pk": pk,
                        "nearest_fit_pk": "fit-pk-1",
                        "nearest_fit_label": "fit-1",
                        "nearest_fit_distance": 0.10 + 0.01 * i,
                        "passed_threshold": True,
                        "threshold_T_prime": 0.25,
                        "qualifying_fit_count": 1,
                        "qualifying_fit_pks": ["fit-pk-1"],
                    }
                )
                + "\n"
            )


def test_resume_post_harvest_rehydrates_from_disk(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST_HARVEST hydration: full ``AnchorResult`` rebuild from
    on-disk sidecars; ``/harvest/result`` then renders the same DTO it
    would after a live harvest. ``read_only`` stays False so the
    operator can advance into refine.
    """
    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    user_dir = app.state.store.runs_dir_for("alice")
    _write_phase2_sidecars(
        user_dir, sid, retained_pks=["chunk-1", "chunk-2", "chunk-3"]
    )

    res = alice.post(f"/api/session/{sid}/resume")
    assert res.status_code == 200, res.text
    snap = res.json()
    assert snap["session_id"] == sid
    assert snap["phase"] == "ANCHOR_DONE"
    assert snap["read_only"] is False

    harvest = alice.get(f"/api/session/{sid}/harvest/result")
    assert harvest.status_code == 200, harvest.text
    payload = harvest.json()
    assert payload["verdict"] == "HEALTHY"
    assert payload["retained_chunks"] == 3
    assert payload["loo_recovered"] == 2
    assert payload["radius_scheme"] == "per_fit"


def test_resume_post_harvest_surfaces_active_multiplier_cutoff(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the on-disk sidecar carries the post-PR-50 fields
    (``multiplier_applied`` + ``multiplier_cutoff``), the hydrated
    ``AnchorResult`` must round-trip them to the DTO so the UI's
    HarvestSummary panel labels the gate regime correctly. Regression
    for the replay display bug surfaced on session ``e1d2cd2b``."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    user_dir = app.state.store.runs_dir_for("alice")
    _write_phase2_sidecars(
        user_dir,
        sid,
        retained_pks=["chunk-1", "chunk-2", "chunk-3"],
        quality_gate={
            "s2c_outlier_multiple": 3.0,
            "T_pre_drop": 0.225305,
            "median_delta_pre_drop": 0.005472,
            "multiplier_applied": True,
            "multiplier_cutoff": 0.016416,
            "dropped": [],
        },
    )

    alice.post(f"/api/session/{sid}/resume")
    harvest = alice.get(f"/api/session/{sid}/harvest/result")
    assert harvest.status_code == 200, harvest.text
    payload = harvest.json()
    assert payload["quality_gate_multiplier"] == pytest.approx(3.0)
    assert payload["quality_gate_multiplier_cutoff"] == pytest.approx(0.016416)


def test_resume_post_harvest_backfills_multiplier_cutoff_for_legacy_sidecar(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-PR-50 sidecars carry ``s2c_outlier_multiple`` and
    ``median_delta_pre_drop`` but not ``multiplier_applied`` /
    ``multiplier_cutoff``. The hydrator must reconstruct the regime
    from those legacy fields (``cutoff = k · median`` when median is
    above the 1e-3 floor) so the UI doesn't false-label active gates
    as disabled on replay of older sessions."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    user_dir = app.state.store.runs_dir_for("alice")
    _write_phase2_sidecars(
        user_dir,
        sid,
        retained_pks=["chunk-1", "chunk-2", "chunk-3"],
        quality_gate={
            "s2c_outlier_multiple": 3.0,
            "T_pre_drop": 0.225305,
            "median_delta_pre_drop": 0.005472,
            "dropped": [],
        },
    )

    alice.post(f"/api/session/{sid}/resume")
    harvest = alice.get(f"/api/session/{sid}/harvest/result")
    assert harvest.status_code == 200, harvest.text
    payload = harvest.json()
    assert payload["quality_gate_multiplier"] == pytest.approx(3.0)
    assert payload["quality_gate_multiplier_cutoff"] == pytest.approx(
        0.016416, abs=1e-5
    )


def test_resume_post_harvest_disabled_when_legacy_median_below_floor(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy sidecar with median below the 1e-3 floor: hydrator
    backfill must yield ``multiplier_cutoff = None`` so the UI shows
    the gate as genuinely disabled."""
    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    user_dir = app.state.store.runs_dir_for("alice")
    _write_phase2_sidecars(
        user_dir,
        sid,
        retained_pks=["chunk-1", "chunk-2", "chunk-3"],
        quality_gate={
            "s2c_outlier_multiple": 3.0,
            "T_pre_drop": 0.225,
            "median_delta_pre_drop": 1e-06,
            "dropped": [],
        },
    )

    alice.post(f"/api/session/{sid}/resume")
    harvest = alice.get(f"/api/session/{sid}/harvest/result")
    assert harvest.status_code == 200, harvest.text
    payload = harvest.json()
    assert payload["quality_gate_multiplier"] == pytest.approx(3.0)
    assert payload["quality_gate_multiplier_cutoff"] is None


def _stub_refine_state(sid: str, *, n_verdicts: int = 2):
    """Build a minimal :class:`RefineRunState` for hydrator tests.

    Bypasses the real ``_build_refine_state`` (which requires a refine
    config + Postgres-backed Phase 3 inputs) and produces just enough
    state for ``/refine/verdicts`` and ``/refine/finalize`` guards to
    behave: rubric_metadata + judge_result populated, sample / cfg /
    inputs are the smallest valid stubs.
    """
    from src.refine.config import RefineConfig
    from src.refine.judge import JudgeResult, JudgeVerdictRecord
    from src.refine.load_session import Phase3SessionInputs
    from src.refine.runner import RefineRunState
    from src.refine.sample import StratifiedSample

    sha = "0" * 64
    rubric_metadata_dict = {
        "query": "the query",
        "source_session_id": sid,
        "derive_model_id": "stub-model",
        "meta_prompt_path": "stub.md",
        "meta_prompt_sha256": sha,
        "checks": [{"id": "c1", "description": "stub check", "required": True}],
        "fit_examples": [{"pk": "fit-pk-1", "span_text": "fit span"}],
        "not_fit_examples": [],
        "prompt_path": f"{sid}.phase3.prompt.md",
        "prompt_sha256": sha,
        "version": 1,
    }
    from src.refine.derive import metadata_from_json

    rubric_metadata = metadata_from_json(json.dumps(rubric_metadata_dict))

    verdicts = [
        JudgeVerdictRecord(
            pk=f"chunk-{i}",
            nearest_fit_distance=0.10 + 0.01 * i,
            decile=0,
            chunk_content=f"chunk {i} content",
            verdict="KEEP",
            evidence_line_indices=[1],
            failed_check=None,
            reason="passes",
            latency_ms=10.0,
            attempts=1,
            rubric_version=1,
            prompt_sha256=sha,
        )
        for i in range(1, n_verdicts + 1)
    ]
    judge_result = JudgeResult(
        verdicts=verdicts,
        parse_error_count=0,
        api_error_count=0,
        total_latency_ms=10.0 * n_verdicts,
    )

    inputs = Phase3SessionInputs(
        session_id=sid,
        query="the query",
        fits=[],
        not_fits=[],
        rated_pks=frozenset(),
        reflection_diagnoses=[],
    )
    cfg = RefineConfig(
        enabled=True,
        sample_size=10,
        n_bins=10,
        seed=0,
        meta_prompt_path="meta.md",
        max_fit_examples=6,
        max_not_fit_examples=6,
        derive_model="m",
        derive_base_url="http://x",
        derive_temperature=0.0,
        judge_model="j",
        judge_base_url="http://x",
        judge_concurrency=1,
        judge_qps_limit=1.0,
        judge_tpm_limit=1.0,
        judge_timeout_seconds=10,
        judge_max_retries=0,
        api_key_env="DUMMY_KEY",
        auto_drop_known_intruders=False,
    )
    sample = StratifiedSample(
        selected=[],
        auto_drop=[],
        decile_boundaries=[0.0] * (cfg.n_bins + 1),
        per_decile_count=[0] * cfg.n_bins,
        per_decile_drawn=[0] * cfg.n_bins,
    )
    return RefineRunState(
        inputs=inputs,
        cfg=cfg,
        derive_result=None,
        rubric_text="stub rubric text",
        rubric_metadata=rubric_metadata,
        sample=sample,
        judge_result=judge_result,
        write_result=None,
        timings={},
        operator_decision="",
    )


def test_resume_post_rubric_rehydrates_from_disk(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST_RUBRIC hydration: lands in ``REFINE_REVIEW`` with the
    judge verdicts loaded so ``/refine/verdicts`` answers immediately.
    """
    from src.web_api import resume as resume_module

    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    user_dir = app.state.store.runs_dir_for("alice")
    _write_phase2_sidecars(
        user_dir, sid, retained_pks=["chunk-1", "chunk-2"]
    )
    # Bare phase3 markers — content doesn't matter because we stub
    # ``_build_refine_state`` below; the classifier just needs both
    # files present to route POST_RUBRIC.
    (user_dir / f"{sid}.phase3.rubric.json").write_text(
        "{}", encoding="utf-8"
    )
    (user_dir / f"{sid}.phase3.evidence.jsonl").write_text(
        "", encoding="utf-8"
    )

    monkeypatch.setattr(
        resume_module,
        "_build_refine_state",
        lambda replay, *, user_runs_dir: _stub_refine_state(sid, n_verdicts=2),
    )

    res = alice.post(f"/api/session/{sid}/resume")
    assert res.status_code == 200, res.text
    snap = res.json()
    assert snap["session_id"] == sid
    assert snap["phase"] == "REFINE_REVIEW"
    assert snap["read_only"] is False

    verdicts = alice.get(f"/api/session/{sid}/refine/verdicts")
    assert verdicts.status_code == 200, verdicts.text
    body = verdicts.json()
    assert len(body) == 2
    assert {v["pk"] for v in body} == {"chunk-1", "chunk-2"}


def test_resume_done_view_rehydrates_from_disk(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DONE_VIEW hydration: ``read_only=True`` is forwarded so every
    mutating endpoint 409s; the verdicts list still reads back so the
    finalised summary can render.
    """
    from src.web_api import resume as resume_module

    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)

    user_dir = app.state.store.runs_dir_for("alice")
    _write_phase2_sidecars(
        user_dir, sid, retained_pks=["chunk-1", "chunk-2"]
    )
    (user_dir / f"{sid}.phase3.rubric.json").write_text(
        "{}", encoding="utf-8"
    )
    (user_dir / f"{sid}.phase3.evidence.jsonl").write_text(
        "", encoding="utf-8"
    )
    (user_dir / f"{sid}.phase3.meta.json").write_text(
        json.dumps({"operator_decision": "agree"}),
        encoding="utf-8",
    )
    # Phase 4 finalised → DONE_VIEW with ``read_only=True``. Without
    # labels.jsonl the classifier returns APPLY_PENDING and resume
    # binds read_only=False (operator can still run Phase 4).
    (user_dir / f"{sid}.phase4.labels.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        resume_module,
        "_build_refine_state",
        lambda replay, *, user_runs_dir: _stub_refine_state(sid, n_verdicts=1),
    )

    res = alice.post(f"/api/session/{sid}/resume")
    assert res.status_code == 200, res.text
    snap = res.json()
    assert snap["phase"] == "DONE"
    assert snap["read_only"] is True

    # Mutating endpoint must 409 (read-only enforcement from PR #32).
    rate = alice.post(
        f"/api/session/{sid}/rate", json={"rank": 1, "rating": "FIT"}
    )
    assert rate.status_code == 409, rate.text

    # Read-only endpoint still serves the loaded verdicts.
    verdicts = alice.get(f"/api/session/{sid}/refine/verdicts")
    assert verdicts.status_code == 200
    assert len(verdicts.json()) == 1

    # GET /refine/summary serves the same DTO the live finalize POST
    # would return, so the SessionList "Done" row routes the operator
    # straight to the summary screen instead of falling through to the
    # phase-1 rating UI on a DONE_VIEW resume.
    summary = alice.get(f"/api/session/{sid}/refine/summary")
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert "decile_rows" in body
    assert "rubric_version" in body


def _write_phase4_sidecars(
    user_dir: Path,
    sid: str,
    *,
    threshold: float = 0.53,
    keep: int = 9337,
    drop: int = 14164,
    total: int = 23501,
) -> None:
    """Write a valid set of phase4 sidecars so DONE_VIEW resume can
    rehydrate ``state.apply_state`` from disk.

    Pydantic-constructs the ClassifierMetadata to dodge the schema
    cross-field validators (coef length == embedding_dim+1, etc.) that
    a hand-rolled JSON literal would have to track.
    """
    from src.apply.schema import (
        ClassBalance,
        ClassifierMetadata,
        EvalMetrics,
        ModelParams,
        ScalerParams,
    )

    metadata = ClassifierMetadata(
        session_id=sid,
        rubric_version=1,
        prompt_sha256="0" * 64,
        embedding_model_id="stub-embed",
        embedding_dim=4,
        feature_layout=[
            "nearest_fit_distance",
            "embedding_0",
            "embedding_1",
            "embedding_2",
            "embedding_3",
        ],
        scaler=ScalerParams(mean=[0.5], scale=[0.1]),
        model=ModelParams(
            coef=[0.1, 0.2, 0.3, 0.4, 0.5],
            intercept=-0.05,
            classes=[0, 1],
        ),
        threshold=threshold,
        min_precision=0.75,
        training_pks=["t-1", "t-2", "t-3", "t-4"],
        training_verdicts=[1, 0, 1, 0],
        eval_pks=["e-1", "e-2"],
        eval_verdicts=[1, 0],
        eval_metrics=EvalMetrics(
            precision_at_threshold=0.889,
            recall_at_threshold=0.421,
            pr_curve=[(0.1, 0.9, 0.5)],
        ),
        class_balance=ClassBalance(keep=2, drop=2),
        trained_at="2026-05-14T16:00:00Z",
    )
    (user_dir / f"{sid}.phase4.classifier.json").write_text(
        metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    eval_payload = {
        "session_id": sid,
        "threshold_default": 0.7,
        "threshold_selected_by_cv": 0.527,
        "min_precision": 0.75,
        "precision_at_threshold": 0.889,
        "recall_at_threshold": 0.421,
        "cv_precision_mean": 0.94,
        "cv_precision_std": 0.07,
        "eval_n": 100,
        "eval_keep_n": 38,
        "eval_drop_n": 62,
        "pr_curve": [[0.1, 0.9, 0.5]],
        "borderline_samples": [],
        "passes_bar": True,
        "written_at": "2026-05-14T16:03:00Z",
    }
    (user_dir / f"{sid}.phase4.eval.json").write_text(
        json.dumps(eval_payload, indent=2) + "\n", encoding="utf-8"
    )

    meta_payload = {
        "session_id": sid,
        "ts": "2026-05-14T16:05:00Z",
        "rubric_version": 1,
        "prompt_sha256": "0" * 64,
        "threshold": threshold,
        "min_precision": 0.75,
        "cohort_projection": {
            "threshold": threshold,
            "keep": keep,
            "drop": drop,
            "total": total,
            "per_decile_keep_rate": [0.9] * 10,
        },
        "operator_decision": "agree",
    }
    (user_dir / f"{sid}.phase4.meta.json").write_text(
        json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Labels file: each line a JSON record. Count is what shows up as
    # write_result.n_labels in the rehydrated apply_state view.
    with (user_dir / f"{sid}.phase4.labels.jsonl").open(
        "w", encoding="utf-8"
    ) as fp:
        for i in range(keep + drop):
            verdict = "KEEP" if i < keep else "DROP"
            fp.write(
                json.dumps({"pk": f"c-{i}", "verdict": verdict, "p_keep": 0.5})
                + "\n"
            )


def test_resume_done_view_rehydrates_phase4_apply_state(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DONE_VIEW with complete Phase 4 sidecars on disk: resume should
    rehydrate ``state.apply_state`` so that:

    * GET /apply/summary returns 200 with cohort / threshold data
      instead of 404, so the SessionList "Done" row routes the operator
      to the Phase 4 summary card and not the Phase 3 fallback.
    * The workflow timeline carries the new detail values — CALIBRATE
      shows ``τ=<threshold>`` and APPLY shows ``<keep> retained``.
    """
    from src.web_api import resume as resume_module

    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)
    user_dir = app.state.store.runs_dir_for("alice")

    _write_phase2_sidecars(user_dir, sid, retained_pks=["chunk-1", "chunk-2"])
    (user_dir / f"{sid}.phase3.rubric.json").write_text("{}", encoding="utf-8")
    (user_dir / f"{sid}.phase3.evidence.jsonl").write_text("", encoding="utf-8")
    (user_dir / f"{sid}.phase3.meta.json").write_text(
        json.dumps({"operator_decision": "agree"}), encoding="utf-8"
    )
    _write_phase4_sidecars(user_dir, sid)

    monkeypatch.setattr(
        resume_module,
        "_build_refine_state",
        lambda replay, *, user_runs_dir: _stub_refine_state(sid, n_verdicts=1),
    )

    res = alice.post(f"/api/session/{sid}/resume")
    assert res.status_code == 200, res.text
    snap = res.json()
    assert snap["phase"] == "DONE"
    assert snap["read_only"] is True

    # Apply summary now reachable (was 404 before the resume rehydration).
    summary = alice.get(f"/api/session/{sid}/apply/summary")
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert body["cohort_projection"]["keep"] == 9337
    assert body["cohort_projection"]["total"] == 23501
    assert abs(body["threshold"] - 0.53) < 1e-6
    assert body["operator_decision"] == "agree"
    # Query echoed from session state. Timing fields exercise both
    # paths: ``_write_phase2_sidecars`` omits ``ts`` so session_started_at
    # stays None (the missing-field branch), while session_ended_at is
    # populated from the phase4.meta.ts we wrote above.
    assert body["query"] == "the query"
    assert body["session_started_at"] is None
    assert body["session_ended_at"] == "2026-05-14T16:05:00Z"

    # Workflow timeline carries the calibrate τ + apply retained details.
    workflow = {step["key"]: step for step in snap["workflow"]}
    assert workflow["CALIBRATE"]["status"] == "done"
    assert workflow["CALIBRATE"]["detail"] == "τ=0.53"
    assert workflow["APPLY"]["status"] == "done"
    assert workflow["APPLY"]["detail"] == "9337 retained"
    # Rubric and sample details come from the refine_state stub.
    assert workflow["RUBRIC"]["detail"] == "v1"
    assert workflow["SAMPLE"]["detail"] == "10"


def test_resume_done_view_surfaces_post_calibrate_metrics(
    app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """phase4.eval.json is the Stage A sidecar — its precision/recall are
    frozen at ``threshold_default`` (the config default τ) and never
    rewritten when the operator calibrates τ down. The post-calibrate
    values live in classifier.json.eval_metrics, which finalize_apply
    overwrites at the chosen threshold. Resume must surface the
    post-calibrate values; reading from eval.json caused real sessions
    to display recall 0.31 when the applied recall was 0.77 (smoke
    session 0a944b9e, query 家长投诉电话骚扰).
    """
    from src.web_api import resume as resume_module

    alice = _client(app, _ALICE_TOKEN)
    sid = _create_then_quit(alice, monkeypatch)
    user_dir = app.state.store.runs_dir_for("alice")

    _write_phase2_sidecars(user_dir, sid, retained_pks=["chunk-1"])
    (user_dir / f"{sid}.phase3.rubric.json").write_text("{}", encoding="utf-8")
    (user_dir / f"{sid}.phase3.evidence.jsonl").write_text("", encoding="utf-8")
    (user_dir / f"{sid}.phase3.meta.json").write_text(
        json.dumps({"operator_decision": "agree"}), encoding="utf-8"
    )
    # Helper writes both files with matching precision/recall (0.889 /
    # 0.421). Surgically diverge eval.json to mirror production: at the
    # default τ=0.7 precision is higher and recall is lower than at the
    # operator's chosen τ.
    _write_phase4_sidecars(user_dir, sid, threshold=0.51)
    eval_path = user_dir / f"{sid}.phase4.eval.json"
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    eval_payload["precision_at_threshold"] = 0.857
    eval_payload["recall_at_threshold"] = 0.308
    eval_path.write_text(
        json.dumps(eval_payload, indent=2) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        resume_module,
        "_build_refine_state",
        lambda replay, *, user_runs_dir: _stub_refine_state(sid, n_verdicts=1),
    )

    res = alice.post(f"/api/session/{sid}/resume")
    assert res.status_code == 200, res.text

    summary = alice.get(f"/api/session/{sid}/apply/summary")
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert abs(body["eval"]["precision_at_threshold"] - 0.889) < 1e-6, (
        "precision_at_threshold should be the post-calibrate metadata "
        "value (0.889), not the Stage A eval.json value (0.857)"
    )
    assert abs(body["eval"]["recall_at_threshold"] - 0.421) < 1e-6, (
        "recall_at_threshold should be the post-calibrate metadata "
        "value (0.421), not the Stage A eval.json value (0.308)"
    )
