"""Tests for src.apply.runner — golden path + reuse-path guardrail.

Mocks the Milvus embeddings_fetcher so the runner can run end-to-end
without standing up a Milvus instance.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from src.apply.config import ApplyConfig
from src.apply.runner import (
    finalize_apply,
    run_apply_calibrate,
    run_apply_reuse,
    run_apply_train,
)


def _seed_session(
    tmp_path: Path,
    *,
    sid: str = "demo",
    n_keep: int = 60,
    n_drop: int = 60,
    cohort_n: int = 50,
    embedding_dim: int = 8,
) -> None:
    """Write a synthetic Phase 1/2/3 session under ``tmp_path``."""
    sha = hashlib.sha256(b"prompt").hexdigest()
    rubric = {"version": 1, "prompt_sha256": sha}
    (tmp_path / f"{sid}.phase3.rubric.json").write_text(
        json.dumps(rubric), encoding="utf-8"
    )
    details = {
        "search": {
            "collection": "fake_collection",
            "embed_url": "http://embed",
            "embed_model_id": "bge-m3-fake",
        }
    }
    (tmp_path / f"{sid}.details.jsonl").write_text(
        json.dumps(details) + "\n", encoding="utf-8"
    )

    rng = random.Random(0)
    np_rng = np.random.default_rng(0)

    evidence_lines: list[str] = []
    for i in range(n_keep):
        evidence_lines.append(
            json.dumps(
                {
                    "pk": f"keep-{i}",
                    "nearest_fit_distance": rng.uniform(0.0, 0.2),
                    "decile": i % 10,
                    "verdict": "KEEP",
                }
            )
        )
    for i in range(n_drop):
        evidence_lines.append(
            json.dumps(
                {
                    "pk": f"drop-{i}",
                    "nearest_fit_distance": rng.uniform(0.3, 0.6),
                    "decile": (i + 5) % 10,
                    "verdict": "DROP",
                }
            )
        )
    (tmp_path / f"{sid}.phase3.evidence.jsonl").write_text(
        "\n".join(evidence_lines) + "\n", encoding="utf-8"
    )

    cohort_lines: list[str] = []
    for i in range(cohort_n):
        # Cohort rows: half KEEP-shaped, half DROP-shaped.
        dist = rng.uniform(0.0, 0.2) if i < cohort_n // 2 else rng.uniform(0.3, 0.6)
        cohort_lines.append(
            json.dumps({"pk": f"cohort-{i}", "nearest_fit_distance": dist})
        )
    (tmp_path / f"{sid}.phase2.jsonl").write_text(
        "\n".join(cohort_lines) + "\n", encoding="utf-8"
    )

    # Persist embeddings by tagging KEEP-shaped rows with positive-mean
    # vectors and DROP-shaped rows with negative-mean vectors so the
    # classifier can actually learn a boundary.
    embeddings: dict[str, list[float]] = {}
    for i in range(n_keep):
        embeddings[f"keep-{i}"] = list(
            np_rng.normal(loc=1.0, scale=0.3, size=embedding_dim)
        )
    for i in range(n_drop):
        embeddings[f"drop-{i}"] = list(
            np_rng.normal(loc=-1.0, scale=0.3, size=embedding_dim)
        )
    for i in range(cohort_n):
        embeddings[f"cohort-{i}"] = list(
            np_rng.normal(
                loc=1.0 if i < cohort_n // 2 else -1.0,
                scale=0.3,
                size=embedding_dim,
            )
        )
    # Save embeddings to disk via a side-channel JSON the test fetcher reads.
    (tmp_path / "fake_embeddings.json").write_text(
        json.dumps(embeddings), encoding="utf-8"
    )


def _make_fetcher(tmp_path: Path):
    """Return a fetcher that reads from the on-disk embedding stash."""
    embeddings = json.loads(
        (tmp_path / "fake_embeddings.json").read_text(encoding="utf-8")
    )

    def fetcher(pks: Iterable[str], *, collection: str) -> dict[str, list[float]]:
        return {pk: embeddings[pk] for pk in pks if pk in embeddings}

    return fetcher


def _test_cfg(embedding_dim: int = 8) -> ApplyConfig:
    return ApplyConfig(
        enabled=True,
        confidence_threshold=0.5,
        min_precision=0.8,
        eval_fraction=0.25,
        eval_n_repeats=5,
        kfold_splits=3,
        seed=0,
        embedding_dim=embedding_dim,
        embedding_fetch_batch=512,
    )


def test_run_apply_train_writes_stage_a(tmp_path: Path):
    _seed_session(tmp_path)
    state = run_apply_train(
        "demo",
        runs_dir=tmp_path,
        cfg=_test_cfg(),
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    assert state.classifier_metadata is not None
    assert state.eval_report is not None
    assert (tmp_path / "demo.phase4.classifier.json").exists()
    assert (tmp_path / "demo.phase4.eval.json").exists()
    # Separable synthetic — eval precision should be high.
    assert state.eval_report.precision_at_threshold >= 0.8
    # Methodology migration: every labelled row is held out at least
    # once across the k-fold loop, so training and eval PKs match.
    md = state.classifier_metadata
    assert md.eval_metrics.eval_methodology == "repeated_kfold"
    assert md.eval_metrics.n_splits == _test_cfg().kfold_splits
    assert md.eval_metrics.n_repeats == _test_cfg().eval_n_repeats
    assert md.training_pks == md.eval_pks
    assert md.training_verdicts == md.eval_verdicts
    assert state.eval_report.eval_n == len(md.training_pks)
    # Pooled arrays reach the runner state for the slider's recompute path.
    assert state.eval_output is not None
    assert (
        state.eval_output.pooled_y.shape[0]
        == state.eval_report.eval_n * _test_cfg().eval_n_repeats
    )
    assert state.eval_output.per_row_p_keep.shape[0] == state.eval_report.eval_n


def test_full_pipeline_train_then_finalize(tmp_path: Path):
    _seed_session(tmp_path)
    state = run_apply_train(
        "demo",
        runs_dir=tmp_path,
        cfg=_test_cfg(),
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    state = finalize_apply(
        state,
        runs_dir=tmp_path,
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    assert state.write_result is not None
    labels_path = tmp_path / "demo.phase4.labels.jsonl"
    meta_path = tmp_path / "demo.phase4.meta.json"
    assert labels_path.exists() and meta_path.exists()
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    # Cohort had 50 rows; embedding map covers all → labels for all.
    assert len(rows) == 50
    meta = json.loads(meta_path.read_text())
    assert meta["operator_decision"] == "agree"
    assert meta["cohort_projection"]["total"] == 50
    assert meta["verdict_counts"]["KEEP"] + meta["verdict_counts"]["DROP"] == 50


def test_calibrate_then_finalize_at_custom_threshold(tmp_path: Path):
    _seed_session(tmp_path)
    state = run_apply_train(
        "demo",
        runs_dir=tmp_path,
        cfg=_test_cfg(),
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    proj, samples = run_apply_calibrate(
        state,
        runs_dir=tmp_path,
        threshold=0.4,
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    assert proj.total > 0
    assert isinstance(samples, list)
    state = finalize_apply(
        state,
        runs_dir=tmp_path,
        threshold=0.4,
    )
    meta = json.loads((tmp_path / "demo.phase4.meta.json").read_text())
    assert abs(meta["threshold"] - 0.4) < 1e-6


def test_reuse_path_rejects_sha_mismatch(tmp_path: Path):
    import pytest

    from src.apply.errors import ApplyGuardrailError

    _seed_session(tmp_path)
    state = run_apply_train(
        "demo",
        runs_dir=tmp_path,
        cfg=_test_cfg(),
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    state = finalize_apply(
        state, runs_dir=tmp_path, embeddings_fetcher=_make_fetcher(tmp_path)
    )

    classifier_path = tmp_path / "demo.phase4.classifier.json"
    # Tamper with the persisted classifier's rubric sha.
    payload = json.loads(classifier_path.read_text())
    payload["prompt_sha256"] = "f" * 64
    classifier_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ApplyGuardrailError):
        run_apply_reuse(
            "demo",
            runs_dir=tmp_path,
            classifier_path=classifier_path,
            cfg=_test_cfg(),
            embeddings_fetcher=_make_fetcher(tmp_path),
        )


def test_reuse_path_happy(tmp_path: Path):
    _seed_session(tmp_path)
    state = run_apply_train(
        "demo",
        runs_dir=tmp_path,
        cfg=_test_cfg(),
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    state = finalize_apply(
        state, runs_dir=tmp_path, embeddings_fetcher=_make_fetcher(tmp_path)
    )
    classifier_path = tmp_path / "demo.phase4.classifier.json"

    # Re-running reuse should produce the same labels file.
    state2 = run_apply_reuse(
        "demo",
        runs_dir=tmp_path,
        classifier_path=classifier_path,
        cfg=_test_cfg(),
        embeddings_fetcher=_make_fetcher(tmp_path),
    )
    assert state2.write_result is not None
    labels_path = tmp_path / "demo.phase4.labels.jsonl"
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    assert len(rows) == 50
