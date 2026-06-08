"""Tests for src.apply.writer — Stage A / Stage B / details-append shapes."""

from __future__ import annotations

import json
from pathlib import Path

from src.apply.schema import (
    ApplyLabel,
    ApplyTimings,
    ClassBalance,
    ClassifierMetadata,
    CohortProjection,
    EvalMetrics,
    EvalReport,
    ModelParams,
    ScalerParams,
)
from src.apply.writer import (
    read_classifier,
    write_apply_stage_a,
    write_apply_stage_b,
)


def _metadata(*, threshold: float = 0.7, embedding_dim: int = 3) -> ClassifierMetadata:
    return ClassifierMetadata(
        session_id="sid",
        rubric_version=1,
        prompt_sha256="a" * 64,
        embedding_model_id="bge-m3-test",
        embedding_dim=embedding_dim,
        feature_layout=[f"embedding[0..{embedding_dim})", "nearest_fit_distance"],
        scaler=ScalerParams(mean=[0.1], scale=[0.05]),
        model=ModelParams(
            coef=[0.1, 0.2, 0.3, -0.4],
            intercept=0.05,
            classes=[0, 1],
        ),
        threshold=threshold,
        min_precision=0.9,
        training_pks=["a", "b"],
        training_verdicts=[1, 0],
        eval_pks=["c"],
        eval_verdicts=[1],
        eval_metrics=EvalMetrics(
            precision_at_threshold=0.95,
            recall_at_threshold=0.8,
            pr_curve=[(0.5, 0.9, 0.95), (0.7, 0.95, 0.8)],
            threshold_selected_by_cv=0.65,
            cv_precision_mean=0.93,
            cv_precision_std=0.02,
        ),
        class_balance=ClassBalance(keep=4, drop=6),
        trained_at="2026-05-14T00:00:00Z",
    )


def _eval_report() -> EvalReport:
    return EvalReport(
        precision_at_threshold=0.95,
        recall_at_threshold=0.8,
        pr_curve=[(0.5, 0.9, 0.95), (0.7, 0.95, 0.8)],
        threshold_default=0.7,
        threshold_selected_by_cv=0.65,
        cv_precision_mean=0.93,
        cv_precision_std=0.02,
        min_precision=0.9,
        eval_n=10,
        eval_keep_n=4,
        eval_drop_n=6,
        borderline_samples=[],
    )


def test_stage_a_writes_classifier_and_eval(tmp_path: Path):
    metadata = _metadata()
    classifier_path, eval_path = write_apply_stage_a(
        session_id="sid",
        runs_dir=tmp_path,
        metadata=metadata,
        eval_report=_eval_report(),
    )
    assert classifier_path.exists()
    assert eval_path.exists()
    eval_data = json.loads(eval_path.read_text())
    assert eval_data["passes_bar"] is True
    assert eval_data["precision_at_threshold"] == 0.95
    # Classifier roundtrip.
    rehydrated = read_classifier(classifier_path)
    assert rehydrated.threshold == 0.7
    assert rehydrated.rubric_version == 1


def test_stage_b_writes_labels_meta_details(tmp_path: Path):
    metadata = _metadata()
    labels = [
        ApplyLabel(pk="x", verdict="KEEP", p_keep=0.9),
        ApplyLabel(pk="y", verdict="DROP", p_keep=0.2),
    ]
    projection = CohortProjection(
        threshold=0.7,
        keep=1,
        drop=1,
        total=2,
        per_decile_keep_rate=[1.0, None, 0.0, None, None, None, None, None, None, None],
    )
    write_apply_stage_b(
        session_id="sid",
        runs_dir=tmp_path,
        metadata=metadata,
        eval_report=_eval_report(),
        labels=labels,
        projection=projection,
        operator_decision="agree",
        timings=ApplyTimings(total_ms=12345.0),
        cohort_total_input=3,  # one row dropped for missing embedding
    )

    labels_path = tmp_path / "sid.phase4.labels.jsonl"
    meta_path = tmp_path / "sid.phase4.meta.json"
    details_path = tmp_path / "sid.details.jsonl"
    assert labels_path.exists() and meta_path.exists() and details_path.exists()

    labels_data = [
        json.loads(line) for line in labels_path.read_text().splitlines() if line
    ]
    assert len(labels_data) == 2
    assert labels_data[0] == {"pk": "x", "verdict": "KEEP", "p_keep": 0.9}

    meta = json.loads(meta_path.read_text())
    assert meta["rubric_version"] == 1
    assert meta["operator_decision"] == "agree"
    assert meta["cohort_projection"]["keep"] == 1
    assert meta["cohort_total_input"] == 3
    assert meta["cohort_dropped_for_missing_embedding"] == 1
    assert meta["verdict_counts"] == {"KEEP": 1, "DROP": 1}

    details_block = json.loads(details_path.read_text().strip().splitlines()[-1])
    assert details_block["turn"] == "phase4"
    assert "phase4" in details_block


def test_stage_b_appends_does_not_overwrite_details(tmp_path: Path):
    details_path = tmp_path / "sid.details.jsonl"
    details_path.write_text(json.dumps({"turn": "phase3"}) + "\n", encoding="utf-8")
    metadata = _metadata()
    write_apply_stage_b(
        session_id="sid",
        runs_dir=tmp_path,
        metadata=metadata,
        eval_report=_eval_report(),
        labels=[],
        projection=CohortProjection(
            threshold=0.7, keep=0, drop=0, total=0, per_decile_keep_rate=[]
        ),
        operator_decision="agree",
        timings=ApplyTimings(),
        cohort_total_input=0,
    )
    lines = [json.loads(line) for line in details_path.read_text().splitlines() if line]
    assert [block["turn"] for block in lines] == ["phase3", "phase4"]


def test_classifier_validates_dimensions():
    """The pydantic model rejects coef-length / embedding-dim mismatches."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClassifierMetadata(
            session_id="sid",
            rubric_version=1,
            prompt_sha256="a" * 64,
            embedding_model_id="bge-m3-test",
            embedding_dim=3,
            feature_layout=["embedding[0..3)", "nearest_fit_distance"],
            scaler=ScalerParams(mean=[0.0], scale=[1.0]),
            # Wrong coef length — should be 4, given embedding_dim=3.
            model=ModelParams(coef=[0.1, 0.2], intercept=0.0, classes=[0, 1]),
            threshold=0.7,
            min_precision=0.9,
            training_pks=[],
            training_verdicts=[],
            eval_pks=[],
            eval_verdicts=[],
            eval_metrics=EvalMetrics(
                precision_at_threshold=0.95, recall_at_threshold=0.8
            ),
            class_balance=ClassBalance(keep=0, drop=0),
            trained_at="2026-05-14T00:00:00Z",
        )
