"""Tests for src.apply.predictor — cohort scoring, projection, reuse path."""

from __future__ import annotations

import numpy as np

from src.apply.features import build_cohort_frame, fit_scaler
from src.apply.load_session import CohortRow
from src.apply.predictor import (
    predict_cohort,
    predict_cohort_from_params,
    project_cohort,
)
from src.apply.schema import ScalerParams
from src.apply.trainer import train_classifier


def _train_separable_model():
    rng = np.random.default_rng(0)
    n = 200
    X_keep = rng.normal(loc=2.0, scale=0.3, size=(n // 2, 4))
    X_drop = rng.normal(loc=-2.0, scale=0.3, size=(n // 2, 4))
    X = np.vstack([X_keep, X_drop])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return train_classifier(X, y, seed=0)


def test_predict_cohort_returns_one_label_per_row():
    result = _train_separable_model()
    cohort = [
        CohortRow(pk=f"p{i}", nearest_fit_distance=0.1 + i * 0.01) for i in range(50)
    ]
    # 3-D embeddings + 1 distance scalar = match the 4-D model.
    rng = np.random.default_rng(1)
    embeddings = {row.pk: list(rng.normal(0.0, 1.0, size=3)) for row in cohort}
    frame = build_cohort_frame(cohort, embeddings=embeddings)
    scaler = fit_scaler(frame.nearest_fit_distance)
    labels = predict_cohort(result.estimator, frame, scaler=scaler, threshold=0.5)
    assert len(labels) == 50
    assert all(label.verdict in ("KEEP", "DROP") for label in labels)
    assert all(0.0 <= label.p_keep <= 1.0 for label in labels)


def test_predict_cohort_empty_frame():
    result = _train_separable_model()
    frame = build_cohort_frame([], embeddings={})
    scaler = ScalerParams(mean=[0.0], scale=[1.0])
    assert predict_cohort(result.estimator, frame, scaler=scaler, threshold=0.5) == []


def test_predict_cohort_from_params_matches_sklearn_predictor():
    result = _train_separable_model()
    cohort = [
        CohortRow(pk=f"p{i}", nearest_fit_distance=0.1 + i * 0.01) for i in range(20)
    ]
    rng = np.random.default_rng(2)
    embeddings = {row.pk: list(rng.normal(0.0, 1.0, size=3)) for row in cohort}
    frame = build_cohort_frame(cohort, embeddings=embeddings)
    scaler = fit_scaler(frame.nearest_fit_distance)
    sklearn_labels = predict_cohort(
        result.estimator, frame, scaler=scaler, threshold=0.5
    )
    json_labels = predict_cohort_from_params(
        result.params, frame, scaler=scaler, threshold=0.5
    )
    assert [label.verdict for label in sklearn_labels] == [
        label.verdict for label in json_labels
    ]
    for s, j in zip(sklearn_labels, json_labels):
        assert s.p_keep == j.p_keep


def test_project_cohort_decile_keep_rate():
    p_keep = [0.9, 0.8, 0.4, 0.3, 0.95, 0.2]
    deciles = [0, 0, 1, 1, 2, 2]
    proj = project_cohort(p_keep, threshold=0.5, deciles=deciles, n_bins=3)
    assert proj.total == 6
    # Threshold 0.5 → indices 0,1,4 KEEP; indices 2,3,5 DROP.
    assert proj.keep == 3
    assert proj.drop == 3
    # Decile 0: both above → 1.0; decile 1: both below → 0.0;
    # decile 2: one above one below → 0.5.
    assert proj.per_decile_keep_rate == [1.0, 0.0, 0.5]


def test_project_cohort_handles_missing_deciles():
    p_keep = [0.9, 0.1]
    proj = project_cohort(p_keep, threshold=0.5, deciles=None, n_bins=10)
    assert proj.total == 2
    assert proj.per_decile_keep_rate == []
