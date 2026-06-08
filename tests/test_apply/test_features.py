"""Tests for src.apply.features — feature-frame join + scaler math."""

from __future__ import annotations

import pytest

from src.apply.features import (
    apply_scaler,
    build_cohort_frame,
    build_training_frame,
    fit_scaler,
    stack_features,
)
from src.apply.load_session import CohortRow, TrainingLabel


def test_fit_scaler_unit_normalises_constant_input():
    # When all distances equal, variance is 0; the scaler avoids
    # div-by-zero by clamping `scale` to a small epsilon.
    scaler = fit_scaler([0.5, 0.5, 0.5])
    assert scaler.mean == [0.5]
    assert scaler.scale[0] > 0.0


def test_fit_scaler_zero_mean_unit_scale_on_known_values():
    scaler = fit_scaler([0.0, 2.0])
    assert scaler.mean == [1.0]
    assert scaler.scale[0] == pytest.approx(1.0, rel=1e-6)


def test_apply_scaler_roundtrips_to_zero_mean():
    distances = [0.1, 0.2, 0.3, 0.4]
    scaler = fit_scaler(distances)
    scaled = apply_scaler(distances, scaler=scaler)
    mean = sum(scaled) / len(scaled)
    assert mean == pytest.approx(0.0, abs=1e-9)


def test_build_training_frame_skips_missing_embedding():
    labels = [
        TrainingLabel(pk="a", nearest_fit_distance=0.1, decile=0, verdict="KEEP"),
        TrainingLabel(pk="b", nearest_fit_distance=0.2, decile=1, verdict="DROP"),
        TrainingLabel(pk="c", nearest_fit_distance=0.3, decile=2, verdict="KEEP"),
    ]
    embeddings = {"a": [1.0, 0.0], "c": [0.0, 1.0]}  # 'b' missing
    frame = build_training_frame(labels, embeddings=embeddings)
    assert frame.n_rows == 2
    assert frame.pks == ["a", "c"]
    assert frame.labels == [1, 1]
    assert frame.deciles == [0, 2]
    assert frame.embedding_dim == 2


def test_build_cohort_frame_preserves_order_and_drops_missing():
    cohort = [
        CohortRow(pk="x", nearest_fit_distance=0.05),
        CohortRow(pk="y", nearest_fit_distance=0.06),
        CohortRow(pk="z", nearest_fit_distance=0.07),
    ]
    embeddings = {"x": [1.0], "z": [3.0]}
    frame = build_cohort_frame(cohort, embeddings=embeddings)
    assert frame.pks == ["x", "z"]
    assert frame.nearest_fit_distance == [0.05, 0.07]
    assert frame.labels is None


def test_stack_features_appends_scaled_distance_per_row():
    labels = [
        TrainingLabel(pk=1, nearest_fit_distance=0.0, decile=0, verdict="KEEP"),
        TrainingLabel(pk=2, nearest_fit_distance=2.0, decile=9, verdict="DROP"),
    ]
    embeddings = {1: [1.0, 2.0, 3.0], 2: [4.0, 5.0, 6.0]}
    frame = build_training_frame(labels, embeddings=embeddings)
    scaler = fit_scaler(frame.nearest_fit_distance)
    X = stack_features(frame, scaler=scaler)
    assert len(X) == 2
    # Embedding columns preserved, last column is scaled distance.
    assert X[0][:3] == [1.0, 2.0, 3.0]
    assert X[1][:3] == [4.0, 5.0, 6.0]
    # Scaled values sum to zero (mean-centred).
    assert X[0][-1] + X[1][-1] == pytest.approx(0.0, abs=1e-9)
