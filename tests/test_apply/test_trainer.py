"""Tests for src.apply.trainer — LR fit, predict_proba, JSON roundtrip."""

from __future__ import annotations

import numpy as np
import pytest

from src.apply.schema import ModelParams
from src.apply.trainer import (
    predict_proba_keep,
    score_from_params,
    train_classifier,
)


def _synthetic_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_keep = rng.normal(loc=1.0, scale=0.5, size=(n // 2, 4))
    X_drop = rng.normal(loc=-1.0, scale=0.5, size=(n // 2, 4))
    X = np.vstack([X_keep, X_drop])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    perm = rng.permutation(n)
    return X[perm], y[perm]


def test_train_classifier_basic_separation():
    X, y = _synthetic_dataset()
    result = train_classifier(X, y, seed=0)
    assert result.n_train == X.shape[0]
    assert result.n_keep == int((y == 1).sum())
    assert result.n_drop == int((y == 0).sum())
    proba = predict_proba_keep(result.estimator, X)
    # Trivially-separable synthetic data — accuracy should be very high.
    pred = (proba >= 0.5).astype(int)
    accuracy = float((pred == y).mean())
    assert accuracy > 0.9


def test_train_classifier_rejects_single_class():
    X = np.zeros((10, 3))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        train_classifier(X, y, seed=0)


def test_train_classifier_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        train_classifier(np.zeros((5, 3)), np.zeros(7, dtype=int), seed=0)


def test_seed_determinism():
    X, y = _synthetic_dataset()
    a = train_classifier(X, y, seed=42)
    b = train_classifier(X, y, seed=42)
    assert a.params.coef == b.params.coef
    assert a.params.intercept == b.params.intercept


def test_score_from_params_matches_estimator_predict_proba():
    X, y = _synthetic_dataset()
    result = train_classifier(X, y, seed=0)
    sklearn_proba = predict_proba_keep(result.estimator, X)
    handrolled = score_from_params(result.params, X)
    assert np.allclose(sklearn_proba, handrolled, atol=1e-8)


def test_model_params_validates_classes():
    with pytest.raises(ValueError):
        ModelParams(coef=[0.1], intercept=0.0, classes=[0, 2])
