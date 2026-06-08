"""Logistic-regression trainer for Phase 4.

Wraps :class:`sklearn.linear_model.LogisticRegression` with the
defaults the proposal nailed down:

- ``class_weight="balanced"`` to handle the typical 1:4 KEEP:DROP skew
  Phase 3 produces without the operator needing to tune anything.
- ``solver="liblinear"`` for stability on the ~800-row training fold;
  liblinear is also deterministic given a seed (lbfgs's stochastic
  fallback can drift between sklearn versions).
- ``random_state`` from ``apply.seed`` so re-runs are reproducible.

The model is returned as a sklearn estimator for the evaluator; the
writer extracts ``coef_`` / ``intercept_`` into a JSON-safe
:class:`src.apply.schema.ModelParams` so the persisted classifier
survives sklearn version bumps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .schema import ModelParams

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """The trained estimator plus a JSON-safe view of its parameters."""

    estimator: LogisticRegression
    params: ModelParams
    n_train: int
    n_keep: int
    n_drop: int


def train_classifier(
    X: np.ndarray | list[list[float]],
    y: np.ndarray | list[int],
    *,
    seed: int,
) -> TrainResult:
    """Fit a binary LR with balanced class weights.

    ``X`` is (n, embedding_dim + 1); ``y`` is (n,) with 0=DROP, 1=KEEP.
    Returns both the live sklearn estimator (for the evaluator's
    ``predict_proba`` calls) and a :class:`ModelParams` snapshot for
    persistence.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    if X_arr.ndim != 2:
        raise ValueError(f"train_classifier: X must be 2-D; got shape {X_arr.shape}")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            "train_classifier: X and y row counts disagree: "
            f"{X_arr.shape[0]} vs {y_arr.shape[0]}"
        )
    unique = sorted(set(int(v) for v in y_arr.tolist()))
    if unique != [0, 1]:
        raise ValueError(
            "train_classifier: y must contain both classes 0 (DROP) and "
            f"1 (KEEP); got {unique}"
        )

    estimator = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
        max_iter=1000,
    )
    estimator.fit(X_arr, y_arr)

    coef = estimator.coef_.reshape(-1).astype(float).tolist()
    intercept = float(estimator.intercept_.reshape(-1)[0])
    classes = [int(c) for c in estimator.classes_.tolist()]

    n_keep = int((y_arr == 1).sum())
    n_drop = int((y_arr == 0).sum())
    log.info(
        "train_classifier: fit n=%d (keep=%d, drop=%d) feat_dim=%d",
        X_arr.shape[0],
        n_keep,
        n_drop,
        X_arr.shape[1],
    )
    return TrainResult(
        estimator=estimator,
        params=ModelParams(coef=coef, intercept=intercept, classes=classes),
        n_train=int(X_arr.shape[0]),
        n_keep=n_keep,
        n_drop=n_drop,
    )


def predict_proba_keep(
    estimator: LogisticRegression, X: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Return ``P(KEEP)`` per row — i.e. the column for class 1.

    Wrapper so callers don't have to chase the class-index in
    ``estimator.classes_``; we asserted ``[0, 1]`` ordering in
    :func:`train_classifier`.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    probs = estimator.predict_proba(X_arr)
    keep_idx = int(np.where(estimator.classes_ == 1)[0][0])
    return probs[:, keep_idx]


def score_from_params(
    params: ModelParams, X: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Compute ``P(KEEP)`` from a persisted ModelParams without sklearn.

    Used by the reuse path: a classifier loaded from JSON does not
    need to instantiate a sklearn estimator just to score the cohort.
    Equivalent to ``predict_proba`` with class 1 selected.
    """
    coef = np.asarray(params.coef, dtype=np.float64)
    intercept = float(params.intercept)
    X_arr = np.asarray(X, dtype=np.float64)
    z = X_arr @ coef + intercept
    return 1.0 / (1.0 + np.exp(-z))
