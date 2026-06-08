"""Cohort prediction — apply trained classifier to every Phase 2 PK.

Two entry points:

- :func:`predict_cohort` — used by the runner's finalise step when the
  sklearn estimator is still in memory.
- :func:`predict_cohort_from_params` — used by the reuse CLI: load
  classifier JSON, score the cohort without instantiating sklearn at
  all.

Both produce a list of :class:`src.apply.schema.ApplyLabel` in the same
order as the input cohort frame so the writer can emit
``phase4.labels.jsonl`` aligned with ``phase2.jsonl``.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import FeatureFrame, stack_features
from .schema import ApplyLabel, CohortProjection, ModelParams, ScalerParams
from .trainer import predict_proba_keep, score_from_params

log = logging.getLogger(__name__)


def predict_cohort(
    estimator: LogisticRegression,
    frame: FeatureFrame,
    *,
    scaler: ScalerParams,
    threshold: float,
) -> list[ApplyLabel]:
    """Score every cohort row and threshold to KEEP/DROP."""
    if frame.n_rows == 0:
        return []
    X = stack_features(frame, scaler=scaler)
    p_keep = predict_proba_keep(estimator, X)
    return _zip_labels(frame, p_keep, threshold)


def predict_cohort_from_params(
    params: ModelParams,
    frame: FeatureFrame,
    *,
    scaler: ScalerParams,
    threshold: float,
) -> list[ApplyLabel]:
    """Reuse-path predictor: scores from persisted JSON, no sklearn."""
    if frame.n_rows == 0:
        return []
    X = stack_features(frame, scaler=scaler)
    p_keep = score_from_params(params, X)
    return _zip_labels(frame, p_keep, threshold)


def project_cohort(
    p_keep: np.ndarray | list[float],
    *,
    threshold: float,
    deciles: list[int] | None,
    n_bins: int = 10,
) -> CohortProjection:
    """Compute the KEEP/DROP split at ``threshold`` over a precomputed
    ``p_keep`` vector.

    Used by the web UI's threshold-preview endpoint: scoring runs once
    when the operator opens the review screen, then projection is just
    a comparison so the slider feels live.
    """
    p_arr = np.asarray(p_keep, dtype=np.float64)
    keep_mask = p_arr >= threshold
    total = int(p_arr.shape[0])
    keep = int(keep_mask.sum())
    drop = total - keep

    per_decile_keep_rate: list[float | None]
    if deciles is None or len(deciles) != total:
        per_decile_keep_rate = []
    else:
        d_arr = np.asarray(deciles, dtype=np.int64)
        per_decile_keep_rate = []
        for b in range(n_bins):
            mask = d_arr == b
            n = int(mask.sum())
            if n == 0:
                per_decile_keep_rate.append(None)
                continue
            kept = int((mask & keep_mask).sum())
            per_decile_keep_rate.append(round(kept / n, 6))
    return CohortProjection(
        threshold=threshold,
        keep=keep,
        drop=drop,
        total=total,
        per_decile_keep_rate=per_decile_keep_rate,
    )


def _zip_labels(
    frame: FeatureFrame,
    p_keep: np.ndarray,
    threshold: float,
) -> list[ApplyLabel]:
    out: list[ApplyLabel] = []
    for pk, prob in zip(frame.pks, p_keep.tolist()):
        verdict = "KEEP" if float(prob) >= threshold else "DROP"
        out.append(ApplyLabel(pk=pk, verdict=verdict, p_keep=float(prob)))
    return out
