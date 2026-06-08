"""Eval-pass precision/recall + threshold selection.

Production methodology is :func:`evaluate_via_repeated_kfold`, which
computes headline metrics on the pooled predictions of a repeated
stratified k-fold over all N labelled rows. The persisted classifier
is a separate single fit on the full N (handled by the runner).

The PR-curve helper :func:`compute_pr_curve` is used by both the
evaluator and the web UI preview endpoint so the on-screen curve and
the persisted curve agree by construction.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .schema import EvalMetrics
from .trainer import predict_proba_keep, train_classifier

log = logging.getLogger(__name__)


def compute_pr_curve(
    y_true: np.ndarray | list[int],
    p_keep: np.ndarray | list[float],
    *,
    max_points: int = 100,
) -> list[tuple[float, float, float]]:
    """Compute ``[(threshold, precision, recall), ...]`` over the curve.

    sklearn's ``precision_recall_curve`` returns ``thresholds`` shorter
    than ``precision``/``recall`` by one (the last point is the
    no-positives endpoint). We zip the aligned prefix and downsample
    to ``max_points`` evenly-spaced rows so the JSON sidecar doesn't
    bloat for a 1000-row eval split.
    """
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.asarray(p_keep, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y_arr, p_arr)
    n = len(thresholds)
    if n == 0:
        return []
    if n <= max_points:
        idxs = list(range(n))
    else:
        step = max(1, n // max_points)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
    out: list[tuple[float, float, float]] = []
    for i in idxs:
        out.append(
            (
                float(thresholds[i]),
                float(precision[i]),
                float(recall[i]),
            )
        )
    return out


def precision_recall_at(
    y_true: np.ndarray | list[int],
    p_keep: np.ndarray | list[float],
    *,
    threshold: float,
) -> tuple[float, float]:
    """Headline precision and recall at a specific threshold.

    Precision is ``TP / (TP + FP)`` over the rows predicted KEEP;
    recall is ``TP / (TP + FN)`` over the rows truly KEEP. When no
    rows are predicted KEEP, precision is 0.0 (precision-over-recall
    stance: we'd rather report a defensible 0 than NaN).
    """
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.asarray(p_keep, dtype=np.float64)
    predicted_keep = p_arr >= threshold
    truly_keep = y_arr == 1
    tp = int((predicted_keep & truly_keep).sum())
    fp = int((predicted_keep & ~truly_keep).sum())
    fn = int((~predicted_keep & truly_keep).sum())
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return precision, recall


def sample_borderline_indices(
    p_keep: np.ndarray | list[float],
    *,
    threshold: float,
    band: float = 0.05,
    k: int = 10,
    seed: int = 0,
) -> list[int]:
    """Pick up to ``k`` row indices whose ``p_keep`` is within ``band``
    of ``threshold``.

    Borderline indices are sampled deterministically (seeded RNG) so
    the operator's review carousel is reproducible between page loads.
    """
    p_arr = np.asarray(p_keep, dtype=np.float64)
    lo, hi = threshold - band, threshold + band
    candidates = [i for i, v in enumerate(p_arr.tolist()) if lo <= v <= hi]
    if len(candidates) <= k:
        return candidates
    rng = random.Random(seed)
    candidates.sort(key=lambda i: abs(float(p_arr[i]) - threshold))
    return sorted(rng.sample(candidates[: max(k * 3, len(candidates) // 2 or 1)], k))


# ---------------------------------------------------------------------------
# Repeated stratified k-fold — production methodology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KFoldEvaluationOutput:
    """Result of :func:`evaluate_via_repeated_kfold`.

    ``pooled_y`` / ``pooled_p_keep`` are the concatenated held-out
    predictions across all ``n_splits × n_repeats`` folds — each
    original row appears in the pool exactly ``n_repeats`` times.
    ``per_row_p_keep`` is the same pool averaged per original row
    (length N, aligned to the input row order) — what the runner
    feeds to the borderline sampler and the web UI histogram.
    """

    metrics: EvalMetrics
    pooled_y: np.ndarray
    pooled_p_keep: np.ndarray
    per_row_p_keep: np.ndarray
    threshold_default: float


def evaluate_via_repeated_kfold(
    X: np.ndarray | list[list[float]],
    y: np.ndarray | list[int],
    *,
    n_splits: int,
    n_repeats: int,
    threshold: float,
    seed: int,
    min_precision: float,
) -> KFoldEvaluationOutput:
    """Score all N rows via repeated stratified k-fold; pool predictions.

    For each of the ``n_splits × n_repeats`` folds the routine fits a
    fresh :func:`train_classifier` on the train slice and scores the
    held-out slice. The resulting ``(y_va, p_keep_va)`` pairs are
    concatenated into the returned pool; the rows in the pool's row
    order are NOT aligned to the input — only ``per_row_p_keep`` is.

    Headline metrics are computed on the pooled set:
      - ``precision_at_threshold`` / ``recall_at_threshold`` at the
        configured default threshold,
      - ``pr_curve`` downsampled from the pooled curve,
      - ``threshold_selected_by_cv`` = lowest threshold whose pooled
        precision clears ``min_precision``,
      - ``cv_precision_mean`` / ``cv_precision_std`` = mean and std of
        per-repeat precision at the CV-selected threshold (or
        the configured default when no threshold clears the bar).

    The classifier the runner persists is a separate single fit on
    all N rows — not produced here.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    n = len(y_arr)
    if X_arr.ndim != 2:
        raise ValueError(
            f"evaluate_via_repeated_kfold: X must be 2-D; got shape {X_arr.shape}"
        )
    if X_arr.shape[0] != n:
        raise ValueError(
            "evaluate_via_repeated_kfold: X and y row counts disagree: "
            f"{X_arr.shape[0]} vs {n}"
        )
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2; got {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1; got {n_repeats}")
    unique = sorted(set(int(v) for v in y_arr.tolist()))
    if unique != [0, 1]:
        raise ValueError(
            "evaluate_via_repeated_kfold: y must contain both classes 0 "
            f"(DROP) and 1 (KEEP); got {unique}"
        )
    if n < n_splits * 2:
        raise ValueError(
            f"evaluate_via_repeated_kfold: N={n} too small for {n_splits}-fold "
            "CV (need at least 2 × n_splits rows)."
        )

    splitter: StratifiedKFold | RepeatedStratifiedKFold
    if n_repeats == 1:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )

    pooled_y: list[int] = []
    pooled_p: list[float] = []
    # Per-row tally for averaging. Each row appears n_repeats times.
    per_row_sum = np.zeros(n, dtype=np.float64)
    per_row_count = np.zeros(n, dtype=np.int64)
    # Per-repeat (y, p) for cv_precision_mean/std calculation.
    per_repeat_pairs: list[tuple[list[int], list[float]]] = [
        ([], []) for _ in range(n_repeats)
    ]

    for fold_idx, (tr, va) in enumerate(splitter.split(X_arr, y_arr)):
        if len(set(int(v) for v in y_arr[tr].tolist())) < 2:
            log.warning(
                "evaluate_via_repeated_kfold fold %d: train slice has only one "
                "class; skipping.",
                fold_idx,
            )
            continue
        result = train_classifier(X_arr[tr], y_arr[tr], seed=seed + fold_idx)
        p_va = predict_proba_keep(result.estimator, X_arr[va])
        pooled_y.extend(int(v) for v in y_arr[va].tolist())
        pooled_p.extend(float(v) for v in p_va.tolist())
        for row_idx, prob in zip(va.tolist(), p_va.tolist()):
            per_row_sum[row_idx] += float(prob)
            per_row_count[row_idx] += 1
        repeat_idx = fold_idx // n_splits
        per_repeat_pairs[repeat_idx][0].extend(int(v) for v in y_arr[va].tolist())
        per_repeat_pairs[repeat_idx][1].extend(float(v) for v in p_va.tolist())

    if not pooled_y:
        raise RuntimeError(
            "evaluate_via_repeated_kfold: no folds produced predictions "
            "(every fold's train slice was single-class)."
        )

    # Safety: any row that never landed in a held-out fold (shouldn't
    # happen with stratified k-fold but be defensive) would have
    # per_row_count==0; fall back to the row's overall mean.
    missing = int((per_row_count == 0).sum())
    if missing:
        log.warning(
            "evaluate_via_repeated_kfold: %d row(s) never held out; "
            "filling per_row_p_keep with the pooled mean.",
            missing,
        )
        fill = float(np.mean(pooled_p))
        per_row_sum = np.where(per_row_count == 0, fill, per_row_sum)
        per_row_count = np.where(per_row_count == 0, 1, per_row_count)
    per_row_p_keep = per_row_sum / per_row_count

    pooled_y_arr = np.asarray(pooled_y, dtype=np.int64)
    pooled_p_arr = np.asarray(pooled_p, dtype=np.float64)

    precision_at_default, recall_at_default = precision_recall_at(
        pooled_y_arr, pooled_p_arr, threshold=threshold
    )
    pr_curve = compute_pr_curve(pooled_y_arr, pooled_p_arr)
    cv_threshold = _select_threshold_from_pool(
        pooled_y_arr, pooled_p_arr, min_precision=min_precision
    )

    # Per-repeat precision at the chosen threshold for cv_precision_mean/std.
    summary_threshold = (
        cv_threshold if cv_threshold is not None else threshold
    )
    per_repeat_precisions: list[float] = []
    for repeat_y, repeat_p in per_repeat_pairs:
        if not repeat_y:
            continue
        p, _ = precision_recall_at(
            np.asarray(repeat_y, dtype=np.int64),
            np.asarray(repeat_p, dtype=np.float64),
            threshold=summary_threshold,
        )
        per_repeat_precisions.append(float(p))
    if per_repeat_precisions:
        cv_mean = float(np.mean(per_repeat_precisions))
        cv_std = float(np.std(per_repeat_precisions, ddof=0))
    else:
        cv_mean = None
        cv_std = None

    metrics = EvalMetrics(
        precision_at_threshold=float(precision_at_default),
        recall_at_threshold=float(recall_at_default),
        pr_curve=pr_curve,
        threshold_selected_by_cv=cv_threshold,
        cv_precision_mean=cv_mean,
        cv_precision_std=cv_std,
        eval_methodology="repeated_kfold",
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
    log.info(
        "evaluate_via_repeated_kfold: N=%d n_splits=%d n_repeats=%d "
        "pooled=%d precision_at_default=%.3f cv_threshold=%s",
        n,
        n_splits,
        n_repeats,
        len(pooled_y_arr),
        precision_at_default,
        f"{cv_threshold:.3f}" if cv_threshold is not None else "None",
    )
    return KFoldEvaluationOutput(
        metrics=metrics,
        pooled_y=pooled_y_arr,
        pooled_p_keep=pooled_p_arr,
        per_row_p_keep=per_row_p_keep,
        threshold_default=threshold,
    )


def _select_threshold_from_pool(
    pooled_y: np.ndarray,
    pooled_p: np.ndarray,
    *,
    min_precision: float,
) -> float | None:
    """Lowest threshold whose pooled precision clears ``min_precision``.

    Uses sklearn's ``precision_recall_curve`` to enumerate candidate
    thresholds — same source as :func:`compute_pr_curve` so the
    on-screen and persisted picks agree.
    """
    precision, _recall, thresholds = precision_recall_curve(pooled_y, pooled_p)
    chosen: float | None = None
    for i, t in enumerate(thresholds):
        if precision[i] >= min_precision:
            tv = float(t)
            if chosen is None or tv < chosen:
                chosen = tv
    return chosen
