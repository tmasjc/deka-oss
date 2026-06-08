"""Tests for src.apply.evaluator — PR curve, precision/recall, k-fold."""

from __future__ import annotations

import numpy as np

from src.apply.evaluator import (
    compute_pr_curve,
    precision_recall_at,
    sample_borderline_indices,
)


def test_precision_recall_at_known_values():
    # 4 truly KEEP, 6 truly DROP. p_keep places KEEP above 0.7 and
    # one DROP at 0.8 (false positive).
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p_keep = [0.9, 0.85, 0.75, 0.71, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]
    precision, recall = precision_recall_at(y, p_keep, threshold=0.7)
    # Predicted KEEP: 5 rows (first 5 sorted by p_keep above 0.7).
    # TP = 4, FP = 1 → precision = 0.8; recall = 4/4 = 1.0.
    assert precision == 0.8
    assert recall == 1.0


def test_precision_recall_at_no_positives_predicted():
    y = [1, 0, 0]
    p_keep = [0.1, 0.05, 0.02]
    precision, recall = precision_recall_at(y, p_keep, threshold=0.5)
    assert precision == 0.0
    assert recall == 0.0


def test_compute_pr_curve_returns_tuples():
    y = [1, 1, 0, 0, 1, 0]
    p_keep = [0.9, 0.7, 0.6, 0.4, 0.3, 0.1]
    curve = compute_pr_curve(y, p_keep)
    assert curve
    for row in curve:
        assert len(row) == 3
        t, p, r = row
        assert 0.0 <= p <= 1.0
        assert 0.0 <= r <= 1.0


def test_sample_borderline_indices_respects_band_and_cap():
    p_keep = [0.1, 0.5, 0.55, 0.6, 0.7, 0.72, 0.9, 0.95]
    idxs = sample_borderline_indices(p_keep, threshold=0.7, band=0.05, k=3)
    assert len(idxs) <= 3
    for i in idxs:
        assert abs(p_keep[i] - 0.7) <= 0.05


# ---------------------------------------------------------------------------
# evaluate_via_repeated_kfold — production methodology
# ---------------------------------------------------------------------------


def _make_separable(n: int, d: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack(
        [
            rng.normal(loc=2.0, scale=0.3, size=(half, d)),
            rng.normal(loc=-2.0, scale=0.3, size=(half, d)),
        ]
    )
    y = np.array([1] * half + [0] * half, dtype=np.int64)
    return X, y


def test_kfold_pool_size_and_per_row_alignment():
    from src.apply.evaluator import evaluate_via_repeated_kfold

    X, y = _make_separable(n=200, d=4)
    out = evaluate_via_repeated_kfold(
        X, y, n_splits=5, n_repeats=5, threshold=0.5, seed=0, min_precision=0.9
    )
    # Pool: N * R = 200 * 5 = 1000.
    assert out.pooled_y.shape == (1000,)
    assert out.pooled_p_keep.shape == (1000,)
    # Per-row averaged: one entry per input row, in input order.
    assert out.per_row_p_keep.shape == (200,)
    # Each labelled row appears in the pool exactly n_repeats times.
    # Without re-running we can at least verify the row order is preserved:
    # the first 100 input rows are the KEEP class, so their averaged
    # p_keep should be substantially higher than the last 100.
    keep_mean = float(out.per_row_p_keep[:100].mean())
    drop_mean = float(out.per_row_p_keep[100:].mean())
    assert keep_mean > 0.8
    assert drop_mean < 0.2


def test_kfold_metadata_records_methodology_fields():
    from src.apply.evaluator import evaluate_via_repeated_kfold

    X, y = _make_separable(n=100, d=4)
    out = evaluate_via_repeated_kfold(
        X, y, n_splits=5, n_repeats=3, threshold=0.5, seed=0, min_precision=0.9
    )
    m = out.metrics
    assert m.eval_methodology == "repeated_kfold"
    assert m.n_splits == 5
    assert m.n_repeats == 3
    assert m.precision_at_threshold >= 0.9
    assert m.recall_at_threshold > 0.0
    assert m.threshold_selected_by_cv is not None
    assert m.cv_precision_mean is not None
    assert m.cv_precision_std is not None


def test_kfold_rejects_tiny_input():
    import pytest

    from src.apply.evaluator import evaluate_via_repeated_kfold

    X = np.zeros((6, 3))
    y = np.array([1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="too small"):
        evaluate_via_repeated_kfold(
            X, y, n_splits=5, n_repeats=2, threshold=0.5, seed=0, min_precision=0.9
        )


def test_kfold_single_repeat_matches_stratified_kfold():
    """``n_repeats=1`` should behave as plain ``StratifiedKFold``: every
    row scored exactly once, pool size equals N.
    """
    from src.apply.evaluator import evaluate_via_repeated_kfold

    X, y = _make_separable(n=100, d=4)
    out = evaluate_via_repeated_kfold(
        X, y, n_splits=5, n_repeats=1, threshold=0.5, seed=0, min_precision=0.9
    )
    assert out.pooled_y.shape == (100,)
    assert out.pooled_p_keep.shape == (100,)
    assert out.metrics.n_repeats == 1
