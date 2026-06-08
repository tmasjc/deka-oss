"""Phase 4 runner — three operator-step boundaries.

Mirrors :mod:`src.refine.runner`. The web review drives train →
calibrate-and-review → finalize as distinct user-visible stages, so
the runner exposes them as separate top-level functions rather than
one monolithic ``run_apply``.

- :func:`run_apply_train` — train the LR, evaluate on held-out split,
  pick a CV-suggested threshold, write Stage A sidecars. Comparable
  to :func:`src.refine.runner.run_refine_judge`.
- :func:`run_apply_calibrate` — pure function: given a candidate
  threshold, project the cohort split + borderline samples. Called
  repeatedly by the web UI as the operator drags the slider.
- :func:`finalize_apply` — operator agreed; predictor runs over the
  cohort, Stage B sidecars are written.
- :func:`run_apply_reuse` — the ``--classifier <path>`` reuse path:
  verify rubric pin, score the current cohort, write Stage B only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.search.evidence import PrimaryKey

from .config import ApplyConfig, load_apply_config
from .errors import (
    ApplyConfigError,
    ApplyPrecisionError,
    ApplyValidationError,
)
from .evaluator import (
    KFoldEvaluationOutput,
    evaluate_via_repeated_kfold,
    precision_recall_at,
    sample_borderline_indices,
)
from .features import (
    FeatureFrame,
    build_cohort_frame,
    build_training_frame,
    fetch_embeddings,
    fit_scaler,
    stack_features,
)
from .load_session import (
    Phase4SessionInputs,
    RubricPin,
    load_phase4_session_inputs,
    verify_rubric_pin,
)
from .predictor import (
    predict_cohort_from_params,
    project_cohort,
)
from .schema import (
    ApplyLabel,
    ApplyTimings,
    ApplyWriteResult,
    BorderlineSample,
    ClassBalance,
    ClassifierMetadata,
    CohortProjection,
    EvalReport,
    ScalerParams,
)
from .trainer import TrainResult, predict_proba_keep, train_classifier
from .writer import read_classifier, write_apply_stage_a, write_apply_stage_b

log = logging.getLogger(__name__)


# A function (pks, *, collection) -> {pk: embedding}.
EmbeddingsFetcher = Callable[..., dict[PrimaryKey, list[float]]]


@dataclass
class ApplyRunState:
    """Mutable container threaded through the three operator stages.

    Since the methodology migration, ``training_frame`` carries all N
    labelled rows (no held-out eval split), and ``eval_output`` is a
    :class:`KFoldEvaluationOutput` whose pooled arrays back the
    re-threshold-and-recompute path in :func:`finalize_apply`.
    """

    inputs: Phase4SessionInputs
    cfg: ApplyConfig
    scaler: ScalerParams | None = None
    train_result: TrainResult | None = None
    eval_output: KFoldEvaluationOutput | None = None
    eval_report: EvalReport | None = None
    classifier_metadata: ClassifierMetadata | None = None
    training_frame: FeatureFrame | None = None
    cohort_frame: FeatureFrame | None = None
    cohort_p_keep: np.ndarray | None = None
    write_result: ApplyWriteResult | None = None
    final_projection: CohortProjection | None = None
    timings: dict[str, float] = field(default_factory=dict)
    operator_decision: str = ""

    @property
    def session_id(self) -> str:
        return self.inputs.session_id


# ---------------------------------------------------------------------------
# Stage 1: train (+ evaluate + Stage A write)
# ---------------------------------------------------------------------------


def run_apply_train(
    session_id: str,
    *,
    runs_dir: Path,
    cfg: ApplyConfig | None = None,
    embeddings_fetcher: EmbeddingsFetcher | None = None,
    apply_overrides: dict[str, Any] | None = None,
) -> ApplyRunState:
    """Train the classifier on the Phase 3 sample.

    Steps (since the k-fold methodology migration):
    1. Hydrate Phase 4 inputs from disk.
    2. Fetch dense embeddings for every labelled PK.
    3. Build one feature matrix for all N labels; fit scaler on all N
       distances; stack features once.
    4. Evaluate via repeated stratified k-fold over the full N. Pool
       held-out predictions; pick threshold from the pooled curve.
    5. Train one final LR on all N — this is what gets persisted.
    6. Assemble :class:`ClassifierMetadata` + :class:`EvalReport`.
    7. Stage A write.

    Returns the :class:`ApplyRunState` with stage A artefacts populated
    and the cohort still unfetched (deferred to ``finalize_apply``).
    """
    cfg = cfg or load_apply_config(session_overrides=apply_overrides)
    if not cfg.enabled:
        raise ApplyConfigError(
            "apply.enabled=false in config; Phase 4 is disabled. "
            "Flip apply.enabled=true to run."
        )

    load_started = time.perf_counter()
    inputs = load_phase4_session_inputs(session_id, runs_dir=runs_dir)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    label_y_check = [1 if label.verdict == "KEEP" else 0 for label in inputs.labels]
    if min(label_y_check.count(0), label_y_check.count(1)) < cfg.kfold_splits:
        raise ApplyValidationError(
            "Phase 3 sample has too few examples of one class for "
            f"{cfg.kfold_splits}-fold stratified CV (need at least "
            f"{cfg.kfold_splits} of each). "
            f"Counts: KEEP={label_y_check.count(1)} DROP={label_y_check.count(0)}."
        )

    # Fetch embeddings for every labelled PK.
    pks_needed = [label.pk for label in inputs.labels]
    fetch_started = time.perf_counter()
    embeddings = _fetch_embeddings_via(
        embeddings_fetcher,
        pks=pks_needed,
        collection=inputs.search.collection,
        batch_size=cfg.embedding_fetch_batch,
    )
    embed_ms = (time.perf_counter() - fetch_started) * 1000.0
    _validate_embedding_dim(embeddings, expected_dim=cfg.embedding_dim)

    training_frame = build_training_frame(inputs.labels, embeddings=embeddings)
    if training_frame.n_rows == 0:
        raise ApplyValidationError(
            "Empty training frame after joining Phase 3 labels with Milvus "
            "embeddings. Likely cause: stale Milvus PKs."
        )

    scaler = fit_scaler(training_frame.nearest_fit_distance)
    X = np.asarray(stack_features(training_frame, scaler=scaler), dtype=np.float64)
    y = np.asarray(training_frame.labels or [], dtype=np.int64)

    eval_started = time.perf_counter()
    eval_output = evaluate_via_repeated_kfold(
        X,
        y,
        n_splits=cfg.kfold_splits,
        n_repeats=cfg.eval_n_repeats,
        threshold=cfg.confidence_threshold,
        seed=cfg.seed,
        min_precision=cfg.min_precision,
    )
    eval_ms = (time.perf_counter() - eval_started) * 1000.0

    # Final classifier: single fit on all N rows — this is what gets
    # persisted and used to score the cohort. The k-fold pool above is
    # the audit-grade eval; this fit is the audit-grade model.
    train_started = time.perf_counter()
    train_result = train_classifier(X, y, seed=cfg.seed)
    train_ms = (time.perf_counter() - train_started) * 1000.0

    borderline_idxs = sample_borderline_indices(
        eval_output.per_row_p_keep,
        threshold=cfg.confidence_threshold,
        seed=cfg.seed,
    )
    deciles_full = training_frame.deciles or [0] * training_frame.n_rows
    borderline_samples = [
        BorderlineSample(
            pk=training_frame.pks[i],
            p_keep=float(eval_output.per_row_p_keep[i]),
            nearest_fit_distance=training_frame.nearest_fit_distance[i],
            decile=deciles_full[i],
        )
        for i in borderline_idxs
    ]

    n_keep = int((y == 1).sum())
    n_drop = int((y == 0).sum())
    eval_report = EvalReport(
        precision_at_threshold=eval_output.metrics.precision_at_threshold,
        recall_at_threshold=eval_output.metrics.recall_at_threshold,
        pr_curve=eval_output.metrics.pr_curve,
        threshold_default=cfg.confidence_threshold,
        threshold_selected_by_cv=eval_output.metrics.threshold_selected_by_cv,
        cv_precision_mean=eval_output.metrics.cv_precision_mean,
        cv_precision_std=eval_output.metrics.cv_precision_std,
        min_precision=cfg.min_precision,
        eval_n=training_frame.n_rows,
        eval_keep_n=n_keep,
        eval_drop_n=n_drop,
        eval_methodology="repeated_kfold",
        n_splits=cfg.kfold_splits,
        n_repeats=cfg.eval_n_repeats,
        borderline_samples=borderline_samples,
    )

    training_pk_strings = [_json_pk(p) for p in training_frame.pks]
    training_verdicts = list(int(v) for v in y.tolist())
    metadata = ClassifierMetadata(
        session_id=session_id,
        rubric_version=inputs.rubric.rubric_version,
        prompt_sha256=inputs.rubric.prompt_sha256,
        embedding_model_id=inputs.search.embed_model_id,
        embedding_dim=cfg.embedding_dim,
        feature_layout=[
            f"embedding[0..{cfg.embedding_dim})",
            "nearest_fit_distance",
        ],
        scaler=scaler,
        model=train_result.params,
        threshold=cfg.confidence_threshold,
        min_precision=cfg.min_precision,
        training_pks=training_pk_strings,
        training_verdicts=training_verdicts,
        # Under repeated_kfold, every labelled row is held out at least
        # once across folds, so eval_pks == training_pks. The pair is
        # kept for back-compat with the web DTO contract.
        eval_pks=training_pk_strings,
        eval_verdicts=training_verdicts,
        eval_metrics=eval_output.metrics,
        class_balance=ClassBalance(keep=n_keep, drop=n_drop),
        trained_at=_now_iso(),
    )

    state = ApplyRunState(
        inputs=inputs,
        cfg=cfg,
        scaler=scaler,
        train_result=train_result,
        eval_output=eval_output,
        eval_report=eval_report,
        classifier_metadata=metadata,
        training_frame=training_frame,
        timings={
            "load_ms": load_ms,
            "embed_fetch_ms": embed_ms,
            "train_ms": train_ms,
            "evaluate_ms": eval_ms,
        },
    )

    write_apply_stage_a(
        session_id=session_id,
        runs_dir=runs_dir,
        metadata=metadata,
        eval_report=eval_report,
    )

    log.info(
        "Phase 4 train: session=%s eval_n=%d precision_at_default=%.3f passes_bar=%s",
        session_id,
        training_frame.n_rows,
        eval_report.precision_at_threshold,
        eval_report.passes_bar,
    )
    return state


# ---------------------------------------------------------------------------
# Stage 2: calibrate (pure helper)
# ---------------------------------------------------------------------------


def run_apply_calibrate(
    state: ApplyRunState,
    *,
    runs_dir: Path,
    threshold: float,
    embeddings_fetcher: EmbeddingsFetcher | None = None,
) -> tuple[CohortProjection, list[BorderlineSample]]:
    """Project the cohort split at ``threshold`` + sample borderlines.

    The first call fetches cohort embeddings and scores the whole
    cohort; subsequent calls reuse the cached ``cohort_p_keep`` so
    slider dragging is free.
    """
    if state.classifier_metadata is None or state.train_result is None:
        raise ApplyValidationError(
            "run_apply_calibrate: state has no trained classifier; "
            "run_apply_train must run first."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ApplyValidationError(
            f"run_apply_calibrate: threshold must be in [0, 1]; got {threshold}"
        )

    if state.cohort_frame is None or state.cohort_p_keep is None:
        cohort_pks = [row.pk for row in state.inputs.cohort]
        embeddings = _fetch_embeddings_via(
            embeddings_fetcher,
            pks=cohort_pks,
            collection=state.inputs.search.collection,
            batch_size=state.cfg.embedding_fetch_batch,
        )
        _validate_embedding_dim(embeddings, expected_dim=state.cfg.embedding_dim)
        cohort_frame = build_cohort_frame(state.inputs.cohort, embeddings=embeddings)
        if cohort_frame.n_rows == 0:
            raise ApplyValidationError(
                "Cohort frame empty after Milvus join. Check Milvus "
                "collection and embedding coverage."
            )
        assert state.scaler is not None
        X_cohort = stack_features(cohort_frame, scaler=state.scaler)
        p_keep = predict_proba_keep(state.train_result.estimator, X_cohort)
        state.cohort_frame = cohort_frame
        state.cohort_p_keep = p_keep

    deciles = _assign_deciles(
        state.cohort_frame.nearest_fit_distance,
        n_bins=10,
    )
    projection = project_cohort(
        state.cohort_p_keep,
        threshold=threshold,
        deciles=deciles,
        n_bins=10,
    )
    borderline_idxs = sample_borderline_indices(
        state.cohort_p_keep,
        threshold=threshold,
        seed=state.cfg.seed,
    )
    samples = [
        BorderlineSample(
            pk=state.cohort_frame.pks[i],
            p_keep=float(state.cohort_p_keep[i]),
            nearest_fit_distance=state.cohort_frame.nearest_fit_distance[i],
            decile=deciles[i],
        )
        for i in borderline_idxs
    ]
    return projection, samples


# ---------------------------------------------------------------------------
# Stage 3: finalize (apply + Stage B write)
# ---------------------------------------------------------------------------


def finalize_apply(
    state: ApplyRunState,
    *,
    runs_dir: Path,
    threshold: float | None = None,
    allow_low_precision: bool = False,
    embeddings_fetcher: EmbeddingsFetcher | None = None,
) -> ApplyRunState:
    """Apply classifier to the full cohort and persist Stage B sidecars.

    ``threshold`` overrides ``cfg.confidence_threshold``; if omitted,
    the default is kept. ``allow_low_precision`` is required when the
    eval precision was below ``apply.min_precision`` — guards against
    silently shipping a sub-bar classifier.
    """
    if (
        state.classifier_metadata is None
        or state.train_result is None
        or state.eval_report is None
        or state.scaler is None
    ):
        raise ApplyValidationError(
            "finalize_apply: state is missing a stage result. Run "
            "run_apply_train before finalize."
        )

    chosen_threshold = (
        threshold if threshold is not None else state.cfg.confidence_threshold
    )
    if not 0.0 <= chosen_threshold <= 1.0:
        raise ApplyValidationError(
            f"finalize_apply: threshold must be in [0, 1]; got {chosen_threshold}"
        )

    # Recompute precision/recall at the chosen threshold from the
    # pooled k-fold cache so the persisted classifier reflects the
    # operator's actual pick — same arrays the slider previews used.
    if state.eval_output is not None:
        precision_at_chosen, recall_at_chosen = precision_recall_at(
            state.eval_output.pooled_y,
            state.eval_output.pooled_p_keep,
            threshold=chosen_threshold,
        )
    else:
        precision_at_chosen = state.eval_report.precision_at_threshold
        recall_at_chosen = state.eval_report.recall_at_threshold

    passes_bar = precision_at_chosen >= state.cfg.min_precision
    operator_decision = "agree"
    if not passes_bar:
        if not allow_low_precision:
            raise ApplyPrecisionError(
                f"Eval precision {precision_at_chosen:.3f} at threshold "
                f"{chosen_threshold:.3f} is below the configured bar "
                f"{state.cfg.min_precision:.3f}. Raise the threshold, "
                "improve the rubric, or pass --allow-low-precision to "
                "override (operator decision will be recorded as "
                "'override_low_precision')."
            )
        operator_decision = "override_low_precision"

    # Score the full cohort if calibrate didn't already.
    if state.cohort_frame is None or state.cohort_p_keep is None:
        run_apply_calibrate(
            state,
            runs_dir=runs_dir,
            threshold=chosen_threshold,
            embeddings_fetcher=embeddings_fetcher,
        )
    assert state.cohort_frame is not None
    assert state.cohort_p_keep is not None

    apply_started = time.perf_counter()
    labels = _zip_apply_labels(
        state.cohort_frame, state.cohort_p_keep, threshold=chosen_threshold
    )
    apply_ms = (time.perf_counter() - apply_started) * 1000.0

    deciles = _assign_deciles(state.cohort_frame.nearest_fit_distance, n_bins=10)
    projection = project_cohort(
        state.cohort_p_keep,
        threshold=chosen_threshold,
        deciles=deciles,
        n_bins=10,
    )

    write_started = time.perf_counter()
    updated_metadata = state.classifier_metadata.model_copy(
        update={
            "threshold": chosen_threshold,
            "eval_metrics": state.classifier_metadata.eval_metrics.model_copy(
                update={
                    "precision_at_threshold": float(precision_at_chosen),
                    "recall_at_threshold": float(recall_at_chosen),
                }
            ),
        }
    )
    state.classifier_metadata = updated_metadata
    state.eval_report = EvalReport(
        precision_at_threshold=float(precision_at_chosen),
        recall_at_threshold=float(recall_at_chosen),
        pr_curve=state.eval_report.pr_curve,
        threshold_default=state.eval_report.threshold_default,
        threshold_selected_by_cv=state.eval_report.threshold_selected_by_cv,
        cv_precision_mean=state.eval_report.cv_precision_mean,
        cv_precision_std=state.eval_report.cv_precision_std,
        min_precision=state.eval_report.min_precision,
        eval_n=state.eval_report.eval_n,
        eval_keep_n=state.eval_report.eval_keep_n,
        eval_drop_n=state.eval_report.eval_drop_n,
        eval_methodology=state.eval_report.eval_methodology,
        n_splits=state.eval_report.n_splits,
        n_repeats=state.eval_report.n_repeats,
        borderline_samples=state.eval_report.borderline_samples,
    )

    timings = ApplyTimings(
        load_ms=state.timings.get("load_ms", 0.0),
        embed_fetch_ms=state.timings.get("embed_fetch_ms", 0.0),
        train_ms=state.timings.get("train_ms", 0.0),
        evaluate_ms=state.timings.get("evaluate_ms", 0.0),
        apply_ms=apply_ms,
        write_ms=0.0,
        total_ms=0.0,
    )

    write_result = write_apply_stage_b(
        session_id=state.session_id,
        runs_dir=runs_dir,
        metadata=updated_metadata,
        eval_report=state.eval_report,
        labels=labels,
        projection=projection,
        operator_decision=operator_decision,
        timings=timings,
        cohort_total_input=len(state.inputs.cohort),
    )
    write_ms = (time.perf_counter() - write_started) * 1000.0
    state.timings["apply_ms"] = apply_ms
    state.timings["write_ms"] = write_ms
    state.timings["total_ms"] = sum(
        state.timings.get(k, 0.0)
        for k in (
            "load_ms",
            "embed_fetch_ms",
            "train_ms",
            "evaluate_ms",
            "apply_ms",
            "write_ms",
        )
    )
    state.write_result = write_result
    state.final_projection = projection
    state.operator_decision = operator_decision
    return state


# ---------------------------------------------------------------------------
# Reuse path: load persisted classifier, apply to current cohort
# ---------------------------------------------------------------------------


def run_apply_reuse(
    session_id: str,
    *,
    runs_dir: Path,
    classifier_path: Path,
    cfg: ApplyConfig | None = None,
    threshold: float | None = None,
    allow_low_precision: bool = False,
    embeddings_fetcher: EmbeddingsFetcher | None = None,
    apply_overrides: dict[str, Any] | None = None,
) -> ApplyRunState:
    """Apply a persisted classifier to the session's current cohort.

    Guardrail: verifies the classifier's ``rubric_version`` and
    ``prompt_sha256`` match the session's current
    ``phase3.rubric.json``. Mismatch raises
    :class:`src.apply.errors.ApplyGuardrailError`.
    """
    cfg = cfg or load_apply_config(session_overrides=apply_overrides)
    inputs = load_phase4_session_inputs(session_id, runs_dir=runs_dir)
    classifier = read_classifier(classifier_path)
    verify_rubric_pin(
        classifier_pin=RubricPin(
            rubric_version=classifier.rubric_version,
            prompt_sha256=classifier.prompt_sha256,
        ),
        session_pin=inputs.rubric,
    )

    chosen_threshold = threshold if threshold is not None else classifier.threshold
    if not 0.0 <= chosen_threshold <= 1.0:
        raise ApplyValidationError(
            f"run_apply_reuse: threshold must be in [0, 1]; got {chosen_threshold}"
        )

    if (
        classifier.eval_metrics.precision_at_threshold < cfg.min_precision
        and not allow_low_precision
    ):
        raise ApplyPrecisionError(
            f"Persisted classifier eval precision "
            f"{classifier.eval_metrics.precision_at_threshold:.3f} is below "
            f"bar {cfg.min_precision:.3f}. Pass --allow-low-precision to "
            "proceed; operator decision will be recorded as "
            "'override_low_precision'."
        )

    fetch_started = time.perf_counter()
    cohort_pks = [row.pk for row in inputs.cohort]
    embeddings = _fetch_embeddings_via(
        embeddings_fetcher,
        pks=cohort_pks,
        collection=inputs.search.collection,
        batch_size=cfg.embedding_fetch_batch,
    )
    embed_ms = (time.perf_counter() - fetch_started) * 1000.0
    _validate_embedding_dim(embeddings, expected_dim=cfg.embedding_dim)
    cohort_frame = build_cohort_frame(inputs.cohort, embeddings=embeddings)

    apply_started = time.perf_counter()
    labels = predict_cohort_from_params(
        classifier.model,
        cohort_frame,
        scaler=classifier.scaler,
        threshold=chosen_threshold,
    )
    apply_ms = (time.perf_counter() - apply_started) * 1000.0

    p_keep = np.asarray([label.p_keep for label in labels], dtype=np.float64)
    deciles = _assign_deciles(cohort_frame.nearest_fit_distance, n_bins=10)
    projection = project_cohort(
        p_keep, threshold=chosen_threshold, deciles=deciles, n_bins=10
    )

    operator_decision = (
        "override_low_precision"
        if classifier.eval_metrics.precision_at_threshold < cfg.min_precision
        else "agree"
    )

    timings = ApplyTimings(
        load_ms=0.0,
        embed_fetch_ms=embed_ms,
        train_ms=0.0,
        evaluate_ms=0.0,
        apply_ms=apply_ms,
        write_ms=0.0,
        total_ms=embed_ms + apply_ms,
    )
    eval_report = EvalReport(
        precision_at_threshold=classifier.eval_metrics.precision_at_threshold,
        recall_at_threshold=classifier.eval_metrics.recall_at_threshold,
        pr_curve=list(classifier.eval_metrics.pr_curve),
        threshold_default=classifier.threshold,
        threshold_selected_by_cv=classifier.eval_metrics.threshold_selected_by_cv,
        cv_precision_mean=classifier.eval_metrics.cv_precision_mean,
        cv_precision_std=classifier.eval_metrics.cv_precision_std,
        min_precision=cfg.min_precision,
        eval_n=len(classifier.eval_pks),
        eval_keep_n=sum(1 for v in classifier.eval_verdicts if v == 1),
        eval_drop_n=sum(1 for v in classifier.eval_verdicts if v == 0),
        eval_methodology=classifier.eval_metrics.eval_methodology,
        n_splits=classifier.eval_metrics.n_splits,
        n_repeats=classifier.eval_metrics.n_repeats,
        borderline_samples=[],
    )
    # Update threshold on the reused classifier for the persisted copy.
    metadata = classifier.model_copy(
        update={"threshold": chosen_threshold, "session_id": session_id}
    )

    write_apply_stage_a(
        session_id=session_id,
        runs_dir=runs_dir,
        metadata=metadata,
        eval_report=eval_report,
    )
    write_result = write_apply_stage_b(
        session_id=session_id,
        runs_dir=runs_dir,
        metadata=metadata,
        eval_report=eval_report,
        labels=labels,
        projection=projection,
        operator_decision=operator_decision,
        timings=timings,
        cohort_total_input=len(inputs.cohort),
    )

    state = ApplyRunState(
        inputs=inputs,
        cfg=cfg,
        scaler=classifier.scaler,
        train_result=None,
        eval_output=None,
        eval_report=eval_report,
        classifier_metadata=metadata,
        training_frame=None,
        cohort_frame=cohort_frame,
        cohort_p_keep=p_keep,
        write_result=write_result,
        final_projection=projection,
        timings={
            "embed_fetch_ms": embed_ms,
            "apply_ms": apply_ms,
            "total_ms": embed_ms + apply_ms,
        },
        operator_decision=operator_decision,
    )
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _json_pk(pk: PrimaryKey) -> str | int:
    return pk if isinstance(pk, int) else str(pk)


def _fetch_embeddings_via(
    fetcher: EmbeddingsFetcher | None,
    *,
    pks: list[PrimaryKey],
    collection: str,
    batch_size: int,
) -> dict[PrimaryKey, list[float]]:
    """Dispatch to the injected fetcher (tests) or the real Milvus path."""
    if fetcher is not None:
        return fetcher(pks, collection=collection)
    return fetch_embeddings(pks, collection=collection, batch_size=batch_size)


def _validate_embedding_dim(
    embeddings: dict[PrimaryKey, list[float]], *, expected_dim: int
) -> None:
    if not embeddings:
        return
    first = next(iter(embeddings.values()))
    if len(first) != expected_dim:
        raise ApplyValidationError(
            "Embedding-dim mismatch: configured apply.embedding_dim is "
            f"{expected_dim} but Milvus returned {len(first)}-D vectors. "
            "Likely cause: Milvus collection rebuilt against a different "
            "embedding model."
        )


def _zip_apply_labels(
    frame: FeatureFrame,
    p_keep: np.ndarray,
    *,
    threshold: float,
) -> list[ApplyLabel]:
    out: list[ApplyLabel] = []
    for pk, prob in zip(frame.pks, p_keep.tolist()):
        verdict = "KEEP" if float(prob) >= threshold else "DROP"
        out.append(ApplyLabel(pk=pk, verdict=verdict, p_keep=float(prob)))
    return out


def _assign_deciles(distances: list[float], *, n_bins: int) -> list[int]:
    """Assign each row a decile index based on the cohort's distance order.

    Same semantics as Phase 3's stratified sampler: sort ascending,
    slice into ``n_bins`` near-equal buckets, last absorbs remainder.
    Returns a list of decile indices in the original row order.
    """
    n = len(distances)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (distances[i], i))
    base = n // n_bins
    out = [0] * n
    for b in range(n_bins):
        start = b * base
        end = (b + 1) * base if b < n_bins - 1 else n
        for pos in range(start, end):
            out[order[pos]] = b
    return out
