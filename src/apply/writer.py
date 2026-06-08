"""Phase 4 output writers — three sidecars + details append.

Mirrors :mod:`src.refine.writer`. Two-stage write so on-disk state
reflects "trained, awaiting operator" distinctly from "operator
agreed, run finalised":

- Stage A (after :func:`run_apply_train`) writes
  ``phase4.classifier.json`` and ``phase4.eval.json`` (the latter is
  ephemeral — replaced on retrain with a different threshold).
- Stage B (after :func:`finalize_apply`) writes
  ``phase4.labels.jsonl`` and ``phase4.meta.json``, appends a
  ``turn="phase4"`` block to ``details.jsonl``, and re-writes
  ``phase4.classifier.json`` with the operator-confirmed threshold.

JSON-only persistence — no pickling — so a persisted classifier
survives sklearn version bumps.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.search.evidence import PrimaryKey

from .schema import (
    ApplyLabel,
    ApplyTimings,
    ApplyWriteResult,
    BorderlineSample,
    ClassBalance,
    ClassifierMetadata,
    CohortProjection,
    EvalMetrics,
    EvalReport,
    ModelParams,
    ScalerParams,
)

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _json_safe_pk(pk: PrimaryKey) -> str | int:
    return pk if isinstance(pk, int) else str(pk)


def _round_or_null(value: float | None, *, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return round(value, ndigits)


def _classifier_payload(metadata: ClassifierMetadata) -> dict[str, Any]:
    return json.loads(metadata.model_dump_json())


def write_apply_stage_a(
    *,
    session_id: str,
    runs_dir: Path,
    metadata: ClassifierMetadata,
    eval_report: EvalReport,
) -> tuple[Path, Path]:
    """Stage A — training done, awaiting operator threshold pick.

    Writes ``phase4.classifier.json`` with the config-default threshold
    baked in, plus ``phase4.eval.json`` carrying the PR curve +
    headline metrics + borderline samples. Both files are
    truncate-overwritten on retrain.

    Returns ``(classifier_path, eval_path)``.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = runs_dir / f"{session_id}.phase4.classifier.json"
    eval_path = runs_dir / f"{session_id}.phase4.eval.json"

    classifier_path.write_text(
        json.dumps(_classifier_payload(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eval_payload = {
        "session_id": session_id,
        "threshold_default": _round_or_null(eval_report.threshold_default),
        "threshold_selected_by_cv": _round_or_null(
            eval_report.threshold_selected_by_cv
        ),
        "min_precision": eval_report.min_precision,
        "precision_at_threshold": _round_or_null(eval_report.precision_at_threshold),
        "recall_at_threshold": _round_or_null(eval_report.recall_at_threshold),
        "cv_precision_mean": _round_or_null(eval_report.cv_precision_mean),
        "cv_precision_std": _round_or_null(eval_report.cv_precision_std),
        "eval_n": eval_report.eval_n,
        "eval_keep_n": eval_report.eval_keep_n,
        "eval_drop_n": eval_report.eval_drop_n,
        "eval_methodology": eval_report.eval_methodology,
        "n_splits": eval_report.n_splits,
        "n_repeats": eval_report.n_repeats,
        "pr_curve": [
            [_round_or_null(t), _round_or_null(p), _round_or_null(r)]
            for (t, p, r) in eval_report.pr_curve
        ],
        "borderline_samples": [
            {
                "pk": _json_safe_pk(s.pk),
                "p_keep": _round_or_null(s.p_keep),
                "nearest_fit_distance": _round_or_null(s.nearest_fit_distance),
                "decile": s.decile,
            }
            for s in eval_report.borderline_samples
        ],
        "passes_bar": eval_report.passes_bar,
        "written_at": _now_iso(),
    }
    eval_path.write_text(
        json.dumps(eval_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log.info(
        "Phase 4 stage A wrote classifier+eval; "
        "precision_at_default=%.3f passes_bar=%s",
        eval_report.precision_at_threshold,
        eval_report.passes_bar,
    )
    return classifier_path, eval_path


def write_apply_stage_b(
    *,
    session_id: str,
    runs_dir: Path,
    metadata: ClassifierMetadata,
    eval_report: EvalReport,
    labels: list[ApplyLabel],
    projection: CohortProjection,
    operator_decision: str,
    timings: ApplyTimings,
    cohort_total_input: int,
) -> ApplyWriteResult:
    """Stage B — operator agreed, finalise the run.

    Re-writes ``phase4.classifier.json`` with the operator-confirmed
    threshold and ``min_precision``, writes ``phase4.labels.jsonl`` and
    ``phase4.meta.json``, and appends a ``turn="phase4"`` block to
    ``details.jsonl``.

    ``cohort_total_input`` is the size of ``phase2.jsonl`` before
    any apply-time embedding-fetch drops; the meta sidecar records
    ``cohort_dropped`` so the operator can sanity-check Milvus drift.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = runs_dir / f"{session_id}.phase4.classifier.json"
    eval_path = runs_dir / f"{session_id}.phase4.eval.json"
    labels_path = runs_dir / f"{session_id}.phase4.labels.jsonl"
    meta_path = runs_dir / f"{session_id}.phase4.meta.json"
    details_path = runs_dir / f"{session_id}.details.jsonl"

    # Re-write classifier with the operator-chosen threshold.
    classifier_path.write_text(
        json.dumps(_classifier_payload(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with labels_path.open("w", encoding="utf-8") as fp:
        for label in labels:
            fp.write(
                json.dumps(
                    {
                        "pk": _json_safe_pk(label.pk),
                        "verdict": label.verdict,
                        "p_keep": _round_or_null(label.p_keep),
                    },
                    ensure_ascii=False,
                )
            )
            fp.write("\n")
        fp.flush()

    meta_payload = _build_meta_payload(
        session_id=session_id,
        metadata=metadata,
        eval_report=eval_report,
        projection=projection,
        operator_decision=operator_decision,
        timings=timings,
        cohort_total_input=cohort_total_input,
        labels=labels,
    )
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    details_block = {"turn": "phase4", "phase4": meta_payload}
    with details_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(details_block, ensure_ascii=False) + "\n")

    log.info(
        "Phase 4 stage B wrote labels+meta+details; decision=%s threshold=%.3f "
        "keep=%d drop=%d",
        operator_decision,
        metadata.threshold,
        projection.keep,
        projection.drop,
    )

    return ApplyWriteResult(
        classifier_path=classifier_path,
        eval_path=eval_path,
        labels_path=labels_path,
        meta_path=meta_path,
        details_path=details_path,
        n_labels=len(labels),
    )


def _build_meta_payload(
    *,
    session_id: str,
    metadata: ClassifierMetadata,
    eval_report: EvalReport,
    projection: CohortProjection,
    operator_decision: str,
    timings: ApplyTimings,
    cohort_total_input: int,
    labels: list[ApplyLabel],
) -> dict[str, Any]:
    verdict_counts = {
        "KEEP": sum(1 for label in labels if label.verdict == "KEEP"),
        "DROP": sum(1 for label in labels if label.verdict == "DROP"),
    }
    return {
        "session_id": session_id,
        "ts": _now_iso(),
        "rubric_version": metadata.rubric_version,
        "prompt_sha256": metadata.prompt_sha256,
        "embedding_model_id": metadata.embedding_model_id,
        "embedding_dim": metadata.embedding_dim,
        "threshold": _round_or_null(metadata.threshold),
        "min_precision": _round_or_null(metadata.min_precision),
        "eval_metrics": {
            "precision_at_threshold": _round_or_null(
                metadata.eval_metrics.precision_at_threshold
            ),
            "recall_at_threshold": _round_or_null(
                metadata.eval_metrics.recall_at_threshold
            ),
            "threshold_selected_by_cv": _round_or_null(
                metadata.eval_metrics.threshold_selected_by_cv
            ),
            "cv_precision_mean": _round_or_null(
                metadata.eval_metrics.cv_precision_mean
            ),
            "cv_precision_std": _round_or_null(metadata.eval_metrics.cv_precision_std),
            "eval_n": eval_report.eval_n,
            "eval_keep_n": eval_report.eval_keep_n,
            "eval_drop_n": eval_report.eval_drop_n,
            "eval_methodology": metadata.eval_metrics.eval_methodology,
            "n_splits": metadata.eval_metrics.n_splits,
            "n_repeats": metadata.eval_metrics.n_repeats,
        },
        "cohort_projection": {
            "threshold": _round_or_null(projection.threshold),
            "keep": projection.keep,
            "drop": projection.drop,
            "total": projection.total,
            "per_decile_keep_rate": [
                _round_or_null(v) for v in projection.per_decile_keep_rate
            ],
        },
        "cohort_total_input": cohort_total_input,
        "cohort_dropped_for_missing_embedding": max(
            0, cohort_total_input - projection.total
        ),
        "verdict_counts": verdict_counts,
        "class_balance_training": {
            "keep": metadata.class_balance.keep,
            "drop": metadata.class_balance.drop,
        },
        "training_n": len(metadata.training_pks),
        "eval_n": len(metadata.eval_pks),
        "operator_decision": operator_decision,
        "timings": {
            "load_ms": round(timings.load_ms, 2),
            "embed_fetch_ms": round(timings.embed_fetch_ms, 2),
            "train_ms": round(timings.train_ms, 2),
            "evaluate_ms": round(timings.evaluate_ms, 2),
            "apply_ms": round(timings.apply_ms, 2),
            "write_ms": round(timings.write_ms, 2),
            "total_ms": round(timings.total_ms, 2),
        },
    }


def read_classifier(path: Path) -> ClassifierMetadata:
    """Rehydrate a persisted classifier JSON into the pydantic model.

    Used by the reuse path: ``python -m src.apply <sid>
    --classifier <path>``. Raises a plain :class:`ValueError`-derived
    pydantic ``ValidationError`` on schema drift; the CLI surfaces it.
    """
    raw = path.read_text(encoding="utf-8")
    return ClassifierMetadata.model_validate_json(raw)


__all__ = [
    "BorderlineSample",  # re-export for callers building stage A reports
    "ClassBalance",
    "ClassifierMetadata",
    "EvalMetrics",
    "ModelParams",
    "ScalerParams",
    "read_classifier",
    "write_apply_stage_a",
    "write_apply_stage_b",
]
