"""Pydantic models + dataclasses for Phase 4 artefacts.

Two surfaces live here:

- :class:`ClassifierMetadata` — the persisted JSON shape of
  ``runs/{sid}.phase4.classifier.json``. Carries the trained model,
  embedding-model id, threshold, training/eval PKs, eval metrics, and
  the rubric pin (``rubric_version`` + ``prompt_sha256``). Stored as
  plain JSON so a classifier survives sklearn version bumps.
- :class:`EvalReport`, :class:`ApplyLabel`, :class:`CohortProjection`
  — lightweight dataclasses passed across the runner / writer / web
  API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.search.evidence import PrimaryKey


# ---------------------------------------------------------------------------
# Classifier persistence
# ---------------------------------------------------------------------------


class ScalerParams(BaseModel):
    """Standardisation params for the scalar ``nearest_fit_distance``
    feature.

    Dense embedding components are not scaled — BGE-M3 dense vectors are
    already unit-normalised, so a per-component scaler would just learn
    the variance of the training-set component-frequencies and hurt the
    apply-time predictions.
    """

    model_config = ConfigDict(extra="forbid")
    mean: list[float] = Field(..., min_length=1)
    scale: list[float] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_lengths(self) -> "ScalerParams":
        if len(self.mean) != len(self.scale):
            raise ValueError(
                "ScalerParams.mean and .scale must have equal length; got "
                f"{len(self.mean)} vs {len(self.scale)}"
            )
        for s in self.scale:
            if s <= 0.0:
                raise ValueError("ScalerParams.scale entries must be positive")
        return self


class ModelParams(BaseModel):
    """LR coefficients + intercept for one binary classifier."""

    model_config = ConfigDict(extra="forbid")
    coef: list[float] = Field(..., min_length=1)
    intercept: float
    classes: list[int] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def _check_classes(self) -> "ModelParams":
        if sorted(self.classes) != [0, 1]:
            raise ValueError(
                f"ModelParams.classes must be [0, 1] (DROP, KEEP); got {self.classes}"
            )
        return self


class EvalMetrics(BaseModel):
    """Headline + PR-curve metrics from the eval pass.

    ``eval_methodology`` records how the metrics were computed:
    ``"single_split"`` is the legacy one-shot 80/20 split;
    ``"repeated_kfold"`` is the production default — repeated
    stratified k-fold pooling predictions over all N labelled rows.
    ``n_splits`` and ``n_repeats`` populate only for ``repeated_kfold``.
    """

    model_config = ConfigDict(extra="forbid")
    precision_at_threshold: float = Field(..., ge=0.0, le=1.0)
    recall_at_threshold: float = Field(..., ge=0.0, le=1.0)
    pr_curve: list[tuple[float, float, float]] = Field(default_factory=list)
    threshold_selected_by_cv: float | None = None
    cv_precision_mean: float | None = None
    cv_precision_std: float | None = None
    eval_methodology: Literal["single_split", "repeated_kfold"] = "single_split"
    n_splits: int | None = None
    n_repeats: int | None = None


class ClassBalance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep: int = Field(..., ge=0)
    drop: int = Field(..., ge=0)


class ClassifierMetadata(BaseModel):
    """Exact JSON shape persisted as
    ``runs/{sid}.phase4.classifier.json``.

    The rubric pin is the hard guardrail for the reuse path: at apply
    time, the runner verifies the session's current
    ``phase3.rubric.json`` matches this ``rubric_version`` and
    ``prompt_sha256``; mismatch raises
    :class:`src.apply.errors.ApplyGuardrailError`.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., min_length=1)
    rubric_version: int = Field(..., ge=1)
    prompt_sha256: str = Field(..., min_length=64, max_length=64)
    embedding_model_id: str
    embedding_dim: int = Field(..., ge=1)
    feature_layout: list[str] = Field(..., min_length=2)
    scaler: ScalerParams
    model: ModelParams
    threshold: float = Field(..., ge=0.0, le=1.0)
    min_precision: float = Field(..., ge=0.0, le=1.0)
    training_pks: list[str | int]
    training_verdicts: list[int]
    eval_pks: list[str | int]
    eval_verdicts: list[int]
    eval_metrics: EvalMetrics
    class_balance: ClassBalance
    trained_at: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check_lengths(self) -> "ClassifierMetadata":
        if len(self.training_pks) != len(self.training_verdicts):
            raise ValueError(
                "training_pks and training_verdicts must have equal length"
            )
        if len(self.eval_pks) != len(self.eval_verdicts):
            raise ValueError("eval_pks and eval_verdicts must have equal length")
        if len(self.model.coef) != self.embedding_dim + 1:
            raise ValueError(
                "ClassifierMetadata.model.coef length must equal "
                "embedding_dim + 1 (for nearest_fit_distance); got "
                f"{len(self.model.coef)} vs {self.embedding_dim + 1}"
            )
        if len(self.scaler.mean) != 1:
            raise ValueError(
                "ClassifierMetadata.scaler must scale exactly the scalar "
                "nearest_fit_distance feature; got "
                f"{len(self.scaler.mean)}-dim scaler"
            )
        return self


# ---------------------------------------------------------------------------
# Runtime dataclasses (not persisted directly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BorderlineSample:
    """A cohort PK with ``p_keep`` near the candidate threshold,
    surfaced for operator sanity-check at review time.
    """

    pk: PrimaryKey
    p_keep: float
    nearest_fit_distance: float
    decile: int


@dataclass(frozen=True)
class CohortProjection:
    """Projected KEEP/DROP split at a given threshold over the full
    Phase 2 cohort. Computed without writing any sidecar — used by the
    web UI's threshold slider preview.
    """

    threshold: float
    keep: int
    drop: int
    total: int
    per_decile_keep_rate: list[float | None]


@dataclass(frozen=True)
class EvalReport:
    """Everything :func:`run_apply_train` knows about the eval pass.

    Under ``eval_methodology="repeated_kfold"`` (the production default
    since the methodology migration), ``eval_n`` is the full labelled
    set N, ``eval_keep_n`` / ``eval_drop_n`` are class counts over the
    same N, and the metrics are computed on the pooled (y, p_keep) of
    size N × ``n_repeats``. ``threshold_selected_by_cv`` is the lowest
    threshold whose pooled precision clears ``min_precision``.
    """

    precision_at_threshold: float
    recall_at_threshold: float
    pr_curve: list[tuple[float, float, float]]
    threshold_default: float
    threshold_selected_by_cv: float | None
    cv_precision_mean: float | None
    cv_precision_std: float | None
    min_precision: float
    eval_n: int
    eval_keep_n: int
    eval_drop_n: int
    eval_methodology: Literal["single_split", "repeated_kfold"] = "single_split"
    n_splits: int | None = None
    n_repeats: int | None = None
    borderline_samples: list[BorderlineSample] = field(default_factory=list)

    @property
    def passes_bar(self) -> bool:
        return self.precision_at_threshold >= self.min_precision


@dataclass(frozen=True)
class ApplyLabel:
    """One cohort PK after the classifier has scored it."""

    pk: PrimaryKey
    verdict: Literal["KEEP", "DROP"]
    p_keep: float


@dataclass(frozen=True)
class ApplyTimings:
    """Wall-clock breakdown of one Phase 4 turn."""

    load_ms: float = 0.0
    embed_fetch_ms: float = 0.0
    train_ms: float = 0.0
    evaluate_ms: float = 0.0
    apply_ms: float = 0.0
    write_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class ApplyWriteResult:
    """Paths the writer produced; passed back to the caller for display."""

    classifier_path: Any
    eval_path: Any
    labels_path: Any
    meta_path: Any
    details_path: Any
    n_labels: int
