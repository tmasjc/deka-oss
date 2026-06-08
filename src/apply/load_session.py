"""Phase 4 session loader — Phase 3 sidecars + Phase 2 cohort.

Phase 4 needs three slices of a finalised Phase 3 session:

- The rubric pin (``rubric_version`` + ``prompt_sha256``) from
  ``runs/{sid}.phase3.rubric.json`` — locked into the classifier so the
  reuse path can refuse on drift.
- The judged sample from ``runs/{sid}.phase3.evidence.jsonl`` — labels
  for training. Errors and auto-drops are filtered out (they don't
  represent operator-meaningful KEEP/DROP boundaries).
- The full Phase 2 cohort from ``runs/{sid}.phase2.jsonl`` — features
  (``nearest_fit_distance``) and the universe to apply to.

The collection name + embed-model id needed for the Milvus lookup live
in ``runs/{sid}.details.jsonl``'s ``search`` block — same path Phase 2
reads from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.search.evidence import PrimaryKey

from .errors import ApplyGuardrailError, ApplyLoadError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingLabel:
    """One row from ``phase3.evidence.jsonl`` projected to the fields
    Phase 4's trainer needs.
    """

    pk: PrimaryKey
    nearest_fit_distance: float
    decile: int
    verdict: str  # "KEEP" or "DROP" — ERRORs filtered out


@dataclass(frozen=True)
class CohortRow:
    """One row of ``phase2.jsonl`` projected to apply-time features."""

    pk: PrimaryKey
    nearest_fit_distance: float


@dataclass(frozen=True)
class RubricPin:
    """The lock fields from ``phase3.rubric.json`` Phase 4 stamps onto
    the classifier sidecar.
    """

    rubric_version: int
    prompt_sha256: str


@dataclass(frozen=True)
class SearchPin:
    """The collection + embed-service identity needed to pull dense
    embeddings from Milvus at apply time.
    """

    collection: str
    embed_url: str
    embed_model_id: str


@dataclass(frozen=True)
class Phase4SessionInputs:
    """Everything Phase 4 needs from disk before it touches Milvus."""

    session_id: str
    rubric: RubricPin
    search: SearchPin
    labels: list[TrainingLabel]
    cohort: list[CohortRow]


def load_phase4_session_inputs(
    session_id: str, *, runs_dir: Path
) -> Phase4SessionInputs:
    """Hydrate Phase 4's inputs from the four on-disk sidecars.

    Validates that Phase 3 finalised (rubric.json present, evidence.jsonl
    non-empty after filtering ERRORs) and Phase 2 produced a non-empty
    cohort. Raises :class:`ApplyLoadError` on any structural problem.
    """
    rubric = _load_rubric_pin(runs_dir, session_id)
    search = _load_search_pin(runs_dir, session_id)
    labels = _load_training_labels(runs_dir, session_id)
    cohort = _load_cohort_rows(runs_dir, session_id)

    if not labels:
        raise ApplyLoadError(
            f"Phase 3 evidence for {session_id} has no non-ERROR rows; "
            "Phase 4 needs labelled examples to train. Re-run Phase 3."
        )
    if not cohort:
        raise ApplyLoadError(
            f"Phase 2 cohort for {session_id} is empty; Phase 4 has "
            "nothing to apply to."
        )

    log.info(
        "Phase 4 load: session=%s rubric_version=%d labels=%d cohort=%d",
        session_id,
        rubric.rubric_version,
        len(labels),
        len(cohort),
    )
    return Phase4SessionInputs(
        session_id=session_id,
        rubric=rubric,
        search=search,
        labels=labels,
        cohort=cohort,
    )


def verify_rubric_pin(*, classifier_pin: RubricPin, session_pin: RubricPin) -> None:
    """Hard guardrail for the reuse path.

    Refuses to apply a persisted classifier whose rubric pin does not
    match the session's current ``phase3.rubric.json``. Raises
    :class:`ApplyGuardrailError` on any mismatch — version, prompt
    sha256, or both.
    """
    if classifier_pin.prompt_sha256 != session_pin.prompt_sha256:
        raise ApplyGuardrailError(
            "Rubric prompt SHA256 mismatch: classifier was trained against "
            f"{classifier_pin.prompt_sha256[:12]}..., session's current rubric "
            f"is {session_pin.prompt_sha256[:12]}.... Re-train Phase 4 or "
            "roll the rubric back."
        )
    if classifier_pin.rubric_version != session_pin.rubric_version:
        raise ApplyGuardrailError(
            "Rubric version mismatch: classifier locked to version "
            f"{classifier_pin.rubric_version}, session's current rubric is "
            f"version {session_pin.rubric_version}. Re-train Phase 4."
        )


def _load_rubric_pin(runs_dir: Path, session_id: str) -> RubricPin:
    path = runs_dir / f"{session_id}.phase3.rubric.json"
    if not path.exists():
        raise ApplyLoadError(
            f"Phase 3 rubric sidecar missing: {path}. Run Phase 3 first "
            "(`python -m src.refine <session_id>`) before Phase 4."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApplyLoadError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return RubricPin(
            rubric_version=int(payload["version"]),
            prompt_sha256=str(payload["prompt_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApplyLoadError(
            f"{path} missing or malformed required field "
            f"('version' or 'prompt_sha256'): {exc}"
        ) from exc


def _load_search_pin(runs_dir: Path, session_id: str) -> SearchPin:
    """Read the first ``search`` block from ``<sid>.details.jsonl``.

    Phase 2's loader uses the same source of truth; we read it again
    rather than re-derive so the two phases agree on the embed-model
    id stamped into the classifier sidecar.
    """
    path = runs_dir / f"{session_id}.details.jsonl"
    if not path.exists():
        raise ApplyLoadError(
            f"Details sidecar missing: {path}. Phase 4 reads collection "
            "and embed-model id from the Phase 1 details block."
        )
    block: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ApplyLoadError(f"{path} line malformed: {exc}") from exc
        if "search" in rec:
            block = rec["search"]
            break
    if block is None:
        raise ApplyLoadError(f"No 'search' block found in {path}")
    collection = str(block.get("collection") or "")
    embed_url = str(block.get("embed_url") or "")
    embed_model_id = str(block.get("embed_model_id") or "")
    if not collection:
        raise ApplyLoadError(f"{path} details.search missing 'collection'")
    if not embed_url:
        raise ApplyLoadError(f"{path} details.search missing 'embed_url'")
    return SearchPin(
        collection=collection,
        embed_url=embed_url,
        embed_model_id=embed_model_id,
    )


def _load_training_labels(runs_dir: Path, session_id: str) -> list[TrainingLabel]:
    """Read evidence.jsonl, drop ERROR rows, keep KEEP/DROP labels.

    Auto-drop rows are retained: the operator's intruder filter is a
    real signal about the boundary and the classifier should learn it.
    """
    path = runs_dir / f"{session_id}.phase3.evidence.jsonl"
    if not path.exists():
        raise ApplyLoadError(
            f"Phase 3 evidence sidecar missing: {path}. Run Phase 3 first."
        )
    out: list[TrainingLabel] = []
    for line_num, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ApplyLoadError(f"{path} line {line_num} malformed: {exc}") from exc
        verdict = str(entry.get("verdict", ""))
        if verdict not in ("KEEP", "DROP"):
            # ERROR rows / unknown verdicts: skip silently — Phase 3
            # already reported their count in phase3.meta.json.
            continue
        try:
            out.append(
                TrainingLabel(
                    pk=entry["pk"],
                    nearest_fit_distance=float(entry["nearest_fit_distance"]),
                    decile=int(entry["decile"]),
                    verdict=verdict,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApplyLoadError(
                f"{path} line {line_num} missing or malformed field: {exc}"
            ) from exc
    return out


def _load_cohort_rows(runs_dir: Path, session_id: str) -> list[CohortRow]:
    """Read phase2.jsonl into ``CohortRow``s preserving file order."""
    path = runs_dir / f"{session_id}.phase2.jsonl"
    if not path.exists():
        raise ApplyLoadError(
            f"Phase 2 cohort sidecar missing: {path}. Run Phase 2 first."
        )
    out: list[CohortRow] = []
    for line_num, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ApplyLoadError(f"{path} line {line_num} malformed: {exc}") from exc
        try:
            out.append(
                CohortRow(
                    pk=entry["pk"],
                    nearest_fit_distance=float(entry["nearest_fit_distance"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApplyLoadError(
                f"{path} line {line_num} missing or malformed field: {exc}"
            ) from exc
    return out
