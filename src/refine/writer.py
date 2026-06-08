"""Phase 3 output writers — four sidecars + details append.

Mirrors :mod:`src.anchor.writer`. Truncate-writes the rubric prompt
markdown, the rubric metadata JSON, and the per-chunk evidence
JSONL. Appends a ``turn="phase3"`` block to the session's details
sidecar.

The four sidecar paths follow the existing
``runs/{session_id}.phase{n}.{kind}.{ext}`` convention; no central
helper exists in the codebase, so the pattern is hardcoded here as
it is in Phase 2's writer.

Two-stage write contract
------------------------

Phase 3's writer fires in two stages so the on-disk layout reflects
"judge done, awaiting operator" distinctly from "operator agreed,
run finalised":

- Stage A (judge done, pre-finalise) — :func:`write_refine_stage_a`
  truncate-writes ``rubric.json`` and ``evidence.jsonl`` only.
- Stage B (finalise on agree) — :func:`write_refine_stage_b`
  truncate-writes ``prompt.md`` and ``meta.json`` and appends the
  ``turn="phase3"`` block to ``details.jsonl``.

The legacy :func:`write_refine_outputs` runs both stages back-to-back
for callers that don't yet need the split (e.g. ``--auto-accept``
collapses to the same on-disk result either way).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.search.evidence import PrimaryKey

from .config import RefineConfig
from .derive import metadata_to_json
from .judge import JudgeResult, JudgeVerdictRecord
from .sample import StratifiedSample
from .schema import RubricMetadata

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefineWriteResult:
    """Paths the writer produced; passed back to the caller for display."""

    prompt_path: Path
    rubric_path: Path
    evidence_path: Path
    meta_path: Path
    details_path: Path
    n_verdicts: int


@dataclass(frozen=True)
class RefineStageAResult:
    """Subset returned by :func:`write_refine_stage_a`.

    Stage A only produces the rubric metadata and evidence sidecars;
    the prompt, meta, and details sidecars are stage B's responsibility.
    """

    rubric_path: Path
    evidence_path: Path
    n_verdicts: int


@dataclass(frozen=True)
class RefineTimings:
    """Wall-clock breakdown of one Phase 3 turn."""

    derive_ms: float = 0.0
    sample_ms: float = 0.0
    judge_total_ms: float = 0.0
    write_ms: float = 0.0
    total_ms: float = 0.0


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _round_or_null(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isinf(value) or math.isnan(value):
        return None
    return round(value, 6)


def _json_safe_pk(pk: PrimaryKey) -> str | int:
    return pk if isinstance(pk, int) else str(pk)


def _verdict_record_dict(record: JudgeVerdictRecord) -> dict[str, Any]:
    indices: list[int] | None
    if record.evidence_line_indices is None:
        indices = None
    else:
        indices = list(record.evidence_line_indices)
    return {
        "pk": _json_safe_pk(record.pk),
        "nearest_fit_distance": _round_or_null(record.nearest_fit_distance),
        "decile": record.decile,
        "chunk_content": record.chunk_content,
        "verdict": record.verdict,
        "evidence_line_indices": indices,
        "failed_check": record.failed_check,
        "reason": record.reason,
        "latency_ms": _round_or_null(record.latency_ms),
        "attempts": record.attempts,
        "rubric_version": record.rubric_version,
        "prompt_sha256": record.prompt_sha256,
        "agreement": None,
    }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    k = max(0, min(len(sorted_v) - 1, int(round((p / 100.0) * (len(sorted_v) - 1)))))
    return sorted_v[k]


def _latency_summary(records: list[JudgeVerdictRecord]) -> dict[str, Any]:
    latencies = [
        float(r.latency_ms)
        for r in records
        if r.latency_ms is not None and r.verdict != "ERROR"
    ]
    if not latencies:
        return {
            "count": 0,
            "p50_ms": None,
            "p90_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(latencies),
        "p50_ms": _round_or_null(_percentile(latencies, 50)),
        "p90_ms": _round_or_null(_percentile(latencies, 90)),
        "p99_ms": _round_or_null(_percentile(latencies, 99)),
        "max_ms": _round_or_null(max(latencies)),
    }


def _per_decile_keep_rate(
    records: list[JudgeVerdictRecord], n_bins: int
) -> list[float | None]:
    keep = [0] * n_bins
    drop = [0] * n_bins
    for r in records:
        if r.verdict == "ERROR":
            continue
        if r.failed_check == "auto_drop_known_intruder":
            continue
        if 0 <= r.decile < n_bins:
            if r.verdict == "KEEP":
                keep[r.decile] += 1
            elif r.verdict == "DROP":
                drop[r.decile] += 1
    out: list[float | None] = []
    for k, d in zip(keep, drop):
        total = k + d
        out.append(round(k / total, 6) if total > 0 else None)
    return out


def _failed_check_histogram(
    records: list[JudgeVerdictRecord],
) -> dict[str, int]:
    """Count DROP verdicts per check id, excluding auto-DROPs and ERRORs."""
    out: dict[str, int] = {}
    for r in records:
        if r.verdict != "DROP":
            continue
        if r.failed_check is None:
            continue
        if r.failed_check == "auto_drop_known_intruder":
            continue
        out[r.failed_check] = out.get(r.failed_check, 0) + 1
    return out


def _verdict_counts(records: list[JudgeVerdictRecord]) -> dict[str, int]:
    keep = drop = err = auto = 0
    for r in records:
        if r.verdict == "ERROR":
            err += 1
        elif r.failed_check == "auto_drop_known_intruder":
            auto += 1
        elif r.verdict == "KEEP":
            keep += 1
        elif r.verdict == "DROP":
            drop += 1
    return {"KEEP": keep, "DROP": drop, "ERROR": err, "auto_drop": auto}


def _prompt_basename(session_id: str) -> str:
    return f"{session_id}.phase3.prompt.md"


def write_refine_stage_a(
    *,
    session_id: str,
    runs_dir: Path,
    rubric_metadata: RubricMetadata,
    judge_result: JudgeResult,
) -> RefineStageAResult:
    """Stage A — judge done, awaiting operator decision.

    Truncate-writes ``<sid>.phase3.rubric.json`` and
    ``<sid>.phase3.evidence.jsonl``. Does not write ``prompt.md`` or
    ``meta.json``, and does not append to ``details.jsonl`` — those
    are stage B's responsibility.

    The rubric.json's ``prompt_path`` field is synced to the eventual
    ``<sid>.phase3.prompt.md`` basename even though that file lands
    in stage B. This keeps rubric.json internally consistent with
    its eventual sibling once the operator agrees.

    Idempotent: re-running stage A truncate-overwrites both files.
    Disagree at review bumps ``rubric.version`` and re-runs derive +
    judge; the next stage A call replaces the previous outputs.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    rubric_path = runs_dir / f"{session_id}.phase3.rubric.json"
    evidence_path = runs_dir / f"{session_id}.phase3.evidence.jsonl"

    metadata = rubric_metadata.model_copy(
        update={"prompt_path": _prompt_basename(session_id)}
    )
    rubric_path.write_text(
        metadata_to_json(metadata) + "\n", encoding="utf-8"
    )

    with evidence_path.open("w", encoding="utf-8") as fp:
        for record in judge_result.verdicts:
            fp.write(
                json.dumps(_verdict_record_dict(record), ensure_ascii=False)
            )
            fp.write("\n")
        fp.flush()

    log.info(
        "Phase 3 stage A wrote rubric+evidence; %d verdict(s); "
        "rubric_version=%d",
        len(judge_result.verdicts),
        metadata.version,
    )

    return RefineStageAResult(
        rubric_path=rubric_path,
        evidence_path=evidence_path,
        n_verdicts=len(judge_result.verdicts),
    )


def write_refine_stage_b(
    *,
    session_id: str,
    runs_dir: Path,
    rubric_text: str,
    rubric_metadata: RubricMetadata,
    judge_result: JudgeResult,
    sample: StratifiedSample,
    cfg: RefineConfig,
    derive_model_id: str,
    judge_model_id: str,
    meta_prompt_path: str,
    meta_prompt_sha256: str,
    operator_decision: str,
    timings: RefineTimings,
) -> RefineWriteResult:
    """Stage B — operator agreed, finalise the run.

    Truncate-writes ``<sid>.phase3.prompt.md`` and
    ``<sid>.phase3.meta.json``, and appends a ``turn="phase3"`` block
    to ``<sid>.details.jsonl``. Stage A's rubric.json and
    evidence.jsonl are *not* re-written here — they stay as stage A
    produced them at the moment the judge run completed.

    ``operator_decision`` is normally ``"agree"`` — the caller has
    decided to ship — but the parameter exists so a future "ship a
    no-go audit trail" mode could pass ``"abort"``.

    The returned :class:`RefineWriteResult` carries paths to all
    four sidecars (the stage A pair are referenced by path even
    though stage B did not write them), matching the legacy
    :func:`write_refine_outputs` contract.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = runs_dir / f"{session_id}.phase3.prompt.md"
    rubric_path = runs_dir / f"{session_id}.phase3.rubric.json"
    evidence_path = runs_dir / f"{session_id}.phase3.evidence.jsonl"
    meta_path = runs_dir / f"{session_id}.phase3.meta.json"
    details_path = runs_dir / f"{session_id}.details.jsonl"

    # Sync metadata.prompt_path to the persisted sidecar's basename
    # for the meta payload's prompt_sha256 + prompt_path bookkeeping.
    metadata = rubric_metadata.model_copy(
        update={"prompt_path": prompt_path.name}
    )

    prompt_path.write_text(rubric_text, encoding="utf-8")

    meta_payload = {
        "session_id": session_id,
        "query": metadata.query,
        "ts": _now_iso(),
        "derive_model_id": derive_model_id,
        "judge_model_id": judge_model_id,
        "meta_prompt_path": meta_prompt_path,
        "meta_prompt_sha256": meta_prompt_sha256,
        "prompt_path": str(prompt_path.name),
        "prompt_sha256": metadata.prompt_sha256,
        "rubric_version": metadata.version,
        "sample_config": {
            "sample_size": cfg.sample_size,
            "n_bins": cfg.n_bins,
            "seed": cfg.seed,
            "auto_drop_known_intruders": cfg.auto_drop_known_intruders,
            "sample_strategy": "stratified",
        },
        "decile_boundaries": [
            _round_or_null(b) for b in sample.decile_boundaries
        ],
        "per_decile_count": list(sample.per_decile_count),
        "per_decile_keep_rate": _per_decile_keep_rate(
            judge_result.verdicts, cfg.n_bins
        ),
        "excluded_rated_pks": sorted(
            (_json_safe_pk(pk) for pk in sample.excluded_pks),
            key=lambda x: str(x),
        ),
        "verdict_counts": _verdict_counts(judge_result.verdicts),
        "failed_check_histogram": _failed_check_histogram(
            judge_result.verdicts
        ),
        "latency_summary": _latency_summary(judge_result.verdicts),
        "parse_error_count": judge_result.parse_error_count,
        "api_error_count": judge_result.api_error_count,
        "timings": {
            "derive_ms": round(timings.derive_ms, 2),
            "sample_ms": round(timings.sample_ms, 2),
            "judge_total_ms": round(timings.judge_total_ms, 2),
            "write_ms": round(timings.write_ms, 2),
            "total_ms": round(timings.total_ms, 2),
        },
        "operator_decision": operator_decision,
    }
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    details_block = {"turn": "phase3", "phase3": meta_payload}
    with details_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(details_block, ensure_ascii=False) + "\n")

    log.info(
        "Phase 3 stage B wrote prompt+meta+details; decision=%s; "
        "rubric_version=%d",
        operator_decision,
        metadata.version,
    )

    return RefineWriteResult(
        prompt_path=prompt_path,
        rubric_path=rubric_path,
        evidence_path=evidence_path,
        meta_path=meta_path,
        details_path=details_path,
        n_verdicts=len(judge_result.verdicts),
    )


def write_refine_outputs(
    *,
    session_id: str,
    runs_dir: Path,
    rubric_text: str,
    rubric_metadata: RubricMetadata,
    judge_result: JudgeResult,
    sample: StratifiedSample,
    cfg: RefineConfig,
    derive_model_id: str,
    judge_model_id: str,
    meta_prompt_path: str,
    meta_prompt_sha256: str,
    operator_decision: str,
    timings: RefineTimings,
) -> RefineWriteResult:
    """Legacy wrapper — runs stage A then stage B back-to-back.

    Preserves the original four-sidecar atomic-write contract for
    callers that don't need the judge-done / finalise distinction
    (notably the ``--auto-accept`` headless path). New code should
    call :func:`write_refine_stage_a` and :func:`write_refine_stage_b`
    directly so the on-disk state reflects the operator's decision
    point.
    """
    write_refine_stage_a(
        session_id=session_id,
        runs_dir=runs_dir,
        rubric_metadata=rubric_metadata,
        judge_result=judge_result,
    )
    return write_refine_stage_b(
        session_id=session_id,
        runs_dir=runs_dir,
        rubric_text=rubric_text,
        rubric_metadata=rubric_metadata,
        judge_result=judge_result,
        sample=sample,
        cfg=cfg,
        derive_model_id=derive_model_id,
        judge_model_id=judge_model_id,
        meta_prompt_path=meta_prompt_path,
        meta_prompt_sha256=meta_prompt_sha256,
        operator_decision=operator_decision,
        timings=timings,
    )
