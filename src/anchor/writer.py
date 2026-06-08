"""Phase 2 output writers — `.phase2.jsonl` + `.phase2.meta.json`
plus a `turn="phase2"` block appended to the details sidecar.

All three share the same meta shape (the details block embeds the
meta verbatim and adds a ``per_fit`` array for the audit trail).

Entity fields (``chunk_content``, ``sample_id``, ``counselor_id``,
``term``, ``chunk_id``) are deliberately not persisted in the JSONL
records — the sidecar travels (shared across systems, attached to
tickets), and downstream consumers re-fetch entity data from Milvus
by pk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.search.evidence import PrimaryKey

from .config import RadiusScheme
from .loader import AnchorInputs
from .retrieve import AnchorCandidate, PerFitPages, RetrievalResult
from .threshold import CalibrationResult, distance_summary
from .validate import LooPerFit, RecoveryResult

if TYPE_CHECKING:
    from .runner import AnchorTimings, FrequencyGateSummary

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriteResult:
    jsonl_path: Path
    meta_path: Path
    details_path: Path
    n_records: int


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _json_safe_pk(pk: PrimaryKey) -> str | int:
    return pk if isinstance(pk, int) else str(pk)


def _round_or_null(value: float) -> float | None:
    """Serialize ``float('inf')`` / NaN as null so JSON stays valid."""
    import math

    if math.isinf(value) or math.isnan(value):
        return None
    return round(value, 6)


def _chunk_record(
    c: AnchorCandidate,
    t_prime_by_fit_pk: dict[PrimaryKey, float],
) -> dict[str, Any]:
    return {
        "pk": _json_safe_pk(c.pk),
        "nearest_fit_pk": _json_safe_pk(c.nearest_fit_pk),
        "nearest_fit_label": c.nearest_fit_label,
        "nearest_fit_distance": round(c.nearest_fit_distance, 6),
        "passed_threshold": True,
        "threshold_T_prime": round(t_prime_by_fit_pk[c.nearest_fit_pk], 6),
        "qualifying_fit_count": c.qualifying_fit_count,
        "qualifying_fit_pks": [_json_safe_pk(pk) for pk in c.qualifying_fit_pks],
    }


def _per_fit_pages_entry(p: PerFitPages) -> dict[str, Any]:
    return {
        "fit_pk": _json_safe_pk(p.fit_pk),
        "fit_chunk_id": p.fit_chunk_id,
        "pages_fetched": p.pages_fetched,
        "total_hits": p.total_hits,
        "final_kth_distance": _round_or_null(p.final_kth_distance),
    }


def _build_meta(
    *,
    inputs: AnchorInputs,
    calibration: CalibrationResult,
    recovery: RecoveryResult,
    retrieval: RetrievalResult,
    not_fit_intrusions: int,
    timings: "AnchorTimings",
    radius_scheme: RadiusScheme = RadiusScheme.PER_FIT,
    cohort_consistency: list[dict[str, Any]] | None = None,
    quality_gate_dropped: list[dict[str, Any]] | None = None,
    quality_gate_median_delta_pre_drop: float = 0.0,
    quality_gate_T_pre_drop: float = 0.0,
    quality_gate_multiplier_cutoff: float | None = None,
    quality_gate_median_floor_applied: bool = False,
    quality_gate_median_floor: float = 0.0,
    s2c_outlier_multiple: float = 0.0,
    n_discard_filtered: int = 0,
    frequency_gate: "FrequencyGateSummary | None" = None,
) -> dict[str, Any]:
    intruder_pks = [
        _json_safe_pk(c.pk)
        for c in retrieval.candidates
        if c.pk in inputs.not_fit_pks
    ]
    cohort_consistency_block = [
        {
            "fit_pk": _json_safe_pk(rec["fit_pk"]),
            "fit_chunk_id": rec["fit_chunk_id"],
            "own_chunk_retained": bool(rec["own_chunk_retained"]),
        }
        for rec in (cohort_consistency or [])
    ]
    return {
        "session_id": inputs.session_id,
        "query": inputs.query,
        "collection": inputs.collection,
        "ts": _now_iso(),
        "n_fit": calibration.n_fit,
        "calibration": {
            "T": round(calibration.T, 6),
            "deltas": [round(d, 6) for d in calibration.deltas],
            "T_primes": [round(t, 6) for t in calibration.T_primes],
            "T_prime_out": round(calibration.T_prime_out, 6),
            "radius_scheme": radius_scheme.value,
            "delta_summary": {
                k: round(v, 6)
                for k, v in distance_summary(calibration.deltas).items()
            },
            "T_prime_summary": {
                k: round(v, 6)
                for k, v in distance_summary(calibration.T_primes).items()
            },
            "span_loo_distances": [round(d, 6) for d in calibration.span_loo_distances],
        },
        "cohort_consistency": cohort_consistency_block,
        "quality_gate": {
            "s2c_outlier_multiple": s2c_outlier_multiple,
            "T_pre_drop": round(quality_gate_T_pre_drop, 6),
            "median_delta_pre_drop": round(quality_gate_median_delta_pre_drop, 6),
            # ``multiplier_applied`` is retained for backward compat
            # with older readers; under the floored-median logic
            # (issue #47) the rule is always live, so this field is
            # permanently True on new runs. ``median_floor_applied``
            # is the new discriminator: True when ``median(δ)`` sat
            # below ``_MEDIAN_DELTA_FLOOR`` and the floor stood in for
            # the median when computing ``multiplier_cutoff``.
            # ``median_floor`` reports the floor constant so consumers
            # don't have to track its source separately.
            "multiplier_applied": quality_gate_multiplier_cutoff is not None,
            "multiplier_cutoff": (
                round(quality_gate_multiplier_cutoff, 6)
                if quality_gate_multiplier_cutoff is not None
                else None
            ),
            "median_floor_applied": bool(quality_gate_median_floor_applied),
            "median_floor": round(float(quality_gate_median_floor), 6),
            "dropped": [
                {
                    "fit_pk": _json_safe_pk(rec["fit_pk"]),
                    "fit_chunk_id": rec["fit_chunk_id"],
                    "delta": round(float(rec["delta"]), 6),
                    "reasons": list(rec["reasons"]),
                }
                for rec in (quality_gate_dropped or [])
            ],
        },
        "discard_pk_filter": {
            "n_dropped": int(n_discard_filtered),
            "n_total": len(inputs.discard_pks),
        },
        "frequency_gate": (
            {
                "f_configured": frequency_gate.f_configured,
                "n_fit_after_quality_gate": frequency_gate.n_fit_after_quality_gate,
                "kept": frequency_gate.kept,
                "dropped": frequency_gate.dropped,
                "qualifying_count_distribution": dict(
                    frequency_gate.qualifying_count_distribution
                ),
                "qualifying_count_histogram": {
                    str(k): v
                    for k, v in frequency_gate.qualifying_count_histogram.items()
                },
            }
            if frequency_gate is not None
            else None
        ),
        "batch_size": retrieval.batch_size,
        "max_k": retrieval.max_k,
        "per_fit_pages": [_per_fit_pages_entry(p) for p in retrieval.per_fit_pages],
        "per_fit_budget_exhausted": [
            {
                "fit_pk": _json_safe_pk(p.fit_pk),
                "fit_chunk_id": p.fit_chunk_id,
                "final_kth_distance": _round_or_null(p.final_kth_distance),
            }
            for p in retrieval.per_fit_pages
            if p.budget_exhausted
        ],
        "loo_recovery": {
            "recovered": recovery.recovered,
            "total": recovery.total,
            "verdict": recovery.verdict,
            "missed_fits": [
                {
                    "fit_pk": _json_safe_pk(p.fit_pk),
                    "fit_chunk_id": p.fit_chunk_id,
                }
                for p in recovery.missed_fits
            ],
        },
        "not_fit_intrusion": {
            "passed": not_fit_intrusions,
            "total": len(inputs.not_fit_pks),
            "intruder_pks": intruder_pks,
        },
        "output_count": len(retrieval.candidates),
        "distance_distribution": {
            k: round(v, 6)
            for k, v in distance_summary(
                [c.nearest_fit_distance for c in retrieval.candidates]
            ).items()
        },
        "milvus_index_type": inputs.milvus_index_type,
        "milvus_index_params": dict(inputs.milvus_index_params),
        "embed_model_id": inputs.embed_model_id,
        "timings": {
            "load_ms": round(timings.load_ms, 2),
            "calibrate_ms": round(timings.calibrate_ms, 2),
            "loo_ms": round(timings.loo_ms, 2),
            "retrieve_ms": round(timings.retrieve_ms, 2),
            "total_ms": round(timings.total_ms, 2),
        },
    }


def _build_per_fit(
    inputs: AnchorInputs,
    calibration: CalibrationResult,
    recovery: RecoveryResult,
    retrieval: RetrievalResult,
) -> list[dict[str, Any]]:
    pages_by_pk: dict[PrimaryKey, PerFitPages] = {
        p.fit_pk: p for p in retrieval.per_fit_pages
    }
    by_pk_loo: dict[PrimaryKey, LooPerFit] = {p.fit_pk: p for p in recovery.per_fit}
    out: list[dict[str, Any]] = []
    for fit, delta, T_prime in zip(
        inputs.fits, calibration.deltas, calibration.T_primes
    ):
        loo = by_pk_loo.get(fit.pk)
        pages = pages_by_pk.get(fit.pk)
        out.append(
            {
                "fit_chunk_id": fit.chunk_id,
                "fit_pk": _json_safe_pk(fit.pk),
                "span_text": fit.span_text,
                "span_line_indices": list(fit.span_line_indices),
                "span_to_own_chunk_distance": round(delta, 6),
                "T_prime": round(T_prime, 6),
                "loo": {
                    "recovered": bool(loo.recovered) if loo else False,
                    "recalibrated_T": (
                        round(loo.recalibrated_T, 6) if loo else None
                    ),
                    "rank_of_own_pk": loo.rank_of_own_pk if loo else None,
                    "distance_of_own_pk": (
                        round(loo.distance_of_own_pk, 6)
                        if loo and loo.distance_of_own_pk is not None
                        else None
                    ),
                },
                "pages_fetched": pages.pages_fetched if pages else 0,
                "total_hits": pages.total_hits if pages else 0,
                "final_kth_distance": (
                    _round_or_null(pages.final_kth_distance) if pages else None
                ),
                "budget_exhausted": bool(pages.budget_exhausted) if pages else False,
            }
        )
    return out


def write_anchor_outputs(
    *,
    inputs: AnchorInputs,
    calibration: CalibrationResult,
    recovery: RecoveryResult,
    retrieval: RetrievalResult,
    not_fit_intrusions: int,
    runs_dir: Path,
    timings: "AnchorTimings",
    radius_scheme: RadiusScheme = RadiusScheme.PER_FIT,
    cohort_consistency: list[dict[str, Any]] | None = None,
    quality_gate_dropped: list[dict[str, Any]] | None = None,
    quality_gate_median_delta_pre_drop: float = 0.0,
    quality_gate_T_pre_drop: float = 0.0,
    quality_gate_multiplier_cutoff: float | None = None,
    quality_gate_median_floor_applied: bool = False,
    quality_gate_median_floor: float = 0.0,
    s2c_outlier_multiple: float = 0.0,
    n_discard_filtered: int = 0,
    frequency_gate: "FrequencyGateSummary | None" = None,
) -> WriteResult:
    """Write .phase2.jsonl + .phase2.meta.json and append the
    `turn=phase2` block to the details sidecar."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    session_id = inputs.session_id
    jsonl_path = runs_dir / f"{session_id}.phase2.jsonl"
    meta_path = runs_dir / f"{session_id}.phase2.meta.json"
    details_path = runs_dir / f"{session_id}.details.jsonl"

    # Under ``decoupled`` the per-record ``threshold_T_prime`` is the
    # session-wide cap for every FIT; under ``per_fit`` it is the
    # attracting FIT's own T'_i.
    if radius_scheme is RadiusScheme.DECOUPLED:
        t_prime_by_fit_pk: dict[PrimaryKey, float] = {
            f.pk: calibration.T_prime_out for f in inputs.fits
        }
    else:
        t_prime_by_fit_pk = {
            f.pk: t for f, t in zip(inputs.fits, calibration.T_primes)
        }

    # Truncate-write the JSONL so re-runs don't accumulate stale lines.
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for cand in retrieval.candidates:
            fp.write(
                json.dumps(
                    _chunk_record(cand, t_prime_by_fit_pk),
                    ensure_ascii=False,
                )
            )
            fp.write("\n")
        fp.flush()

    meta = _build_meta(
        inputs=inputs,
        calibration=calibration,
        recovery=recovery,
        retrieval=retrieval,
        not_fit_intrusions=not_fit_intrusions,
        timings=timings,
        radius_scheme=radius_scheme,
        cohort_consistency=cohort_consistency,
        quality_gate_dropped=quality_gate_dropped,
        quality_gate_median_delta_pre_drop=quality_gate_median_delta_pre_drop,
        quality_gate_T_pre_drop=quality_gate_T_pre_drop,
        quality_gate_multiplier_cutoff=quality_gate_multiplier_cutoff,
        quality_gate_median_floor_applied=quality_gate_median_floor_applied,
        quality_gate_median_floor=quality_gate_median_floor,
        s2c_outlier_multiple=s2c_outlier_multiple,
        n_discard_filtered=n_discard_filtered,
        frequency_gate=frequency_gate,
    )
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Append phase2 block to details sidecar. Append-only by design
    # (two Phase 2 invocations → two blocks).
    details_block = {
        "turn": "phase2",
        "phase2": {
            **meta,
            "per_fit": _build_per_fit(inputs, calibration, recovery, retrieval),
        },
    }
    with details_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(details_block, ensure_ascii=False) + "\n")

    t_prime_summary = distance_summary(calibration.T_primes)
    log.info(
        "phase2: wrote %d candidates → %s (verdict=%s, T' med=%.4f "
        "[%.4f–%.4f])",
        len(retrieval.candidates),
        jsonl_path,
        recovery.verdict,
        t_prime_summary["median"],
        t_prime_summary["min"],
        t_prime_summary["max"],
    )
    return WriteResult(
        jsonl_path=jsonl_path,
        meta_path=meta_path,
        details_path=details_path,
        n_records=len(retrieval.candidates),
    )
