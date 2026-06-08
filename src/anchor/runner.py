"""Orchestrate the Phase 2 span-anchored pass end-to-end.

Pipeline:

1. ``load_anchor_inputs`` — progress log → FITs with spans + span/
   chunk embeddings; model-version check.
2. ``derive_threshold_prime`` — T (p90 span-LOO) + δ (median span-to-
   chunk) + T'.
3. ``loo_fit_recovery`` — per-slice iterator LOO gate. FAILED aborts
   before the main pass.
4. ``retrieve_anchored`` — main pass against the target collection
   with ``search_iterator``-based widening per FIT.
5. ``write_anchor_outputs`` — unless ``dry_run``.

Pure of UI dependencies. Both the CLI (``python -m src.anchor``)
and the web UI hand-off call :func:`run_anchor`.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from src.search.config import SearchConfig, load_default_config

from .config import HarvestConfig, RadiusScheme, load_harvest_config
from .errors import AnchorError, AnchorValidationError
from .loader import AnchorInputs, FitChunk, load_anchor_inputs
from .retrieve import RetrievalResult, retrieve_anchored
from .threshold import CalibrationResult, derive_threshold_prime
from .validate import RecoveryResult, count_not_fit_intrusions, loo_fit_recovery
from .writer import WriteResult, write_anchor_outputs

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

# LOO recalibration needs ``rest >= 2``, so the quality gate's
# survivor cohort must have ≥ 3 FITs. Dropping below this is
# catastrophic and aborts the run.
_GATE_SURVIVOR_FLOOR = 3

# Below this median(δ), the cohort sits in the float-noise band — a
# 1e-3 cosine distance corresponds to a 99.9 %+ span/chunk match, the
# regime where the rater accepted "span = whole chunk." Used as the
# upper bound of the bimodal-cohort warning condition (see the gate
# below).
_MEDIAN_DELTA_EPSILON = 1e-3

# Floor anchoring the ``δ > k · median(δ)`` cutoff when the cohort's
# median sinks into the float-noise band. The cutoff becomes
# ``k · max(median(δ), _MEDIAN_DELTA_FLOOR)`` so two regimes hand off
# cleanly (issue #47):
#
# * All-whole-chunk cohort (every δ ≈ 1e-6): median is below the
#   floor → cutoff = k · floor = 0.015 at k=3. No δ exceeds — the
#   float-noise protection the older epsilon-disable branch gave is
#   preserved.
# * Mixed cohort (≥ half whole-chunk + a minority of real-magnitude
#   anchors): median is still below the floor → cutoff = 0.015, and
#   real outliers δ ≥ 0.03 are caught. The old branch silently
#   disabled the rule and let these slip through, producing 3×
#   harvest-size swings on identical query/scope.
# * Well-behaved cohort (median ≥ floor): the floor is inert and the
#   cutoff tracks the median as before.
#
# 0.005 sits one order below the legitimate-outlier band documented
# in issue #47 (0.03–0.12 in the two reference sessions) and three
# orders above float noise on unit vectors. Hardcoded rather than a
# HarvestConfig key — the value is wedded to the regime split, not
# something operators tune per session.
_MEDIAN_DELTA_FLOOR = 0.005


@dataclass(frozen=True)
class AnchorTimings:
    load_ms: float
    calibrate_ms: float
    loo_ms: float
    retrieve_ms: float
    total_ms: float


@dataclass(frozen=True)
class FrequencyGateSummary:
    """Summary of the anchor-frequency gate's effect on the cohort.

    ``f_configured`` is the user-supplied ``harvest.anchor_frequency_gate``;
    ``n_fit_after_quality_gate`` is the FIT count surviving the anchor-
    quality gate (the ceiling f could possibly equal). ``kept`` and
    ``dropped`` are counts over the candidate list (FIT-own pks
    excluded). ``qualifying_count_distribution`` reports
    ``{min, median, max}`` of ``qualifying_fit_count`` over the kept
    cohort — a quick lever for tuning higher / lower f values.
    ``qualifying_count_histogram`` is the full ``{count: pk_count}``
    map over the kept cohort, sufficient to derive at_f/above_f counts
    or a shape plot downstream.
    """

    f_configured: int
    n_fit_after_quality_gate: int
    kept: int
    dropped: int
    qualifying_count_distribution: dict[str, int] = field(default_factory=dict)
    qualifying_count_histogram: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AnchorResult:
    inputs: AnchorInputs
    calibration: CalibrationResult
    recovery: RecoveryResult
    retrieval: RetrievalResult
    write: WriteResult | None
    not_fit_intrusions: int
    timings: AnchorTimings
    radius_scheme: RadiusScheme = RadiusScheme.PER_FIT
    cohort_consistency: list[dict[str, Any]] = field(default_factory=list)
    quality_gate_dropped: list[dict[str, Any]] = field(default_factory=list)
    quality_gate_median_delta_pre_drop: float = 0.0
    quality_gate_T_pre_drop: float = 0.0
    # ``quality_gate_multiplier_cutoff`` is the cutoff used for the
    # ``δ > k · median(δ)`` rule. Under the floored-median logic
    # (issue #47) it's always populated; the new
    # ``quality_gate_median_floor_applied`` flag distinguishes
    # "median ≥ floor → cutoff tracks the median" from
    # "median < floor → cutoff backstopped by ``_MEDIAN_DELTA_FLOOR``"
    # (the bimodality / whole-chunk-dominated regime). Surfacing both
    # lets operators see which regime the gate was in (issues #47, #48).
    quality_gate_multiplier: float = 0.0
    quality_gate_multiplier_cutoff: float | None = None
    quality_gate_median_floor_applied: bool = False
    n_discard_filtered: int = 0
    frequency_gate: FrequencyGateSummary | None = None


def _emit(progress: ProgressCallback | None, msg: str) -> None:
    log.info(msg)
    if progress is not None:
        try:
            progress(msg)
        except Exception as exc:  # noqa: BLE001
            log.debug("progress callback raised: %s", exc)


def _apply_quality_gate(
    *,
    inputs: AnchorInputs,
    calibration: CalibrationResult,
    harvest_cfg: HarvestConfig,
    progress: ProgressCallback | None,
) -> tuple[
    AnchorInputs,
    CalibrationResult,
    list[dict[str, Any]],
    float,
    float,
    float | None,
    bool,
]:
    """Drop FITs whose span-to-own-chunk δ_i flags as a bad anchor.

    Two criteria, OR-combined:

    * ``δ_i > T`` — span further from its own chunk than a typical
      span sits from another span. The anchor is structurally
      degenerate; its ``T'_i`` would admit chunks beyond the cohort's
      internal spread.
    * ``δ_i > k · max(median(δ), _MEDIAN_DELTA_FLOOR)``
      (``k = harvest.s2c_outlier_multiple``) — robust-outlier
      heuristic catching FITs whose δ_i sits far above the cohort's
      typical span-to-chunk distance. The floor prevents the cutoff
      from collapsing to ~0 on whole-chunk-dominated cohorts where
      the median sinks into float noise; see ``_MEDIAN_DELTA_FLOOR``
      for the regime analysis.

    Returns ``(inputs', calibration', dropped, median_pre, T_pre,
    multiplier_cutoff, median_floor_applied)``. ``multiplier_cutoff``
    is always populated under the floored logic (issue #47).
    ``median_floor_applied`` is ``True`` when the floor stood in for
    the median — i.e. ``median(δ) < _MEDIAN_DELTA_FLOOR``. If no FIT
    fires, ``inputs`` and ``calibration`` pass through unchanged and
    ``dropped`` is empty. Aborts with :class:`AnchorValidationError`
    when survivors fall below the structural LOO floor
    (``_GATE_SURVIVOR_FLOOR``).
    """
    deltas_pre = list(calibration.deltas)
    T_pre = calibration.T
    median_delta = median(deltas_pre) if deltas_pre else 0.0
    median_floor_applied = median_delta < _MEDIAN_DELTA_FLOOR
    effective_median = max(median_delta, _MEDIAN_DELTA_FLOOR)
    multiple_cutoff = harvest_cfg.s2c_outlier_multiple * effective_median
    multiplier_cutoff: float | None = multiple_cutoff

    # Loud warning when the cohort straddles the float-noise / real-
    # magnitude boundary: median in the noise band, but max δ at
    # least an order of magnitude larger. This is the exact bimodal
    # regime that masked real outliers under the old epsilon-disable
    # logic — surface it so operators investigating divergent
    # harvests on identical inputs can spot the regime in logs.
    if (
        deltas_pre
        and median_delta < _MEDIAN_DELTA_EPSILON
        and max(deltas_pre) > _MEDIAN_DELTA_EPSILON
    ):
        _emit(
            progress,
            f"Quality gate: WARNING — bimodal δ distribution "
            f"(median={median_delta:.6f} below {_MEDIAN_DELTA_EPSILON:g}, "
            f"max={max(deltas_pre):.4f}). Floor backstop active: "
            f"cutoff={multiple_cutoff:.4f} "
            f"(k={harvest_cfg.s2c_outlier_multiple} × floor="
            f"{_MEDIAN_DELTA_FLOOR}).",
        )

    surviving: list[FitChunk] = []
    dropped: list[dict[str, Any]] = []
    for fit, delta in zip(inputs.fits, deltas_pre):
        reasons: list[str] = []
        if delta > T_pre:
            reasons.append("exceeds_T")
        if delta > multiple_cutoff:
            reasons.append("exceeds_median_multiple")
        if reasons:
            dropped.append(
                {
                    "fit_pk": fit.pk,
                    "fit_chunk_id": fit.chunk_id,
                    "delta": delta,
                    "reasons": reasons,
                }
            )
            _emit(
                progress,
                f"Quality gate: dropping {fit.chunk_id} "
                f"(δ={delta:.4f}, T={T_pre:.4f}, "
                f"{harvest_cfg.s2c_outlier_multiple}× median="
                f"{multiple_cutoff:.4f}) — reasons: {','.join(reasons)}",
            )
        else:
            surviving.append(fit)

    # Structural abort: LOO needs ``rest >= 2``, so the survivor cohort
    # must have ≥ 3 FITs for the gate to even reach LOO. Dropping below
    # that is catastrophic — abort with the same hard-fail path as a
    # LOO FAILED verdict.
    if len(surviving) < _GATE_SURVIVOR_FLOOR:
        raise AnchorValidationError(
            f"Quality gate dropped {len(dropped)}/{len(inputs.fits)} "
            f"FIT(s); {len(surviving)} survivors below structural "
            f"floor {_GATE_SURVIVOR_FLOOR}. No output written."
        )

    # Soft warning: survivors below ``harvest.min_fit`` (the Phase 1
    # convergence knob) typically means upstream produced a borderline
    # session. Phase 2 still proceeds.
    if dropped and len(surviving) < harvest_cfg.min_fit:
        _emit(
            progress,
            f"Quality gate: WARNING — {len(surviving)} survivors below "
            f"harvest.min_fit={harvest_cfg.min_fit}; Phase 2 proceeds but "
            "the cohort is thin.",
        )

    if not dropped:
        return (
            inputs,
            calibration,
            [],
            median_delta,
            T_pre,
            multiplier_cutoff,
            median_floor_applied,
        )

    _emit(
        progress,
        f"Quality gate: dropped {len(dropped)}/{len(inputs.fits)} "
        f"FIT(s); proceeding with {len(surviving)}.",
    )
    new_inputs = dataclasses.replace(inputs, fits=surviving)
    new_calibration = derive_threshold_prime(
        [f.span_embedding for f in surviving],
        [f.chunk_embedding for f in surviving],
    )
    return (
        new_inputs,
        new_calibration,
        dropped,
        median_delta,
        T_pre,
        multiplier_cutoff,
        median_floor_applied,
    )


def run_anchor(
    session_target: str | Path,
    *,
    runs_dir: Path | None = None,
    config: SearchConfig | None = None,
    milvus_client: MilvusClient | None = None,
    batch_size: int | None = None,
    max_k: int | None = None,
    radius_scheme: RadiusScheme | None = None,
    dry_run: bool = False,
    allow_unconverged: bool = False,
    progress: ProgressCallback | None = None,
    harvest_overrides: dict[str, Any] | None = None,
) -> AnchorResult:
    """Run the Phase 2 pass on ``session_target`` (session id or path).

    ``batch_size`` / ``max_k`` / ``radius_scheme`` = ``None`` means
    "resolve from the ``harvest`` section of config.yaml". Explicit
    values (set by the CLI or programmatic callers) win.

    Phase 2 requires a Phase 1 session that satisfies the dual
    convergence gate. The check fires inside ``load_anchor_inputs``;
    pass ``allow_unconverged=True`` to bypass it (the CLI surfaces
    this as ``--allow-unconverged``).

    Raises :class:`AnchorError` subclasses on any abort. Always runs
    LOO before the main pass so a FAILED verdict can short-circuit.
    Budget exhaustion on the main pass is non-fatal — the partial
    sidecar is written and
    ``retrieval.per_fit_pages[i].budget_exhausted`` flags the affected
    FITs for the caller to surface.
    """
    t_total_start = time.perf_counter()
    runs_dir_path = Path(runs_dir) if runs_dir is not None else Path("runs")

    if config is None:
        config = load_default_config()

    harvest_cfg = load_harvest_config(session_overrides=harvest_overrides)
    if batch_size is None:
        batch_size = harvest_cfg.batch_size
    if max_k is None:
        max_k = harvest_cfg.max_k
    if radius_scheme is None:
        radius_scheme = harvest_cfg.radius_scheme
    if max_k < batch_size:
        raise AnchorError(
            f"max_k ({max_k}) must be >= batch_size ({batch_size}); "
            "raise harvest.max_k in config.yaml or lower --batch-size."
        )

    owns_client = False
    client = milvus_client
    if client is None:
        try:
            client = MilvusClient(uri=config.milvus_uri)
        except MilvusException as exc:
            raise AnchorError(
                f"Failed to connect to Milvus at {config.milvus_uri}: {exc}"
            ) from exc
        owns_client = True

    try:
        _emit(progress, "Loading FITs from session...")
        t0 = time.perf_counter()
        inputs = load_anchor_inputs(
            session_target,
            runs_dir=runs_dir_path,
            config=config,
            milvus_client=client,
            allow_unconverged=allow_unconverged,
            harvest_overrides=harvest_overrides,
        )
        load_ms = (time.perf_counter() - t0) * 1000

        _emit(progress, f"Calibrating T' over {len(inputs.fits)} FITs...")
        t0 = time.perf_counter()
        calibration = derive_threshold_prime(
            [f.span_embedding for f in inputs.fits],
            [f.chunk_embedding for f in inputs.fits],
        )
        calibrate_ms = (time.perf_counter() - t0) * 1000

        (
            inputs,
            calibration,
            dropped_records,
            median_delta_pre,
            T_pre,
            multiplier_cutoff,
            median_floor_applied,
        ) = _apply_quality_gate(
            inputs=inputs,
            calibration=calibration,
            harvest_cfg=harvest_cfg,
            progress=progress,
        )

        # Anchor-frequency gate feasibility (issue #22). Run after the
        # quality gate so ``n_fit_after_qg`` reflects the cohort the
        # main pass will actually use. We hard-fail rather than silently
        # clamp: a misconfigured ``f`` either produces an empty cohort
        # or causes the operator to misread the output as gated when
        # it wasn't.
        f_config = harvest_cfg.anchor_frequency_gate
        n_fit_after_qg = len(inputs.fits)
        if f_config > n_fit_after_qg:
            raise AnchorValidationError(
                f"harvest.anchor_frequency_gate ({f_config}) exceeds "
                f"surviving FIT cohort ({n_fit_after_qg}) after the anchor-"
                f"quality gate. Lower harvest.anchor_frequency_gate or "
                "investigate why so many FITs were dropped. No output written."
            )

        _emit(
            progress,
            f"── Stage 1/2: LOO recovery ({len(inputs.fits)} FITs) ──",
        )
        t0 = time.perf_counter()
        recovery = loo_fit_recovery(
            inputs.fits,
            collection=inputs.collection,
            client=client,
            max_k=max_k,
            progress=progress,
        )
        loo_ms = (time.perf_counter() - t0) * 1000

        if recovery.verdict == "FAILED":
            missed_pks = [p.fit_pk for p in recovery.missed_fits]
            raise AnchorValidationError(
                f"LOO recovery FAILED: {recovery.recovered}/{recovery.total} "
                f"FITs recovered; missed {missed_pks}. "
                "No output written."
            )

        # Branch main-pass radii by scheme. LOO above always used per-FIT
        # T'_i; the main pass either inherits that (``per_fit``) or uses
        # a single session-wide cap T'_out = T + min(δ) (``decoupled``,
        # issue #20).
        if radius_scheme is RadiusScheme.DECOUPLED:
            main_T_primes = [calibration.T_prime_out] * len(inputs.fits)
            _emit(
                progress,
                f"Radius scheme: decoupled — main pass uses "
                f"T'_out={calibration.T_prime_out:.4f} "
                f"(T={calibration.T:.4f} + min(δ)="
                f"{calibration.T_prime_out - calibration.T:.4f}).",
            )
        else:
            main_T_primes = list(calibration.T_primes)
            _emit(
                progress,
                "Radius scheme: per_fit — main pass uses per-FIT T'_i.",
            )

        if dry_run:
            # Dry-run validates calibration + LOO only. The main pass
            # and write are the expensive / irreversible steps, so we
            # skip both and return an empty RetrievalResult.
            _emit(progress, "Dry-run — skipping main retrieve pass.")
            retrieval = RetrievalResult(
                candidates=[],
                n_raw_hits=0,
                n_unique=0,
                batch_size=batch_size,
                max_k=max_k,
                per_fit_pages=[],
            )
            retrieve_ms = 0.0
        else:
            _emit(
                progress,
                f"── Stage 2/2: Main retrieve (batch_size={batch_size}, "
                f"max_k={max_k}) ──",
            )
            t0 = time.perf_counter()
            retrieval = retrieve_anchored(
                inputs.fits,
                T_primes=main_T_primes,
                collection=inputs.collection,
                client=client,
                batch_size=batch_size,
                max_k=max_k,
                progress=progress,
            )
            retrieve_ms = (time.perf_counter() - t0) * 1000

        # Apply the anchor-frequency gate as a post-filter on the union
        # the per-FIT iterators returned. Under f=1 this is a no-op
        # (every retained chunk satisfies ≥1 anchor by construction).
        # Under f≥2 we drop chunks below the agreement threshold and
        # rewrite ``retrieval.candidates`` so writers / downstream
        # consumers see the gated cohort.
        pre_gate_count = len(retrieval.candidates)
        if f_config <= 1:
            gated_candidates = list(retrieval.candidates)
        else:
            gated_candidates = [
                c for c in retrieval.candidates
                if c.qualifying_fit_count >= f_config
            ]
        kept = len(gated_candidates)
        dropped = pre_gate_count - kept
        if dropped:
            _emit(
                progress,
                f"Anchor-frequency gate (f={f_config}): dropped {dropped} "
                f"of {pre_gate_count} candidates; {kept} retained.",
            )

        if gated_candidates:
            qcounts = sorted(c.qualifying_fit_count for c in gated_candidates)
            mid = len(qcounts) // 2
            if len(qcounts) % 2 == 0:
                median_qcount = (qcounts[mid - 1] + qcounts[mid]) // 2
            else:
                median_qcount = qcounts[mid]
            qualifying_distribution = {
                "min": qcounts[0],
                "median": median_qcount,
                "max": qcounts[-1],
            }
            qualifying_histogram: dict[int, int] = {}
            for q in qcounts:
                qualifying_histogram[q] = qualifying_histogram.get(q, 0) + 1
        else:
            qualifying_distribution = {"min": 0, "median": 0, "max": 0}
            qualifying_histogram = {}

        frequency_gate = FrequencyGateSummary(
            f_configured=f_config,
            n_fit_after_quality_gate=n_fit_after_qg,
            kept=kept,
            dropped=dropped,
            qualifying_count_distribution=qualifying_distribution,
            qualifying_count_histogram=qualifying_histogram,
        )

        # Drop any DISCARD pks the operator invalidated in Phase 1
        # (issue #46) — they must not reach Phase 3 even if Milvus
        # surfaced them as similar-to-FIT.
        n_discard_dropped = 0
        if inputs.discard_pks:
            pre_discard = len(gated_candidates)
            gated_candidates = [
                c for c in gated_candidates if c.pk not in inputs.discard_pks
            ]
            n_discard_dropped = pre_discard - len(gated_candidates)
            if n_discard_dropped:
                _emit(
                    progress,
                    f"Filtered {n_discard_dropped} DISCARD pk(s) from "
                    "harvest output.",
                )

        # Replace the retrieval result with the gated candidate list
        # so writer / downstream consumers see the post-filter cohort.
        retrieval = dataclasses.replace(retrieval, candidates=gated_candidates)

        intrusions = count_not_fit_intrusions(
            [c.pk for c in retrieval.candidates],
            inputs.not_fit_pks,
        )

        # Cohort consistency: each surviving FIT's own chunk should be
        # admitted by ≥ f distinct anchors (the same gate applied to
        # the candidate cohort). Under f=1 this collapses to "passed
        # some FIT's T' filter", matching pre-#22 semantics. We use
        # ``qualifying_count_by_pk`` (which includes FIT-own pks)
        # rather than the post-FIT-exclusion ``candidates`` list so a
        # FIT whose own chunk would have passed the gate still reads
        # as ``own_chunk_retained=true``. Warning only, never aborts.
        cohort_consistency = [
            {
                "fit_pk": fit.pk,
                "fit_chunk_id": fit.chunk_id,
                "own_chunk_retained": (
                    retrieval.qualifying_count_by_pk.get(fit.pk, 0)
                    >= f_config
                ),
            }
            for fit in inputs.fits
        ]
        if not dry_run:
            missing = [
                r for r in cohort_consistency if not r["own_chunk_retained"]
            ]
            if missing:
                _emit(
                    progress,
                    f"Cohort consistency: {len(missing)}/{len(inputs.fits)} "
                    f"FIT(s) own chunk absent from output — "
                    f"{[r['fit_chunk_id'] for r in missing]}",
                )

        write_result: WriteResult | None = None
        if not dry_run:
            _emit(progress, "Writing phase2 sidecars...")
            timings_partial = AnchorTimings(
                load_ms=load_ms,
                calibrate_ms=calibrate_ms,
                loo_ms=loo_ms,
                retrieve_ms=retrieve_ms,
                total_ms=(time.perf_counter() - t_total_start) * 1000,
            )
            write_result = write_anchor_outputs(
                inputs=inputs,
                calibration=calibration,
                recovery=recovery,
                retrieval=retrieval,
                not_fit_intrusions=intrusions,
                runs_dir=runs_dir_path,
                timings=timings_partial,
                radius_scheme=radius_scheme,
                cohort_consistency=cohort_consistency,
                quality_gate_dropped=dropped_records,
                quality_gate_median_delta_pre_drop=median_delta_pre,
                quality_gate_T_pre_drop=T_pre,
                quality_gate_multiplier_cutoff=multiplier_cutoff,
                quality_gate_median_floor_applied=median_floor_applied,
                quality_gate_median_floor=_MEDIAN_DELTA_FLOOR,
                s2c_outlier_multiple=harvest_cfg.s2c_outlier_multiple,
                n_discard_filtered=n_discard_dropped,
                frequency_gate=frequency_gate,
            )

        total_ms = (time.perf_counter() - t_total_start) * 1000

        return AnchorResult(
            inputs=inputs,
            calibration=calibration,
            recovery=recovery,
            retrieval=retrieval,
            write=write_result,
            not_fit_intrusions=intrusions,
            timings=AnchorTimings(
                load_ms=load_ms,
                calibrate_ms=calibrate_ms,
                loo_ms=loo_ms,
                retrieve_ms=retrieve_ms,
                total_ms=total_ms,
            ),
            radius_scheme=radius_scheme,
            cohort_consistency=cohort_consistency,
            quality_gate_dropped=dropped_records,
            quality_gate_median_delta_pre_drop=median_delta_pre,
            quality_gate_T_pre_drop=T_pre,
            quality_gate_multiplier=harvest_cfg.s2c_outlier_multiple,
            quality_gate_multiplier_cutoff=multiplier_cutoff,
            quality_gate_median_floor_applied=median_floor_applied,
            n_discard_filtered=n_discard_dropped,
            frequency_gate=frequency_gate,
        )
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("MilvusClient.close() raised: %s", exc)
