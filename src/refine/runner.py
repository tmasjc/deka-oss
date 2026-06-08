"""Phase 3 runner — three operator-step boundaries.

The TUI drives derive → editor → judge → review → finalise as
distinct user-visible stages, so the runner exposes them as
separate top-level functions rather than one monolithic
``run_refine``. The headless CLI in ``__main__`` chains them.

- :func:`run_refine_derive` — runs the meta-prompt and returns a
  ``DeriveResult``. The TUI pushes this into the editor screen
  before locking.
- :func:`run_refine_judge` — runs async judge over a locked rubric;
  wraps :func:`asyncio.run` for the sync TUI worker thread.
- :func:`finalize_refine` — operator agreed; writer persists.

All three propagate :class:`src.refine.errors.RefineError` subclasses
so the CLI / TUI can funnel them through one error surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.postgres.config import PostgresConfig

from .config import RefineConfig, load_refine_config
from .derive import DeriveResult, derive_rubric
from .errors import RefineConfigError
from .judge import ContentFetcher, JudgeResult, ProgressCallback, run_judge
from .load_session import (
    Phase3SessionInputs,
    load_phase3_session_inputs,
)
from .sample import (
    StratifiedSample,
    load_known_intruder_pks,
    load_phase2_records,
    stratified_sample,
)
from .schema import RubricMetadata
from .writer import (
    RefineTimings,
    RefineWriteResult,
    write_refine_stage_a,
    write_refine_stage_b,
)

log = logging.getLogger(__name__)


@dataclass
class RefineRunState:
    """Mutable container threaded through the three operator stages.

    The TUI holds one of these on ``SessionState.refine_state``; the
    CLI holds it across function calls. Each stage updates the
    relevant fields.
    """

    inputs: Phase3SessionInputs
    cfg: RefineConfig
    derive_result: DeriveResult | None = None
    rubric_text: str | None = None
    rubric_metadata: RubricMetadata | None = None
    sample: StratifiedSample | None = None
    judge_result: JudgeResult | None = None
    write_result: RefineWriteResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    operator_decision: str = ""

    @property
    def session_id(self) -> str:
        return self.inputs.session_id


# ---------------------------------------------------------------------------
# Stage 1: derive
# ---------------------------------------------------------------------------


def run_refine_derive(
    session_target: str | Path,
    *,
    runs_dir: Path | None = None,
    cfg: RefineConfig | None = None,
    derive_client=None,
    api_key: str | None = None,
    refine_overrides: dict[str, Any] | None = None,
) -> RefineRunState:
    """Run the meta-prompt and return a :class:`RefineRunState` carrying
    the derived rubric text + metadata.

    The caller (TUI / CLI) takes the rubric text into the editor next.
    Only after the operator locks the prompt does
    :func:`run_refine_judge` get called.

    ``refine_overrides`` lets a web caller inject per-session knobs from
    ``<sid>.overrides.json``; ignored when ``cfg`` is supplied directly.
    """
    cfg = cfg or load_refine_config(session_overrides=refine_overrides)
    if not cfg.enabled:
        raise RefineConfigError(
            "refine.enabled=false in config; Phase 3 is disabled. "
            "Flip refine.enabled=true to run."
        )

    inputs = load_phase3_session_inputs(session_target, runs_dir=runs_dir)

    started = time.perf_counter()
    derive = derive_rubric(
        inputs=inputs,
        cfg=cfg,
        client=derive_client,
        api_key=api_key,
    )
    derive_ms = (time.perf_counter() - started) * 1000.0

    state = RefineRunState(
        inputs=inputs,
        cfg=cfg,
        derive_result=derive,
        rubric_text=derive.rubric_text,
        rubric_metadata=derive.metadata,
        timings={"derive_ms": derive_ms},
    )
    log.info(
        "Phase 3 derive: session=%s checks=%s attempts=%d latency=%.1fms",
        inputs.session_id,
        [c.id for c in derive.metadata.checks],
        derive.attempts,
        derive.latency_ms,
    )
    return state


# ---------------------------------------------------------------------------
# Stage 2: judge
# ---------------------------------------------------------------------------


def run_refine_judge(
    state: RefineRunState,
    *,
    runs_dir: Path,
    fetcher: ContentFetcher,
    judge_client=None,
    api_key: str | None = None,
    progress: ProgressCallback | None = None,
) -> RefineRunState:
    """Stratified-sample the Phase 2 cohort and judge each chunk.

    Mutates ``state`` in place: sets ``sample``, ``judge_result``, and
    accumulates ``timings`` keys. Returns the same state for chaining.

    The rubric prompt locked at this point is the one the editor
    last saved (``state.rubric_text``); ``state.rubric_metadata`` is
    its parsed companion. If the operator never opened the editor,
    these are the unedited derive output.
    """
    if state.rubric_text is None or state.rubric_metadata is None:
        raise RefineConfigError(
            "run_refine_judge: state has no rubric_text/rubric_metadata; "
            "run_refine_derive must run first."
        )

    sample_started = time.perf_counter()
    records = load_phase2_records(runs_dir, state.session_id)
    intruders = load_known_intruder_pks(runs_dir, state.session_id)
    sample = stratified_sample(
        records,
        sample_size=state.cfg.sample_size,
        n_bins=state.cfg.n_bins,
        seed=state.cfg.seed,
        exclude_pks=state.inputs.rated_pks,
        known_intruder_pks=intruders,
        auto_drop_known_intruders=state.cfg.auto_drop_known_intruders,
    )
    state.sample = sample
    sample_ms = (time.perf_counter() - sample_started) * 1000.0
    state.timings["sample_ms"] = sample_ms

    judge_started = time.perf_counter()
    judge_result = asyncio.run(
        run_judge(
            sample=sample,
            rubric_text=state.rubric_text,
            rubric_metadata=state.rubric_metadata,
            cfg=state.cfg,
            fetcher=fetcher,
            api_key=api_key,
            client=judge_client,
            progress=progress,
        )
    )
    judge_ms = (time.perf_counter() - judge_started) * 1000.0
    state.judge_result = judge_result
    state.timings["judge_total_ms"] = judge_ms

    # Stage A: judge done, awaiting operator decision. Writing the
    # rubric.json + evidence.jsonl pair here — before the review
    # panel opens — is what makes "POST_RUBRIC" detectable on disk.
    # Disagree at review bumps rubric.version and re-runs derive +
    # judge; the next stage A call truncate-overwrites both files.
    write_refine_stage_a(
        session_id=state.session_id,
        runs_dir=runs_dir,
        rubric_metadata=state.rubric_metadata,
        judge_result=judge_result,
    )

    log.info(
        "Phase 3 judge: %d verdicts (parse_errors=%d api_errors=%d) in %.1fms",
        len(judge_result.verdicts),
        judge_result.parse_error_count,
        judge_result.api_error_count,
        judge_result.total_latency_ms,
    )
    return state


# ---------------------------------------------------------------------------
# Stage 3: finalize (writer)
# ---------------------------------------------------------------------------


def finalize_refine(
    state: RefineRunState,
    *,
    runs_dir: Path,
    operator_decision: str = "agree",
    meta_prompt_full_text: str | None = None,
) -> RefineRunState:
    """Persist the four sidecars and append the details block.

    ``meta_prompt_full_text`` is the on-disk meta-prompt's text at run
    time. If omitted, the writer falls back to recomputing it from
    ``state.rubric_metadata.meta_prompt_path`` — which is normally the
    same file but may have drifted between derive and finalise. Pass
    explicitly when the caller has already read the file (the TUI
    does this).
    """
    if (
        state.rubric_text is None
        or state.rubric_metadata is None
        or state.judge_result is None
        or state.sample is None
    ):
        raise RefineConfigError(
            "finalize_refine: state is missing a required stage result. "
            "Run derive → judge before finalise."
        )

    write_started = time.perf_counter()
    metadata = state.rubric_metadata
    if meta_prompt_full_text is not None:
        from src.prompt_io import prompt_sha256

        meta_sha = prompt_sha256(meta_prompt_full_text)
    else:
        meta_sha = metadata.meta_prompt_sha256

    derive_model_id = (
        state.derive_result.derive_model_id
        if state.derive_result is not None
        else metadata.derive_model_id
    )

    timings = RefineTimings(
        derive_ms=state.timings.get("derive_ms", 0.0),
        sample_ms=state.timings.get("sample_ms", 0.0),
        judge_total_ms=state.timings.get("judge_total_ms", 0.0),
        write_ms=0.0,
        total_ms=0.0,
    )

    # Stage B: operator agreed; truncate-write prompt.md + meta.json
    # and append the details block. Stage A already produced the
    # rubric.json + evidence.jsonl pair when the judge run completed.
    write_result = write_refine_stage_b(
        session_id=state.session_id,
        runs_dir=runs_dir,
        rubric_text=state.rubric_text,
        rubric_metadata=metadata,
        judge_result=state.judge_result,
        sample=state.sample,
        cfg=state.cfg,
        derive_model_id=derive_model_id,
        judge_model_id=state.cfg.judge_model,
        meta_prompt_path=metadata.meta_prompt_path,
        meta_prompt_sha256=meta_sha,
        operator_decision=operator_decision,
        timings=timings,
    )
    write_ms = (time.perf_counter() - write_started) * 1000.0
    state.timings["write_ms"] = write_ms
    state.timings["total_ms"] = sum(
        state.timings.get(k, 0.0)
        for k in ("derive_ms", "sample_ms", "judge_total_ms", "write_ms")
    )
    state.write_result = write_result
    state.operator_decision = operator_decision
    return state


# ---------------------------------------------------------------------------
# Helpers used by the CLI
# ---------------------------------------------------------------------------


def _open_default_fetcher() -> ContentFetcher:
    """Default fetcher: the production Postgres-backed one. Imported
    lazily so unit tests / CLI dry-runs don't need the dependency.
    """
    from src.postgres.config import load_postgres_config
    from src.postgres.fetch import OriginalContentFetcher

    pg_cfg: PostgresConfig = load_postgres_config()
    if not pg_cfg.enabled:
        raise RefineConfigError(
            "Phase 3 judge needs Postgres for chunk content but "
            "postgres.enabled=false in config.yaml."
        )
    return OriginalContentFetcher(pg_cfg)
