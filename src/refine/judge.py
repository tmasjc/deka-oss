"""Async per-chunk judge — applies a locked rubric prompt to each
sampled chunk and emits a strict-JSON verdict.

Distilled from the deleted ``src/labeling/pipeline.py`` (commit
``a22ff72^``): kept the async fan-out + dual-bucket rate-limiter +
retry-with-backoff core; dropped the per-batch grouping (refine
judges one chunk per call), the Postgres bulk-fetch (refine reuses
the singleton :class:`OriginalContentFetcher`), the checkpoint /
resume path (one Phase 3 turn judges ~500 chunks ≪ 1 hr; rerunning
a turn is cheaper than persisting partial state), and the Phase 2
JSONL reader (the sampler already produced :class:`SampledRecord`
instances).

The judge writes nothing — it returns :class:`JudgeResult` and the
writer persists the verdicts. Errors that exhaust the retry budget
become ``verdict="ERROR"`` rows, not exceptions, so a few
unreachable verdicts don't tank the whole run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import RefineConfig
from .errors import RefineConfigError, RefineError
from .judge_client import AsyncRateLimiter
from .sample import SampledRecord, StratifiedSample
from .schema import RubricMetadata, make_judge_verdict_model

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ContentFetcher(Protocol):
    """Subset of :class:`src.postgres.fetch.OriginalContentFetcher`
    the judge needs. Defined as a Protocol so tests can pass a dict
    or a stub without dragging Postgres into unit tests.
    """

    def fetch_original(self, pk: int | str) -> str | None: ...


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeVerdictRecord:
    """One judged (or auto-DROPped, or errored) chunk."""

    pk: int | str
    nearest_fit_distance: float
    decile: int
    chunk_content: str
    verdict: str  # KEEP / DROP / ERROR
    evidence_line_indices: list[int] | None
    failed_check: str | None
    reason: str
    latency_ms: float | None
    attempts: int
    rubric_version: int
    prompt_sha256: str


@dataclass(frozen=True)
class JudgeResult:
    """Return shape of :func:`run_judge`."""

    verdicts: list[JudgeVerdictRecord]
    parse_error_count: int
    api_error_count: int
    total_latency_ms: float


ProgressCallback = Callable[[int, int], None]
"""Called as ``progress(done, total)`` after each verdict completes
(including auto-DROPs and ERRORs). Optional."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_judge(
    *,
    sample: StratifiedSample,
    rubric_text: str,
    rubric_metadata: RubricMetadata,
    cfg: RefineConfig,
    fetcher: ContentFetcher,
    api_key: str | None = None,
    client: AsyncOpenAI | None = None,
    progress: ProgressCallback | None = None,
) -> JudgeResult:
    """Judge every record in ``sample.selected`` against the locked
    rubric prompt.

    Auto-DROP records (``sample.auto_drop``) bypass the LLM and land
    in the verdict list with ``verdict="DROP"`` and
    ``reason="auto_drop_known_intruder"``.

    The rubric prompt's text (``rubric_text``) is split into its
    ``## 系统``, ``## 上下文`` and ``## 用户消息（渲染后）`` blocks
    via the same parser the editor uses. The system + context blocks
    concatenate into one ``system`` message — byte-identical across
    every chunk in this run, so the static prefix (instructions +
    checks + query + FIT/NOT_FIT examples) sits inside DashScope's
    automatic prefix-cache region. The ``user`` message is the user
    template with its ``{numbered_chunk}`` placeholder substituted
    per call — the only varying content per request.

    The rubric prompt is **locked** — its SHA-256 is stamped on
    every verdict so re-readers can verify they're looking at output
    from a single predicate.
    """
    system_block, context_block, user_template = _split_rubric(rubric_text)
    if "{numbered_chunk}" not in user_template:
        raise RefineError(
            "Rubric prompt's user-message block missing {numbered_chunk} "
            "placeholder. Re-derive or edit the rubric before re-running judge."
        )
    system_message = system_block + "\n\n# 上下文\n\n" + context_block

    allowed_checks = rubric_metadata.allowed_check_ids
    rubric_version = rubric_metadata.version
    prompt_sha = rubric_metadata.prompt_sha256

    client = client or _build_client(cfg, api_key)
    limiter = AsyncRateLimiter(
        qps_limit=cfg.judge_qps_limit, tpm_limit=cfg.judge_tpm_limit
    )
    semaphore = asyncio.Semaphore(cfg.judge_concurrency)

    counters = _Counters()
    total = len(sample.selected) + len(sample.auto_drop)
    completed = 0
    verdicts: list[JudgeVerdictRecord] = []

    # Auto-drop rows first — they don't depend on async fan-out.
    for sampled in sample.auto_drop:
        chunk_content = fetcher.fetch_original(sampled.record.pk) or ""
        verdicts.append(
            JudgeVerdictRecord(
                pk=sampled.record.pk,
                nearest_fit_distance=sampled.record.nearest_fit_distance,
                decile=sampled.decile,
                chunk_content=chunk_content,
                verdict="DROP",
                evidence_line_indices=None,
                failed_check="auto_drop_known_intruder",
                reason="auto_drop_known_intruder",
                latency_ms=None,
                attempts=0,
                rubric_version=rubric_version,
                prompt_sha256=prompt_sha,
            )
        )
        completed += 1
        if progress:
            progress(completed, total)

    progress_lock = asyncio.Lock()

    async def judge_one(sampled: SampledRecord) -> JudgeVerdictRecord:
        nonlocal completed
        async with semaphore:
            record = await _judge_chunk(
                sampled=sampled,
                system_block=system_message,
                user_template=user_template,
                allowed_checks=allowed_checks,
                rubric_version=rubric_version,
                prompt_sha=prompt_sha,
                cfg=cfg,
                client=client,
                limiter=limiter,
                fetcher=fetcher,
                counters=counters,
            )
        async with progress_lock:
            completed += 1
            if progress:
                progress(completed, total)
        return record

    started = time.monotonic()
    judged = await asyncio.gather(
        *(judge_one(s) for s in sample.selected), return_exceptions=False
    )
    elapsed = (time.monotonic() - started) * 1000.0
    verdicts.extend(judged)

    return JudgeResult(
        verdicts=verdicts,
        parse_error_count=counters.parse_errors,
        api_error_count=counters.api_errors,
        total_latency_ms=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _Counters:
    parse_errors: int = 0
    api_errors: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def _judge_chunk(
    *,
    sampled: SampledRecord,
    system_block: str,
    user_template: str,
    allowed_checks: frozenset[str],
    rubric_version: int,
    prompt_sha: str,
    cfg: RefineConfig,
    client: AsyncOpenAI,
    limiter: AsyncRateLimiter,
    fetcher: ContentFetcher,
    counters: _Counters,
) -> JudgeVerdictRecord:
    chunk_content = fetcher.fetch_original(sampled.record.pk) or ""
    if not chunk_content:
        log.warning(
            "Judge: empty chunk_content for pk=%s — recording ERROR",
            sampled.record.pk,
        )
        async with counters.lock:
            counters.api_errors += 1
        return JudgeVerdictRecord(
            pk=sampled.record.pk,
            nearest_fit_distance=sampled.record.nearest_fit_distance,
            decile=sampled.decile,
            chunk_content="",
            verdict="ERROR",
            evidence_line_indices=None,
            failed_check=None,
            reason="empty_chunk_content",
            latency_ms=None,
            attempts=0,
            rubric_version=rubric_version,
            prompt_sha256=prompt_sha,
        )

    numbered = _number_lines(chunk_content)
    line_count = chunk_content.count("\n") + 1
    user_message = user_template.replace("{numbered_chunk}", numbered)

    verdict_model = make_judge_verdict_model(
        allowed_checks=allowed_checks, chunk_line_count=line_count
    )
    estimated = int(len(numbered) / 1.5 * 1.3) + 400

    last_error: str = ""
    attempts = 0
    total_latency_ms = 0.0
    for attempt in range(cfg.judge_max_retries + 1):
        attempts = attempt + 1
        await limiter.acquire(estimated)
        call_start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=cfg.judge_model,
                    messages=[
                        {"role": "system", "content": system_block},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                ),
                timeout=cfg.judge_timeout_seconds,
            )
        except Exception as exc:
            total_latency_ms += (time.monotonic() - call_start) * 1000.0
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg.judge_max_retries:
                wait = (2**attempt) * 1.0
                await asyncio.sleep(wait)
                continue
            async with counters.lock:
                counters.api_errors += 1
            return _error_record(
                sampled=sampled,
                chunk_content=chunk_content,
                rubric_version=rubric_version,
                prompt_sha=prompt_sha,
                latency_ms=total_latency_ms,
                attempts=attempts,
                reason=f"api_error: {last_error[:200]}",
            )
        total_latency_ms += (time.monotonic() - call_start) * 1000.0

        content = response.choices[0].message.content or ""
        try:
            parsed = _parse_verdict(content, verdict_model)
        except _ParseFailure as exc:
            last_error = str(exc)
            if attempt < cfg.judge_max_retries:
                continue
            async with counters.lock:
                counters.parse_errors += 1
            return _error_record(
                sampled=sampled,
                chunk_content=chunk_content,
                rubric_version=rubric_version,
                prompt_sha=prompt_sha,
                latency_ms=total_latency_ms,
                attempts=attempts,
                reason=f"schema_validation_failed: {last_error[:200]}",
            )

        return JudgeVerdictRecord(
            pk=sampled.record.pk,
            nearest_fit_distance=sampled.record.nearest_fit_distance,
            decile=sampled.decile,
            chunk_content=chunk_content,
            verdict=parsed.verdict,  # type: ignore[attr-defined]
            evidence_line_indices=list(parsed.evidence_line_indices),  # type: ignore[attr-defined]
            failed_check=parsed.failed_check,  # type: ignore[attr-defined]
            reason=parsed.reason,  # type: ignore[attr-defined]
            latency_ms=round(total_latency_ms, 2),
            attempts=attempts,
            rubric_version=rubric_version,
            prompt_sha256=prompt_sha,
        )

    # Loop exits cleanly only via return; this is unreachable but keeps
    # mypy/ pyright honest.
    return _error_record(
        sampled=sampled,
        chunk_content=chunk_content,
        rubric_version=rubric_version,
        prompt_sha=prompt_sha,
        latency_ms=total_latency_ms,
        attempts=attempts,
        reason="judge_loop_exited_without_result",
    )


def _error_record(
    *,
    sampled: SampledRecord,
    chunk_content: str,
    rubric_version: int,
    prompt_sha: str,
    latency_ms: float,
    attempts: int,
    reason: str,
) -> JudgeVerdictRecord:
    return JudgeVerdictRecord(
        pk=sampled.record.pk,
        nearest_fit_distance=sampled.record.nearest_fit_distance,
        decile=sampled.decile,
        chunk_content=chunk_content,
        verdict="ERROR",
        evidence_line_indices=None,
        failed_check=None,
        reason=reason,
        latency_ms=round(latency_ms, 2),
        attempts=attempts,
        rubric_version=rubric_version,
        prompt_sha256=prompt_sha,
    )


class _ParseFailure(Exception):
    """Raised internally when a judge response can't be coerced into
    the verdict model. Caller decides whether to retry or record
    ERROR.
    """


def _parse_verdict(raw: str, model: type[BaseModel]) -> BaseModel:
    cleaned = _THINK_RE.sub("", raw).strip()
    if not cleaned:
        raise _ParseFailure("empty content")
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise _ParseFailure(f"no JSON object in response: {cleaned[:120]!r}")
        cleaned = cleaned[start : end + 1]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise _ParseFailure(f"invalid JSON: {exc}") from exc
    if isinstance(payload, dict):
        _salvage_evidence_indices(payload)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise _ParseFailure(str(exc)) from exc


def _salvage_evidence_indices(payload: dict) -> None:
    """Truncate over-long ``evidence_line_indices`` in place.

    Small judge models (Qwen3.5-9B-AWQ at low precision in particular)
    sometimes emit 4+ indices despite the rubric's explicit 1-3
    instruction. The verdict and reason are usually still valid, so
    we keep the first three ascending-unique ints rather than fail
    the whole row to ERROR. Length-correct lists are passed through
    untouched and reach the schema validator as before.
    """
    indices = payload.get("evidence_line_indices")
    if not isinstance(indices, list) or len(indices) <= 3:
        return
    seen: set[int] = set()
    kept: list[int] = []
    for v in indices:
        if isinstance(v, int) and not isinstance(v, bool) and v not in seen:
            kept.append(v)
            seen.add(v)
            if len(kept) >= 3:
                break
    payload["evidence_line_indices"] = sorted(kept)


def _number_lines(content: str) -> str:
    """Render the chunk with 1-based line indices the judge cites."""
    lines = content.split("\n")
    return "\n".join(f"[{i + 1}] {line}" for i, line in enumerate(lines))


def _split_rubric(text: str) -> tuple[str, str, str]:
    """Split a rubric prompt into
    ``(system_block, context_block, user_template)``.

    Delegates to :func:`src.refine.derive._split_rubric_sections` so
    the judge sees exactly what derive's parser sees — same fenced /
    bare permissiveness, same heading semantics.
    """
    from .derive import _split_rubric_sections

    try:
        return _split_rubric_sections(text)
    except Exception as exc:
        raise RefineError(f"Locked rubric prompt unparseable: {exc}") from exc


def _build_client(cfg: RefineConfig, api_key: str | None) -> AsyncOpenAI:
    # ``judge_api_key_env`` takes precedence when set so a judge endpoint
    # that needs a different bearer (or a dummy bearer for an
    # unauthenticated vLLM) doesn't reuse the derive endpoint's key.
    env_var = cfg.judge_api_key_env or cfg.api_key_env
    resolved = api_key or os.environ.get(env_var)
    if resolved is None:
        raise RefineConfigError(
            f"No API key for judge LLM: set {env_var} or pass api_key=/client="
        )
    return AsyncOpenAI(
        api_key=resolved,
        base_url=cfg.judge_base_url,
        timeout=float(cfg.judge_timeout_seconds),
    )
