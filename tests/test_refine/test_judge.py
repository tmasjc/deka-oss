"""Tests for the async judge — KEEP/DROP/ERROR mix, retry, schema."""

from __future__ import annotations

import asyncio
import json


from src.refine.config import RefineConfig
from src.refine.derive import render_rubric_prompt
from src.refine.judge import run_judge
from src.refine.sample import Phase2Record, SampledRecord, StratifiedSample
from src.refine.schema import (
    RubricCheck,
    RubricFitExample,
    RubricMetadata,
    RubricNotFitExample,
)


def _meta() -> RubricMetadata:
    return RubricMetadata(
        query="q",
        source_session_id="s",
        derive_model_id="m",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        checks=[RubricCheck(id="speech_act", description="A request, not a question.")],
        fit_examples=[RubricFitExample(pk=1, span_text="x")],
        not_fit_examples=[RubricNotFitExample(pk=2, span_text="y", fails=["speech_act"])],
        prompt_path="r",
        prompt_sha256="b" * 64,
        version=1,
    )


def _cfg(*, retries: int = 1) -> RefineConfig:
    return RefineConfig(
        enabled=True, sample_size=2, n_bins=2, seed=0,
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        max_fit_examples=6, max_not_fit_examples=6,
        derive_model="d", derive_base_url="x", derive_temperature=0.2,
        judge_model="j", judge_base_url="x",
        judge_concurrency=2, judge_qps_limit=100, judge_tpm_limit=1_000_000,
        judge_timeout_seconds=30, judge_max_retries=retries,
        api_key_env="X", auto_drop_known_intruders=True,
    )


class _StubResponse:
    def __init__(self, content: str) -> None:
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        self.choices = [choice]
        self.usage = None


class _StubChat:
    def __init__(self, contents: list[str]) -> None:
        self._contents = contents
        self._idx = 0

    async def create(self, **kwargs):
        if self._idx >= len(self._contents):
            return _StubResponse(self._contents[-1])
        out = self._contents[self._idx]
        self._idx += 1
        return _StubResponse(out)


class _StubClient:
    def __init__(self, contents: list[str]) -> None:
        chat_struct = type(
            "Chat",
            (),
            {"completions": _StubChat(contents)},
        )()
        self.chat = chat_struct


class _StubFetcher:
    def __init__(self, content: str = "alpha\nbeta\ngamma"):
        self._c = content

    def fetch_original(self, pk):
        return self._c


def _basic_sample() -> StratifiedSample:
    return StratifiedSample(
        selected=[
            SampledRecord(Phase2Record(pk=10, nearest_fit_distance=0.1, raw={}), 0),
            SampledRecord(Phase2Record(pk=20, nearest_fit_distance=0.2, raw={}), 1),
        ],
        auto_drop=[],
        decile_boundaries=[0.1, 0.2],
        per_decile_count=[1, 1],
        per_decile_drawn=[1, 1],
    )


def test_basic_keep_drop_mix():
    meta = _meta()
    text = render_rubric_prompt(meta)

    keep = json.dumps(
        {"verdict": "KEEP", "evidence_line_indices": [1], "failed_check": None, "reason": "ok"}
    )
    drop = json.dumps(
        {"verdict": "DROP", "evidence_line_indices": [2], "failed_check": "speech_act", "reason": "no"}
    )

    res = asyncio.run(
        run_judge(
            sample=_basic_sample(),
            rubric_text=text,
            rubric_metadata=meta,
            cfg=_cfg(),
            fetcher=_StubFetcher(),
            client=_StubClient([keep, drop]),
        )
    )
    assert len(res.verdicts) == 2
    assert {v.verdict for v in res.verdicts} == {"KEEP", "DROP"}


def test_schema_failure_then_retry_succeeds():
    meta = _meta()
    text = render_rubric_prompt(meta)

    # First call: malformed JSON. Second: valid KEEP.
    bad = "not json at all"
    good = json.dumps(
        {"verdict": "KEEP", "evidence_line_indices": [1], "failed_check": None, "reason": "ok"}
    )

    res = asyncio.run(
        run_judge(
            sample=StratifiedSample(
                selected=[SampledRecord(Phase2Record(pk=10, nearest_fit_distance=0.1, raw={}), 0)],
                auto_drop=[],
                decile_boundaries=[0.1],
                per_decile_count=[1],
                per_decile_drawn=[1],
            ),
            rubric_text=text,
            rubric_metadata=meta,
            cfg=_cfg(retries=2),
            fetcher=_StubFetcher(),
            client=_StubClient([bad, good]),
        )
    )
    assert len(res.verdicts) == 1
    assert res.verdicts[0].verdict == "KEEP"
    assert res.verdicts[0].attempts == 2


def test_schema_failure_exhausts_retries_records_error():
    meta = _meta()
    text = render_rubric_prompt(meta)

    bad = "not json at all"
    res = asyncio.run(
        run_judge(
            sample=StratifiedSample(
                selected=[SampledRecord(Phase2Record(pk=10, nearest_fit_distance=0.1, raw={}), 0)],
                auto_drop=[],
                decile_boundaries=[0.1],
                per_decile_count=[1],
                per_decile_drawn=[1],
            ),
            rubric_text=text,
            rubric_metadata=meta,
            cfg=_cfg(retries=1),
            fetcher=_StubFetcher(),
            client=_StubClient([bad, bad]),
        )
    )
    assert res.verdicts[0].verdict == "ERROR"
    assert res.parse_error_count == 1


def test_auto_drop_paths_through():
    meta = _meta()
    text = render_rubric_prompt(meta)
    keep = json.dumps(
        {"verdict": "KEEP", "evidence_line_indices": [1], "failed_check": None, "reason": "ok"}
    )

    sample = StratifiedSample(
        selected=[SampledRecord(Phase2Record(pk=10, nearest_fit_distance=0.1, raw={}), 0)],
        auto_drop=[SampledRecord(Phase2Record(pk=99, nearest_fit_distance=0.99, raw={}), 1)],
        decile_boundaries=[0.1, 0.99],
        per_decile_count=[1, 1],
        per_decile_drawn=[1, 1],
    )
    res = asyncio.run(
        run_judge(
            sample=sample,
            rubric_text=text,
            rubric_metadata=meta,
            cfg=_cfg(),
            fetcher=_StubFetcher(),
            client=_StubClient([keep]),
        )
    )
    assert len(res.verdicts) == 2
    auto = next(v for v in res.verdicts if v.failed_check == "auto_drop_known_intruder")
    assert auto.attempts == 0
    assert auto.latency_ms is None
    assert auto.verdict == "DROP"


def test_salvages_overlong_evidence_indices(monkeypatch):
    """Judge models sometimes emit 4+ indices despite the 1-3 contract.
    The parse path keeps the verdict by truncating to the first 3
    ascending-unique ints rather than ERRORing the row."""
    from src.refine.judge import _parse_verdict
    from src.refine.schema import make_judge_verdict_model

    Model = make_judge_verdict_model(
        chunk_line_count=20,
        allowed_checks=frozenset({"speech_act"}),
    )

    # 7 indices, mixed order — should keep first 3 unique, then sort.
    raw = json.dumps({
        "verdict": "DROP",
        "evidence_line_indices": [5, 3, 2, 7, 1, 4, 9],
        "failed_check": "speech_act",
        "reason": "too long",
    })
    parsed = _parse_verdict(raw, Model)
    assert list(parsed.evidence_line_indices) == [2, 3, 5]
    assert parsed.verdict == "DROP"
    assert parsed.failed_check == "speech_act"


def test_salvages_overlong_with_duplicates(monkeypatch):
    """Duplicates within the first 3 collapse — never more than 3 kept."""
    from src.refine.judge import _parse_verdict
    from src.refine.schema import make_judge_verdict_model

    Model = make_judge_verdict_model(
        chunk_line_count=20,
        allowed_checks=frozenset({"speech_act"}),
    )
    raw = json.dumps({
        "verdict": "KEEP",
        "evidence_line_indices": [1, 1, 1, 2, 3, 5],
        "failed_check": None,
        "reason": "ok",
    })
    parsed = _parse_verdict(raw, Model)
    # First three UNIQUE ints in order are [1, 2, 3] → sorted = [1, 2, 3].
    assert list(parsed.evidence_line_indices) == [1, 2, 3]


def test_length_3_or_less_passes_through_untouched(monkeypatch):
    """Schema-compliant lists are not rewritten by the salvage hook."""
    from src.refine.judge import _parse_verdict
    from src.refine.schema import make_judge_verdict_model

    Model = make_judge_verdict_model(
        chunk_line_count=20,
        allowed_checks=frozenset({"speech_act"}),
    )
    raw = json.dumps({
        "verdict": "KEEP",
        "evidence_line_indices": [2, 4],
        "failed_check": None,
        "reason": "ok",
    })
    parsed = _parse_verdict(raw, Model)
    assert list(parsed.evidence_line_indices) == [2, 4]
