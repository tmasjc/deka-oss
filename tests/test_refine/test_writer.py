"""Tests for src.refine.writer — sidecar shape + details append."""

from __future__ import annotations

import json


from src.refine.config import RefineConfig
from src.refine.derive import render_rubric_prompt
from src.refine.judge import JudgeResult, JudgeVerdictRecord
from src.refine.sample import Phase2Record, SampledRecord, StratifiedSample
from src.refine.schema import (
    RubricCheck,
    RubricFitExample,
    RubricMetadata,
    RubricNotFitExample,
)
from src.refine.writer import (
    RefineStageAResult,
    RefineTimings,
    write_refine_outputs,
    write_refine_stage_a,
    write_refine_stage_b,
)


def _meta() -> RubricMetadata:
    return RubricMetadata(
        query="q",
        source_session_id="abcd",
        derive_model_id="m",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        checks=[RubricCheck(id="x", description="d")],
        fit_examples=[RubricFitExample(pk=1, span_text="hi")],
        not_fit_examples=[RubricNotFitExample(pk=2, span_text="bye", fails=["x"])],
        prompt_path="runs/abcd.phase3.prompt.md",
        prompt_sha256="b" * 64,
        version=2,
    )


def _cfg() -> RefineConfig:
    return RefineConfig(
        enabled=True,
        sample_size=2,
        n_bins=2,
        seed=0,
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        max_fit_examples=6,
        max_not_fit_examples=6,
        derive_model="dm",
        derive_base_url="x",
        derive_temperature=0.2,
        judge_model="jm",
        judge_base_url="x",
        judge_concurrency=4,
        judge_qps_limit=1,
        judge_tpm_limit=1,
        judge_timeout_seconds=30,
        judge_max_retries=1,
        api_key_env="X",
        auto_drop_known_intruders=True,
    )


def _verdicts() -> list[JudgeVerdictRecord]:
    return [
        JudgeVerdictRecord(
            pk=10,
            nearest_fit_distance=0.1,
            decile=0,
            chunk_content="line a\nline b",
            verdict="KEEP",
            evidence_line_indices=[1, 2],
            failed_check=None,
            reason="good",
            latency_ms=42.0,
            attempts=1,
            rubric_version=2,
            prompt_sha256="b" * 64,
        ),
        JudgeVerdictRecord(
            pk=20,
            nearest_fit_distance=0.2,
            decile=1,
            chunk_content="line c",
            verdict="DROP",
            evidence_line_indices=[1],
            failed_check="x",
            reason="bad",
            latency_ms=35.0,
            attempts=1,
            rubric_version=2,
            prompt_sha256="b" * 64,
        ),
        JudgeVerdictRecord(
            pk=30,
            nearest_fit_distance=0.3,
            decile=1,
            chunk_content="line d",
            verdict="DROP",
            evidence_line_indices=None,
            failed_check="auto_drop_known_intruder",
            reason="auto_drop_known_intruder",
            latency_ms=None,
            attempts=0,
            rubric_version=2,
            prompt_sha256="b" * 64,
        ),
    ]


def test_write_produces_four_sidecars(tmp_path):
    meta = _meta()
    text = render_rubric_prompt(meta)
    sample = StratifiedSample(
        selected=[
            SampledRecord(Phase2Record(pk=10, nearest_fit_distance=0.1, raw={}), 0)
        ],
        auto_drop=[
            SampledRecord(Phase2Record(pk=30, nearest_fit_distance=0.3, raw={}), 1)
        ],
        decile_boundaries=[0.1, 0.2, 0.3],
        per_decile_count=[1, 1],
        per_decile_drawn=[1, 1],
        excluded_pks=frozenset({99}),
    )
    judge_result = JudgeResult(
        verdicts=_verdicts(),
        parse_error_count=0,
        api_error_count=0,
        total_latency_ms=100.0,
    )

    res = write_refine_outputs(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_text=text,
        rubric_metadata=meta,
        judge_result=judge_result,
        sample=sample,
        cfg=_cfg(),
        derive_model_id="m",
        judge_model_id="jm",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        operator_decision="agree",
        timings=RefineTimings(),
    )

    assert res.prompt_path.exists()
    assert res.rubric_path.exists()
    assert res.evidence_path.exists()
    assert res.meta_path.exists()
    assert res.details_path.exists()

    # Evidence is one record per line.
    evidence_lines = res.evidence_path.read_text().strip().splitlines()
    assert len(evidence_lines) == 3
    for line in evidence_lines:
        rec = json.loads(line)
        assert "pk" in rec and "verdict" in rec
        assert rec["rubric_version"] == 2
        assert rec["prompt_sha256"] == "b" * 64

    # Meta carries verdict counts and per-decile keep rates.
    meta_obj = json.loads(res.meta_path.read_text())
    assert meta_obj["verdict_counts"] == {
        "KEEP": 1,
        "DROP": 1,
        "ERROR": 0,
        "auto_drop": 1,
    }
    assert meta_obj["failed_check_histogram"] == {"x": 1}
    assert meta_obj["rubric_version"] == 2
    assert meta_obj["operator_decision"] == "agree"
    # Decile 0: 1 KEEP, 0 DROP → 1.0; Decile 1: 0 KEEP, 1 DROP → 0.0
    # (auto_drop excluded from keep-rate denominator).
    assert meta_obj["per_decile_keep_rate"] == [1.0, 0.0]


def test_details_append(tmp_path):
    meta = _meta()
    text = render_rubric_prompt(meta)
    sample = StratifiedSample(
        selected=[],
        auto_drop=[],
        decile_boundaries=[],
        per_decile_count=[],
        per_decile_drawn=[],
        excluded_pks=frozenset(),
    )
    judge_result = JudgeResult(
        verdicts=[], parse_error_count=0, api_error_count=0, total_latency_ms=0.0
    )

    # Pre-populate details with something else
    details = tmp_path / "abcd.details.jsonl"
    details.write_text(json.dumps({"turn": "phase1"}) + "\n", encoding="utf-8")

    res = write_refine_outputs(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_text=text,
        rubric_metadata=meta,
        judge_result=judge_result,
        sample=sample,
        cfg=_cfg(),
        derive_model_id="m",
        judge_model_id="jm",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        operator_decision="agree",
        timings=RefineTimings(),
    )

    lines = res.details_path.read_text().splitlines()
    assert len(lines) == 2  # phase1 (pre-existing) + phase3
    assert json.loads(lines[0]) == {"turn": "phase1"}
    assert json.loads(lines[1])["turn"] == "phase3"


# ---------------------------------------------------------------------------
# Two-stage writer tests
# ---------------------------------------------------------------------------


def _stratified_sample() -> StratifiedSample:
    return StratifiedSample(
        selected=[
            SampledRecord(Phase2Record(pk=10, nearest_fit_distance=0.1, raw={}), 0)
        ],
        auto_drop=[
            SampledRecord(Phase2Record(pk=30, nearest_fit_distance=0.3, raw={}), 1)
        ],
        decile_boundaries=[0.1, 0.2, 0.3],
        per_decile_count=[1, 1],
        per_decile_drawn=[1, 1],
        excluded_pks=frozenset({99}),
    )


def _judge_result() -> JudgeResult:
    return JudgeResult(
        verdicts=_verdicts(),
        parse_error_count=0,
        api_error_count=0,
        total_latency_ms=100.0,
    )


def test_stage_a_writes_only_rubric_and_evidence(tmp_path):
    res = write_refine_stage_a(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_metadata=_meta(),
        judge_result=_judge_result(),
    )

    assert isinstance(res, RefineStageAResult)
    assert res.rubric_path.exists()
    assert res.evidence_path.exists()
    # Stage A must NOT touch prompt.md, meta.json, or details.jsonl.
    assert not (tmp_path / "abcd.phase3.prompt.md").exists()
    assert not (tmp_path / "abcd.phase3.meta.json").exists()
    assert not (tmp_path / "abcd.details.jsonl").exists()

    # Evidence still contains every verdict, one per line.
    evidence_lines = res.evidence_path.read_text().strip().splitlines()
    assert len(evidence_lines) == 3
    rubric_obj = json.loads(res.rubric_path.read_text())
    # rubric.json forward-references the eventual prompt.md basename.
    assert rubric_obj["prompt_path"] == "abcd.phase3.prompt.md"
    assert rubric_obj["version"] == 2


def test_stage_a_is_idempotent_on_rerun(tmp_path):
    write_refine_stage_a(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_metadata=_meta(),
        judge_result=_judge_result(),
    )
    # Re-run with a different verdict set; truncate-overwrite expected.
    new_judge = JudgeResult(
        verdicts=[_verdicts()[0]],
        parse_error_count=0,
        api_error_count=0,
        total_latency_ms=10.0,
    )
    res = write_refine_stage_a(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_metadata=_meta(),
        judge_result=new_judge,
    )

    evidence_lines = res.evidence_path.read_text().strip().splitlines()
    assert len(evidence_lines) == 1


def test_stage_b_writes_prompt_meta_and_appends_details(tmp_path):
    meta = _meta()
    text = render_rubric_prompt(meta)
    judge_result = _judge_result()

    # Stage A first so the rubric.json + evidence.jsonl exist.
    write_refine_stage_a(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_metadata=meta,
        judge_result=judge_result,
    )

    res = write_refine_stage_b(
        session_id="abcd",
        runs_dir=tmp_path,
        rubric_text=text,
        rubric_metadata=meta,
        judge_result=judge_result,
        sample=_stratified_sample(),
        cfg=_cfg(),
        derive_model_id="m",
        judge_model_id="jm",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        operator_decision="agree",
        timings=RefineTimings(),
    )

    assert res.prompt_path.exists()
    assert res.meta_path.exists()
    # Stage A's outputs are still present and unchanged.
    assert res.rubric_path.exists()
    assert res.evidence_path.exists()
    # Details now carries the phase3 block.
    assert res.details_path.exists()
    lines = res.details_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["turn"] == "phase3"

    meta_obj = json.loads(res.meta_path.read_text())
    assert meta_obj["operator_decision"] == "agree"
    assert meta_obj["rubric_version"] == 2


def test_stage_a_then_b_equals_legacy_atomic_layout(tmp_path):
    """Running stage A + stage B yields the same on-disk shape as the
    legacy atomic write_refine_outputs call.
    """
    meta = _meta()
    text = render_rubric_prompt(meta)
    sample = _stratified_sample()
    judge_result = _judge_result()

    legacy_dir = tmp_path / "legacy"
    split_dir = tmp_path / "split"
    legacy_dir.mkdir()
    split_dir.mkdir()

    common_kwargs = dict(
        session_id="abcd",
        rubric_text=text,
        rubric_metadata=meta,
        judge_result=judge_result,
        sample=sample,
        cfg=_cfg(),
        derive_model_id="m",
        judge_model_id="jm",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        operator_decision="agree",
        timings=RefineTimings(),
    )

    write_refine_outputs(runs_dir=legacy_dir, **common_kwargs)
    write_refine_stage_a(
        session_id="abcd",
        runs_dir=split_dir,
        rubric_metadata=meta,
        judge_result=judge_result,
    )
    write_refine_stage_b(runs_dir=split_dir, **common_kwargs)

    # Every sidecar exists in both layouts.
    for ext in (
        "phase3.prompt.md",
        "phase3.rubric.json",
        "phase3.evidence.jsonl",
        "phase3.meta.json",
        "details.jsonl",
    ):
        assert (legacy_dir / f"abcd.{ext}").exists()
        assert (split_dir / f"abcd.{ext}").exists()

    # rubric.json + evidence.jsonl + prompt.md content matches byte-for-byte.
    for ext in ("phase3.rubric.json", "phase3.evidence.jsonl", "phase3.prompt.md"):
        assert (legacy_dir / f"abcd.{ext}").read_text() == (
            split_dir / f"abcd.{ext}"
        ).read_text()

    # meta.json's `ts` field will differ (different write timestamps);
    # everything else must match.
    legacy_meta = json.loads((legacy_dir / "abcd.phase3.meta.json").read_text())
    split_meta = json.loads((split_dir / "abcd.phase3.meta.json").read_text())
    legacy_meta.pop("ts", None)
    split_meta.pop("ts", None)
    assert legacy_meta == split_meta

    # details.jsonl: same content modulo the embedded phase3.ts.
    legacy_details = [
        json.loads(line)
        for line in (legacy_dir / "abcd.details.jsonl").read_text().strip().splitlines()
    ]
    split_details = [
        json.loads(line)
        for line in (split_dir / "abcd.details.jsonl").read_text().strip().splitlines()
    ]
    for entry in (*legacy_details, *split_details):
        if entry.get("turn") == "phase3":
            entry["phase3"].pop("ts", None)
    assert legacy_details == split_details
