"""Tests for the JudgeVerdict factory and RubricMetadata model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.refine.schema import (
    RubricCheck,
    RubricFitExample,
    RubricMetadata,
    RubricNotFitExample,
    make_judge_verdict_model,
)


def _make_model(*, allowed=("a", "b"), line_count=5):
    return make_judge_verdict_model(
        allowed_checks=frozenset(allowed),
        chunk_line_count=line_count,
    )


def test_keep_with_indices_and_null_failed_check():
    M = _make_model()
    v = M.model_validate(
        {
            "verdict": "KEEP",
            "evidence_line_indices": [1, 2, 3],
            "failed_check": None,
            "reason": "good",
        }
    )
    assert v.verdict == "KEEP"
    assert v.evidence_line_indices == [1, 2, 3]


def test_drop_requires_failed_check_in_enum():
    M = _make_model(allowed=("speaker",))
    M.model_validate(
        {
            "verdict": "DROP",
            "evidence_line_indices": [1],
            "failed_check": "speaker",
            "reason": "wrong speaker",
        }
    )
    with pytest.raises(ValidationError):
        M.model_validate(
            {
                "verdict": "DROP",
                "evidence_line_indices": [1],
                "failed_check": "made_up",
                "reason": "x",
            }
        )


def test_keep_rejects_failed_check():
    M = _make_model()
    with pytest.raises(ValidationError):
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [1],
                "failed_check": "a",
                "reason": "",
            }
        )


def test_drop_requires_failed_check_present():
    M = _make_model()
    with pytest.raises(ValidationError):
        M.model_validate(
            {
                "verdict": "DROP",
                "evidence_line_indices": [1],
                "failed_check": None,
                "reason": "",
            }
        )


def test_indices_must_be_ascending_unique_within_range():
    M = _make_model(line_count=5)
    with pytest.raises(ValidationError):
        # not ascending
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [3, 2],
                "failed_check": None,
                "reason": "",
            }
        )
    with pytest.raises(ValidationError):
        # duplicate
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [1, 1],
                "failed_check": None,
                "reason": "",
            }
        )
    with pytest.raises(ValidationError):
        # out of range
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [6],
                "failed_check": None,
                "reason": "",
            }
        )


def test_indices_length_bounds():
    M = _make_model()
    with pytest.raises(ValidationError):
        # empty
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [],
                "failed_check": None,
                "reason": "",
            }
        )
    with pytest.raises(ValidationError):
        # too long
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [1, 2, 3, 4],
                "failed_check": None,
                "reason": "",
            }
        )


def test_unknown_top_level_field_rejected():
    M = _make_model()
    with pytest.raises(ValidationError):
        M.model_validate(
            {
                "verdict": "KEEP",
                "evidence_line_indices": [1],
                "failed_check": None,
                "reason": "",
                "extra": "stray",
            }
        )


def test_factory_rejects_zero_line_count():
    with pytest.raises(ValueError):
        make_judge_verdict_model(allowed_checks=frozenset({"a"}), chunk_line_count=0)


def test_rubric_metadata_dedup_check_ids():
    with pytest.raises(ValidationError):
        RubricMetadata(
            query="q",
            source_session_id="s",
            derive_model_id="m",
            meta_prompt_path="p",
            meta_prompt_sha256="a" * 64,
            checks=[
                RubricCheck(id="x", description="d"),
                RubricCheck(id="x", description="d2"),
            ],
            fit_examples=[RubricFitExample(pk=1, span_text="hi")],
            not_fit_examples=[],
            prompt_path="r",
            prompt_sha256="b" * 64,
            version=1,
        )


def test_rubric_metadata_not_fit_must_reference_declared_check():
    with pytest.raises(ValidationError):
        RubricMetadata(
            query="q",
            source_session_id="s",
            derive_model_id="m",
            meta_prompt_path="p",
            meta_prompt_sha256="a" * 64,
            checks=[RubricCheck(id="x", description="d")],
            fit_examples=[RubricFitExample(pk=1, span_text="hi")],
            not_fit_examples=[
                RubricNotFitExample(pk=2, span_text="bye", fails=["nonexistent"])
            ],
            prompt_path="r",
            prompt_sha256="b" * 64,
            version=1,
        )
