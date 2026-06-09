"""Tests for parse_rubric_prompt + render_rubric_prompt round-trip."""

from __future__ import annotations

import pytest

from src.refine.derive import parse_rubric_prompt, render_rubric_prompt
from src.refine.errors import RefineParseError
from src.refine.schema import (
    RubricCheck,
    RubricFitExample,
    RubricMetadata,
    RubricNotFitExample,
)


def _golden_metadata() -> RubricMetadata:
    return RubricMetadata(
        query="requests for material",
        source_session_id="abcd1234",
        derive_model_id="m",
        meta_prompt_path="harness/prompts/RUBRIC_DERIVE.md",
        meta_prompt_sha256="a" * 64,
        checks=[
            RubricCheck(id="speaker", description="Spoken by parent."),
            RubricCheck(id="speech_act", description="A request, not a question."),
        ],
        fit_examples=[
            RubricFitExample(pk=1, span_text="span one"),
            RubricFitExample(pk=2, span_text="span two"),
        ],
        not_fit_examples=[
            RubricNotFitExample(pk=3, span_text="bad", fails=["speech_act"])
        ],
        prompt_path="runs/x.phase3.prompt.md",
        prompt_sha256="b" * 64,
        version=1,
    )


def _parse_kwargs(meta: RubricMetadata) -> dict:
    return {
        "source_session_id": meta.source_session_id,
        "derive_model_id": meta.derive_model_id,
        "meta_prompt_path": meta.meta_prompt_path,
        "meta_prompt_sha256": meta.meta_prompt_sha256,
        "prompt_path": meta.prompt_path,
        "version": meta.version,
    }


def test_round_trip_parses_back_to_input():
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    parsed = parse_rubric_prompt(text, **_parse_kwargs(meta))
    assert [c.id for c in parsed.checks] == [c.id for c in meta.checks]
    assert [(f.pk, f.span_text) for f in parsed.fit_examples] == [
        (f.pk, f.span_text) for f in meta.fit_examples
    ]
    assert [(n.pk, n.span_text, n.fails) for n in parsed.not_fit_examples] == [
        (n.pk, n.span_text, n.fails) for n in meta.not_fit_examples
    ]
    assert parsed.query == meta.query


def test_missing_check_block_rejected():
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    bad = text.replace(
        "<!-- check_id: speech_act -->\n- A request, not a question.\n<!-- /check -->",
        "",
    )
    with pytest.raises(RefineParseError):
        parse_rubric_prompt(bad, **_parse_kwargs(meta))


def test_unknown_check_in_not_fit_rejected():
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    bad = text.replace("fails=speech_act", "fails=invented_check")
    with pytest.raises(RefineParseError):
        parse_rubric_prompt(bad, **_parse_kwargs(meta))


def test_missing_numbered_chunk_placeholder_rejected():
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    bad = text.replace("{numbered_chunk}", "(removed)")
    with pytest.raises(RefineParseError):
        parse_rubric_prompt(bad, **_parse_kwargs(meta))


def test_missing_failed_check_enum_line_rejected():
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    bad = text.replace("## 失败检查枚举:", "## (gone):")
    with pytest.raises(RefineParseError):
        parse_rubric_prompt(bad, **_parse_kwargs(meta))


def test_empty_text_rejected():
    meta = _golden_metadata()
    with pytest.raises(RefineParseError):
        parse_rubric_prompt("", **_parse_kwargs(meta))


def test_check_id_format_enforced():
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    bad = text.replace("speech_act", "Bad-Id")
    with pytest.raises(RefineParseError):
        parse_rubric_prompt(bad, **_parse_kwargs(meta))


def test_jsoncontract_field_check():
    """If derive forgets to include the JSON contract entirely, parse fails."""
    meta = _golden_metadata()
    text = render_rubric_prompt(meta)
    # Strip the contract block
    bad = text.replace("evidence_line_indices", "REPLACED")
    with pytest.raises(RefineParseError):
        parse_rubric_prompt(bad, **_parse_kwargs(meta))
