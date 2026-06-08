"""Tests for extraction prompt assembly."""

from __future__ import annotations

from src.extraction.prompt import (
    build_messages,
    load_extraction_prompts,
)


class TestLoadExtractionPrompts:
    def test_system_block_nonempty(self) -> None:
        system, user = load_extraction_prompts()
        assert "strict JSON" in system
        assert "Maximum span length is 3 lines" in system

    def test_user_template_has_slots(self) -> None:
        _, user = load_extraction_prompts()
        assert "{query}" in user
        assert "{prior_fit_spans_block}" in user
        assert "{numbered_chunk}" in user


class TestBuildMessages:
    def test_zero_shot(self) -> None:
        messages = build_messages(
            "家长觉得课程有点难",
            [],
            "老师：hi\n家长：hello",
        )
        assert [m["role"] for m in messages] == ["system", "user"]
        user = messages[1]["content"]
        assert "家长觉得课程有点难" in user
        assert "[0] 老师：hi" in user
        assert "[1] 家长：hello" in user
        # No prior-spans block when empty
        assert "Prior FIT spans" not in user

    def test_few_shot_includes_prior_spans(self) -> None:
        messages = build_messages(
            "query",
            ["老师：ref span 1", "家长：ref span 2"],
            "老师：a\n家长：b",
        )
        user = messages[1]["content"]
        assert "Prior FIT spans" in user
        assert "ref span 1" in user
        assert "ref span 2" in user

    def test_prior_spans_capped_at_five(self) -> None:
        spans = [f"span {i}" for i in range(10)]
        messages = build_messages("query", spans, "a\nb")
        user = messages[1]["content"]
        # Last 5 included, first 5 dropped
        for i in range(5, 10):
            assert f"span {i}" in user
        for i in range(5):
            assert f"span {i}" not in user

    def test_line_numbering_zero_indexed(self) -> None:
        messages = build_messages("q", [], "first\nsecond\nthird")
        user = messages[1]["content"]
        assert "[0] first" in user
        assert "[1] second" in user
        assert "[2] third" in user

    def test_custom_system_block_respected(self) -> None:
        messages = build_messages(
            "q",
            [],
            "a\nb",
            system_block="CUSTOM SYSTEM",
            user_template="Q={query}\nSpans:{prior_fit_spans_block}\nChunk:\n{numbered_chunk}",
        )
        assert messages[0]["content"] == "CUSTOM SYSTEM"
        assert "Q=q" in messages[1]["content"]
        assert "[0] a" in messages[1]["content"]
