"""Tests for src.prompt_io two-fence parser."""

from __future__ import annotations

import pytest

from src.prompt_io import (
    PromptParseError,
    load_fenced_blocks,
    load_named_fence_sections,
    load_two_fence_sections,
    prompt_sha256,
)


def test_load_fenced_blocks_finds_all(tmp_path):
    p = tmp_path / "x.md"
    p.write_text(
        "intro\n\n```\nblock one\n```\n\nmiddle\n\n```\nblock two\n```\n",
        encoding="utf-8",
    )
    blocks = load_fenced_blocks(p)
    assert blocks == ["block one", "block two"]


def test_load_fenced_blocks_missing_file(tmp_path):
    with pytest.raises(PromptParseError):
        load_fenced_blocks(tmp_path / "missing.md")


def test_load_fenced_blocks_no_fences(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("just some prose", encoding="utf-8")
    with pytest.raises(PromptParseError):
        load_fenced_blocks(p)


def test_load_two_fence_sections_happy(tmp_path):
    p = tmp_path / "ok.md"
    p.write_text(
        "# Title\n\n## System\n\n```\nsys body\n```\n\n"
        "## User message (rendered)\n\n```\nuser body\n```\n",
        encoding="utf-8",
    )
    sys_, user = load_two_fence_sections(p, ("System", "User message (rendered)"))
    assert sys_ == "sys body"
    assert user == "user body"


def test_load_two_fence_sections_missing_first(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text(
        "## User message (rendered)\n\n```\nuser body\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(PromptParseError):
        load_two_fence_sections(p, ("System", "User message (rendered)"))


def test_load_two_fence_sections_missing_block(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text(
        "## System\n\nno fence\n\n## User message (rendered)\n\n```\nu\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(PromptParseError):
        load_two_fence_sections(p, ("System", "User message (rendered)"))


def test_load_named_fence_sections_three(tmp_path):
    p = tmp_path / "three.md"
    p.write_text(
        "# Title\n\n"
        "## System\n\n```\nsys body\n```\n\n"
        "## Context\n\n```\nctx body\n```\n\n"
        "## User\n\n```\nuser body\n```\n",
        encoding="utf-8",
    )
    sys_, ctx, user = load_named_fence_sections(p, ("System", "Context", "User"))
    assert sys_ == "sys body"
    assert ctx == "ctx body"
    assert user == "user body"


def test_load_named_fence_sections_missing_middle(tmp_path):
    p = tmp_path / "three.md"
    p.write_text(
        "## System\n\n```\nsys\n```\n\n## User\n\n```\nuser\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(PromptParseError):
        load_named_fence_sections(p, ("System", "Context", "User"))


def test_load_named_fence_sections_out_of_order(tmp_path):
    p = tmp_path / "three.md"
    # User precedes Context in the file — out of declared order.
    p.write_text(
        "## System\n\n```\nsys\n```\n\n"
        "## User\n\n```\nuser\n```\n\n"
        "## Context\n\n```\nctx\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(PromptParseError):
        load_named_fence_sections(p, ("System", "Context", "User"))


def test_prompt_sha256_stable():
    assert prompt_sha256("hello") == prompt_sha256("hello")
    assert prompt_sha256("hello") != prompt_sha256("world")
    # Specific known value
    assert prompt_sha256("hello world").startswith("b94d27b9")
