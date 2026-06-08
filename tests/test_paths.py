"""Tests for the prompts-directory resolver."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _reload_paths():
    import src.paths

    return importlib.reload(src.paths)


@pytest.fixture(autouse=True)
def _restore_paths_module():
    """Ensure later tests see the unmodified module regardless of monkeypatching here."""
    yield
    _reload_paths()


def test_default_prompts_dir_points_at_harness_prompts(monkeypatch):
    monkeypatch.delenv("DEKA_PROMPTS_DIR", raising=False)
    paths = _reload_paths()
    assert paths.PROMPTS_DIR.name == "prompts"
    assert paths.PROMPTS_DIR.parent.name == "harness"
    # Sanity: the default location actually contains the canonical prompt files.
    for name in ("SYSTEM.md", "REFLECTION.md", "EXTRACTION.md", "RUBRIC_DERIVE.md"):
        assert (paths.PROMPTS_DIR / name).is_file(), name


def test_env_override_redirects_prompt_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(tmp_path))
    paths = _reload_paths()
    assert paths.PROMPTS_DIR == Path(str(tmp_path))
    assert paths.prompt_path("SYSTEM.md") == tmp_path / "SYSTEM.md"


def test_env_override_threads_through_reflection_loader(tmp_path, monkeypatch):
    """``load_system_prompt`` reads from the overridden directory when no
    explicit ``repo_root`` is passed. Mirrors the production call from
    ``src.reflection.agent``.
    """
    (tmp_path / "SYSTEM.md").write_text("```\nstub-system\n```\n", encoding="utf-8")
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(tmp_path))
    _reload_paths()
    import src.reflection.prompt as reflection_prompt

    importlib.reload(reflection_prompt)
    assert reflection_prompt.load_system_prompt() == "stub-system"


def test_env_override_threads_through_extraction_loader(tmp_path, monkeypatch):
    """``load_extraction_prompts`` reads from the overridden directory."""
    (tmp_path / "EXTRACTION.md").write_text(
        "```\nsystem-block\n```\n\n```\nuser-template {query}\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEKA_PROMPTS_DIR", str(tmp_path))
    _reload_paths()
    import src.extraction.prompt as extraction_prompt

    importlib.reload(extraction_prompt)
    system, user = extraction_prompt.load_extraction_prompts()
    assert system == "system-block"
    assert user == "user-template {query}"
