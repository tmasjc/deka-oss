"""Resolve on-disk asset locations.

The repo layout assumes ``harness/prompts/*.md`` lives at the project
root, and prompt loaders compute their path with
``Path(__file__).resolve().parents[2] / "harness" / "prompts" / ...``.
That assumption breaks under bundle-mode packaging
(PyInstaller ``--onefile``, Nuitka ``--onefile``) and any deploy where
prompts ship next to the binary rather than inside it. The
``DEKA_PROMPTS_DIR`` environment variable overrides the lookup so a
packaged build can point loaders at an external directory without code
changes.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPTS_DIR: Path = Path(
    os.environ.get("DEKA_PROMPTS_DIR", _REPO_ROOT / "harness" / "prompts")
)


def prompt_path(name: str) -> Path:
    """Resolve a prompt filename to an absolute path under :data:`PROMPTS_DIR`."""
    return PROMPTS_DIR / name


def resolve_prompts_dir() -> Path:
    """Return the prompts directory, re-reading ``DEKA_PROMPTS_DIR`` each
    call so live re-deploys and monkeypatched tests both see the current
    value. Falls back to ``<repo>/harness/prompts`` when the env var is
    unset.
    """
    raw = os.environ.get("DEKA_PROMPTS_DIR")
    return Path(raw) if raw else (_REPO_ROOT / "harness" / "prompts")
