"""Test-environment defaults.

Several web-API endpoints run a *preflight* that requires an LLM API key in the
environment before they will start a derive/judge step — even though every test
monkeypatches the actual LLM call. On a clean, secret-free checkout there is no
``.env``, so those keys are unset and the refine endpoints fail preflight with
``MISSING_LLM_KEY`` (HTTP 422).

We seed inert dummy keys *only when the real ones are absent*, at import time
with ``os.environ.setdefault``. This is deliberately NOT a ``monkeypatch``
fixture: a root-conftest autouse fixture that requested ``monkeypatch`` would
pull it into the autouse setup phase and flip its teardown order relative to
``tests/test_paths.py``'s ``_restore_paths_module`` fixture — which reloads
``src.paths`` and relies on ``monkeypatch`` reverting ``DEKA_PROMPTS_DIR``
*first*. Seeding at module import keeps the fixture graph untouched.

Tests that exercise the missing-key path are unaffected: they use synthetic
env-var names, ``monkeypatch.delenv`` the var in the test body, or stub
``run_preflight``.
"""

from __future__ import annotations

import os

# API-key env vars referenced by ``config.yaml.example`` (reflection +
# extraction use OPENROUTER; refine derive/judge use DASHSCOPE). A developer's
# real key (from ``.env`` or the shell) is preserved by setdefault.
os.environ.setdefault("OPENROUTER_API_KEY", "test-dummy-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dummy-key")
