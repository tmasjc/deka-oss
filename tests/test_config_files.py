"""Smoke tests: every committed config file loads without error.

Protects against a bad config landing on ``main``. Each loader is
strict, so a missing or mistyped key in any of these YAMLs would surface
here rather than at first user start-up.
"""

from __future__ import annotations

from src.anchor.config import load_harvest_config
from src.extraction.extractor import _load_config as load_extraction_config
from src.postgres.config import load_postgres_config
from src.reflection.agent import _load_config as load_reflection_config
from src.search.config import load_default_config


def test_defaults_yaml_loads() -> None:
    cfg = load_default_config()
    assert cfg.rrf_k > 0
    assert cfg.per_path_limit > 0
    assert cfg.top_k > 0
    assert cfg.min_survivors > 0
    assert cfg.min_survivors <= cfg.top_k
    assert cfg.http_timeout > 0
    assert cfg.embed_url.startswith("http")
    assert cfg.milvus_uri.startswith("http")
    # ``collection`` is intentionally blank in the loaded config — it
    # is resolved per-session from the chosen scope's
    # ``milvus_collection`` (see ``scopes.yaml``).
    assert cfg.collection == ""


def test_extraction_yaml_loads() -> None:
    cfg = load_extraction_config()
    assert cfg.model
    assert cfg.base_url.startswith("http")
    assert cfg.prompt_version
    assert cfg.api_key_env
    assert cfg.cache_root.is_absolute()


def test_reflection_yaml_loads() -> None:
    cfg = load_reflection_config()
    assert cfg.model
    assert cfg.base_url.startswith("http")
    assert 0.0 <= cfg.temperature <= 2.0
    assert cfg.api_key_env


def test_harvest_yaml_loads() -> None:
    cfg = load_harvest_config()
    assert cfg.min_fit > 0
    assert 0.0 <= cfg.precision_at_k <= 1.0
    assert cfg.batch_size > 0
    assert cfg.max_k >= cfg.batch_size


def test_postgres_yaml_loads() -> None:
    cfg = load_postgres_config()
    if not cfg.enabled:
        # Admin has disabled context expansion — connection fields
        # intentionally unvalidated in that mode.
        return
    assert cfg.dsn
    # ``table`` is intentionally blank in the loaded config — it is
    # resolved per-session from the chosen scope's
    # ``postgres_table`` (see ``scopes.yaml``).
    assert cfg.table == ""
    assert cfg.id_column
    assert cfg.content_column
    assert cfg.connect_timeout > 0
