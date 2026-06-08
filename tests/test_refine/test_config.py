"""Tests for src.refine.config — strict required-keys + value validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.refine.config import load_refine_config
from src.refine.errors import RefineConfigError


_GOOD = {
    "enabled": True,
    "sample_size": 500,
    "n_bins": 10,
    "seed": 0,
    "meta_prompt_path": "harness/prompts/RUBRIC_DERIVE.md",
    "max_fit_examples": 6,
    "max_not_fit_examples": 6,
    "derive_model": "openai/gpt-x",
    "derive_base_url": "https://example.com/v1",
    "derive_temperature": 0.2,
    "judge_model": "openai/gpt-y",
    "judge_base_url": "https://example.com/v1",
    "judge_concurrency": 16,
    "judge_qps_limit": 8.0,
    "judge_tpm_limit": 200000.0,
    "judge_timeout_seconds": 30,
    "judge_max_retries": 2,
    "api_key_env": "OPENROUTER_API_KEY",
    "auto_drop_known_intruders": True,
}


def _write(tmp_path: Path, refine: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"refine": refine}), encoding="utf-8")
    return p


def test_happy_path(tmp_path):
    cfg = load_refine_config(_write(tmp_path, _GOOD))
    assert cfg.sample_size == 500
    assert cfg.n_bins == 10
    assert cfg.per_decile_sample == 50
    assert cfg.max_fit_examples == 6
    assert cfg.max_not_fit_examples == 6


def test_max_fit_examples_required(tmp_path):
    bad = dict(_GOOD)
    bad.pop("max_fit_examples")
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, bad))
    assert "max_fit_examples" in str(ei.value)


def test_max_not_fit_examples_required(tmp_path):
    bad = dict(_GOOD)
    bad.pop("max_not_fit_examples")
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, bad))
    assert "max_not_fit_examples" in str(ei.value)


def test_max_fit_examples_must_be_positive(tmp_path):
    for bad_val in (0, -1):
        bad = dict(_GOOD, max_fit_examples=bad_val)
        with pytest.raises(RefineConfigError) as ei:
            load_refine_config(_write(tmp_path, bad))
        assert "max_fit_examples" in str(ei.value)


def test_max_not_fit_examples_must_be_positive(tmp_path):
    for bad_val in (0, -1):
        bad = dict(_GOOD, max_not_fit_examples=bad_val)
        with pytest.raises(RefineConfigError) as ei:
            load_refine_config(_write(tmp_path, bad))
        assert "max_not_fit_examples" in str(ei.value)


def test_missing_key_rejected(tmp_path):
    bad = dict(_GOOD)
    bad.pop("seed")
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, bad))
    assert "seed" in str(ei.value)


def test_unknown_key_rejected(tmp_path):
    bad = dict(_GOOD)
    bad["wat"] = 1
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, bad))
    assert "wat" in str(ei.value)


def test_sample_size_must_be_divisible_by_n_bins(tmp_path):
    bad = dict(_GOOD, sample_size=51, n_bins=10)
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, bad))
    assert "divisible" in str(ei.value)


def test_negative_retries_rejected(tmp_path):
    bad = dict(_GOOD, judge_max_retries=-1)
    with pytest.raises(RefineConfigError):
        load_refine_config(_write(tmp_path, bad))


def test_zero_retries_allowed(tmp_path):
    cfg = load_refine_config(_write(tmp_path, dict(_GOOD, judge_max_retries=0)))
    assert cfg.judge_max_retries == 0


def test_qps_limit_must_be_positive(tmp_path):
    with pytest.raises(RefineConfigError):
        load_refine_config(_write(tmp_path, dict(_GOOD, judge_qps_limit=0.0)))


def test_temperature_upper_bound(tmp_path):
    with pytest.raises(RefineConfigError):
        load_refine_config(_write(tmp_path, dict(_GOOD, derive_temperature=2.5)))


def test_enabled_must_be_bool(tmp_path):
    with pytest.raises(RefineConfigError):
        load_refine_config(_write(tmp_path, dict(_GOOD, enabled="yes")))


def test_judge_api_key_env_optional_defaults_none(tmp_path):
    """Back-compat: configs without judge_api_key_env still load."""
    cfg = load_refine_config(_write(tmp_path, _GOOD))
    assert cfg.judge_api_key_env is None


def test_judge_api_key_env_accepts_override(tmp_path):
    """When set, judge_api_key_env decouples judge's bearer from api_key_env."""
    raw = dict(_GOOD)
    raw["judge_api_key_env"] = "VLLM_API_KEY"
    cfg = load_refine_config(_write(tmp_path, raw))
    assert cfg.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.judge_api_key_env == "VLLM_API_KEY"


def test_judge_api_key_env_rejects_empty_string(tmp_path):
    """If present, judge_api_key_env must be a non-empty string."""
    raw = dict(_GOOD)
    raw["judge_api_key_env"] = ""
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, raw))
    assert "judge_api_key_env" in str(ei.value)


def test_derive_enable_thinking_defaults_false(tmp_path):
    cfg = load_refine_config(_write(tmp_path, _GOOD))
    assert cfg.derive_enable_thinking is False


def test_derive_enable_thinking_accepts_true(tmp_path):
    raw = dict(_GOOD)
    raw["derive_enable_thinking"] = True
    cfg = load_refine_config(_write(tmp_path, raw))
    assert cfg.derive_enable_thinking is True


def test_derive_enable_thinking_rejects_non_bool(tmp_path):
    raw = dict(_GOOD)
    raw["derive_enable_thinking"] = "yes"
    with pytest.raises(RefineConfigError) as ei:
        load_refine_config(_write(tmp_path, raw))
    assert "derive_enable_thinking" in str(ei.value)
