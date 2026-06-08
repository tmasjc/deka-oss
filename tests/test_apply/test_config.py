"""Tests for src.apply.config — strict required-keys + value validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.apply.config import load_apply_config
from src.apply.errors import ApplyConfigError


_GOOD = {
    "enabled": True,
    "confidence_threshold": 0.7,
    "min_precision": 0.9,
    "eval_fraction": 0.2,
    "eval_n_repeats": 5,
    "kfold_splits": 5,
    "seed": 0,
    "embedding_dim": 1024,
    "embedding_fetch_batch": 2000,
}


def _write(tmp_path: Path, apply: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"apply": apply}), encoding="utf-8")
    return p


def test_happy_path(tmp_path):
    cfg = load_apply_config(_write(tmp_path, _GOOD))
    assert cfg.enabled is True
    assert cfg.confidence_threshold == 0.7
    assert cfg.min_precision == 0.9
    assert cfg.eval_fraction == 0.2
    assert cfg.kfold_splits == 5
    assert cfg.embedding_dim == 1024


def test_missing_key_rejected(tmp_path):
    bad = dict(_GOOD)
    bad.pop("min_precision")
    with pytest.raises(ApplyConfigError) as ei:
        load_apply_config(_write(tmp_path, bad))
    assert "min_precision" in str(ei.value)


def test_unknown_key_rejected(tmp_path):
    bad = dict(_GOOD, surprise=1)
    with pytest.raises(ApplyConfigError) as ei:
        load_apply_config(_write(tmp_path, bad))
    assert "surprise" in str(ei.value)


def test_eval_n_repeats_defaults_when_missing(tmp_path):
    """Back-compat: pre-migration configs omit ``eval_n_repeats`` and
    must still load (with the default), so a code-only redeploy
    doesn't break operators who haven't touched their YAML.
    """
    legacy = dict(_GOOD)
    legacy.pop("eval_n_repeats")
    cfg = load_apply_config(_write(tmp_path, legacy))
    assert cfg.eval_n_repeats == 5


def test_eval_n_repeats_respects_explicit_value(tmp_path):
    cfg = load_apply_config(_write(tmp_path, dict(_GOOD, eval_n_repeats=3)))
    assert cfg.eval_n_repeats == 3


def test_eval_n_repeats_rejects_zero(tmp_path):
    with pytest.raises(ApplyConfigError):
        load_apply_config(_write(tmp_path, dict(_GOOD, eval_n_repeats=0)))


def test_threshold_outside_unit_interval(tmp_path):
    with pytest.raises(ApplyConfigError):
        load_apply_config(_write(tmp_path, dict(_GOOD, confidence_threshold=1.2)))
    with pytest.raises(ApplyConfigError):
        load_apply_config(_write(tmp_path, dict(_GOOD, confidence_threshold=-0.1)))


def test_eval_fraction_must_be_strictly_between_0_and_1(tmp_path):
    with pytest.raises(ApplyConfigError):
        load_apply_config(_write(tmp_path, dict(_GOOD, eval_fraction=0.0)))
    with pytest.raises(ApplyConfigError):
        load_apply_config(_write(tmp_path, dict(_GOOD, eval_fraction=1.0)))


def test_kfold_splits_minimum(tmp_path):
    with pytest.raises(ApplyConfigError):
        load_apply_config(_write(tmp_path, dict(_GOOD, kfold_splits=1)))


def test_seed_allows_zero(tmp_path):
    cfg = load_apply_config(_write(tmp_path, dict(_GOOD, seed=0)))
    assert cfg.seed == 0
