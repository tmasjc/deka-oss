"""Unit tests for src.anchor.config — HarvestConfig YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.anchor.config import HarvestConfig, RadiusScheme, load_harvest_config
from src.anchor.errors import AnchorError


_COMPLETE_SECTION = (
    "harvest:\n"
    "  min_fit: 15\n"
    "  min_not_fit: 3\n"
    "  precision_at_k: 0.8\n"
    "  batch_size: 2000\n"
    "  max_k: 200000\n"
    "  s2c_outlier_multiple: 3.0\n"
    "  radius_scheme: per_fit\n"
    "  anchor_frequency_gate: 2\n"
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_valid_section(tmp_path: Path) -> None:
    cfg = load_harvest_config(_write_yaml(tmp_path, _COMPLETE_SECTION))
    assert isinstance(cfg, HarvestConfig)
    assert cfg.min_fit == 15
    assert cfg.min_not_fit == 3
    assert cfg.precision_at_k == 0.8
    assert cfg.batch_size == 2000
    assert cfg.max_k == 200000
    assert cfg.s2c_outlier_multiple == 3.0
    assert cfg.radius_scheme is RadiusScheme.PER_FIT
    assert cfg.anchor_frequency_gate == 2


def test_missing_key_raises(tmp_path: Path) -> None:
    partial = _COMPLETE_SECTION.replace("  max_k: 200000\n", "")
    with pytest.raises(AnchorError, match="missing required keys"):
        load_harvest_config(_write_yaml(tmp_path, partial))


def test_unknown_key_raises(tmp_path: Path) -> None:
    extra = _COMPLETE_SECTION + "  surprise: 42\n"
    with pytest.raises(AnchorError, match="unknown keys"):
        load_harvest_config(_write_yaml(tmp_path, extra))


def test_obsolete_keys_rejected(tmp_path: Path) -> None:
    """k_per_fit / auto_widen / allow_saturation were retired when
    Phase 2 moved to search_iterator. They must be rejected so stale
    configs surface loudly."""
    stale = _COMPLETE_SECTION + "  k_per_fit: 2000\n"
    with pytest.raises(AnchorError, match="unknown keys"):
        load_harvest_config(_write_yaml(tmp_path, stale))


def test_min_fit_must_be_positive_int(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("min_fit: 15", "min_fit: 0")
    with pytest.raises(AnchorError, match="min_fit"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_min_not_fit_required(tmp_path: Path) -> None:
    missing = _COMPLETE_SECTION.replace("  min_not_fit: 3\n", "")
    with pytest.raises(AnchorError, match="missing required keys"):
        load_harvest_config(_write_yaml(tmp_path, missing))


def test_min_not_fit_must_be_positive_int(tmp_path: Path) -> None:
    for bad_val in ("0", "-1"):
        bad = _COMPLETE_SECTION.replace("min_not_fit: 3", f"min_not_fit: {bad_val}")
        with pytest.raises(AnchorError, match="min_not_fit"):
            load_harvest_config(_write_yaml(tmp_path, bad))


def test_min_not_fit_rejects_non_int(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("min_not_fit: 3", "min_not_fit: 1.5")
    with pytest.raises(AnchorError, match="min_not_fit"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_min_not_fit_rejects_bool(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("min_not_fit: 3", "min_not_fit: true")
    with pytest.raises(AnchorError, match="min_not_fit"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_precision_at_k_range(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("precision_at_k: 0.8", "precision_at_k: 1.5")
    with pytest.raises(AnchorError, match="precision_at_k"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_batch_size_must_be_positive_int(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("batch_size: 2000", "batch_size: 0")
    with pytest.raises(AnchorError, match="batch_size"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_batch_size_cannot_exceed_milvus_cap(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("batch_size: 2000", "batch_size: 16385")
    with pytest.raises(AnchorError, match="batch_size"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_max_k_cannot_be_below_batch_size(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("max_k: 200000", "max_k: 500")
    with pytest.raises(AnchorError, match="max_k"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_max_k_must_be_positive_int(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace("max_k: 200000", "max_k: -1")
    with pytest.raises(AnchorError, match="max_k"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_s2c_outlier_multiple_must_exceed_one(tmp_path: Path) -> None:
    """Multipliers ≤ 1 drop every FIT with above-median δ — almost
    certainly a misconfig. Reject loudly."""
    for bad_val in ("1.0", "0.5", "-1"):
        bad = _COMPLETE_SECTION.replace(
            "s2c_outlier_multiple: 3.0", f"s2c_outlier_multiple: {bad_val}"
        )
        with pytest.raises(AnchorError, match="s2c_outlier_multiple"):
            load_harvest_config(_write_yaml(tmp_path, bad))


def test_s2c_outlier_multiple_must_be_numeric(tmp_path: Path) -> None:
    bad = _COMPLETE_SECTION.replace(
        "s2c_outlier_multiple: 3.0", 's2c_outlier_multiple: "oops"'
    )
    with pytest.raises(AnchorError, match="s2c_outlier_multiple"):
        load_harvest_config(_write_yaml(tmp_path, bad))


def test_s2c_outlier_multiple_required(tmp_path: Path) -> None:
    missing = _COMPLETE_SECTION.replace("  s2c_outlier_multiple: 3.0\n", "")
    with pytest.raises(AnchorError, match="missing required keys"):
        load_harvest_config(_write_yaml(tmp_path, missing))


def test_radius_scheme_required(tmp_path: Path) -> None:
    missing = _COMPLETE_SECTION.replace("  radius_scheme: per_fit\n", "")
    with pytest.raises(AnchorError, match="missing required keys"):
        load_harvest_config(_write_yaml(tmp_path, missing))


def test_radius_scheme_decoupled_parses(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "radius_scheme: per_fit", "radius_scheme: decoupled"
    )
    cfg = load_harvest_config(_write_yaml(tmp_path, body))
    assert cfg.radius_scheme is RadiusScheme.DECOUPLED


def test_radius_scheme_rejects_unknown_value(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "radius_scheme: per_fit", "radius_scheme: nonsense"
    )
    with pytest.raises(AnchorError, match="radius_scheme"):
        load_harvest_config(_write_yaml(tmp_path, body))


def test_radius_scheme_rejects_non_string(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "radius_scheme: per_fit", "radius_scheme: 42"
    )
    with pytest.raises(AnchorError, match="radius_scheme"):
        load_harvest_config(_write_yaml(tmp_path, body))


def test_anchor_frequency_gate_required(tmp_path: Path) -> None:
    """The gate is part of the required config surface — omitting it
    surfaces loudly rather than silently defaulting at the loader."""
    missing = _COMPLETE_SECTION.replace("  anchor_frequency_gate: 2\n", "")
    with pytest.raises(AnchorError, match="missing required keys"):
        load_harvest_config(_write_yaml(tmp_path, missing))


def test_anchor_frequency_gate_disabled_at_one_loads(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "anchor_frequency_gate: 2", "anchor_frequency_gate: 1"
    )
    cfg = load_harvest_config(_write_yaml(tmp_path, body))
    assert cfg.anchor_frequency_gate == 1


def test_anchor_frequency_gate_rejects_zero(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "anchor_frequency_gate: 2", "anchor_frequency_gate: 0"
    )
    with pytest.raises(AnchorError, match="anchor_frequency_gate"):
        load_harvest_config(_write_yaml(tmp_path, body))


def test_anchor_frequency_gate_rejects_negative(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "anchor_frequency_gate: 2", "anchor_frequency_gate: -1"
    )
    with pytest.raises(AnchorError, match="anchor_frequency_gate"):
        load_harvest_config(_write_yaml(tmp_path, body))


def test_anchor_frequency_gate_rejects_float(tmp_path: Path) -> None:
    body = _COMPLETE_SECTION.replace(
        "anchor_frequency_gate: 2", "anchor_frequency_gate: 1.5"
    )
    with pytest.raises(AnchorError, match="anchor_frequency_gate"):
        load_harvest_config(_write_yaml(tmp_path, body))


def test_anchor_frequency_gate_rejects_bool(tmp_path: Path) -> None:
    """``True`` is a Python int subclass; the loader must reject it
    explicitly so misformatted YAML doesn't silently coerce to f=1."""
    body = _COMPLETE_SECTION.replace(
        "anchor_frequency_gate: 2", "anchor_frequency_gate: true"
    )
    with pytest.raises(AnchorError, match="anchor_frequency_gate"):
        load_harvest_config(_write_yaml(tmp_path, body))
