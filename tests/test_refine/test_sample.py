"""Tests for the stratified sampler."""

from __future__ import annotations

import pytest

from src.refine.errors import RefineValidationError
from src.refine.sample import Phase2Record, stratified_sample


def _records(n: int = 200) -> list[Phase2Record]:
    return [
        Phase2Record(pk=i, nearest_fit_distance=i / n, raw={"pk": i}) for i in range(n)
    ]


def test_basic_per_decile_balance():
    sample = stratified_sample(
        _records(),
        sample_size=50,
        n_bins=5,
        seed=0,
        exclude_pks=frozenset(),
        known_intruder_pks=frozenset(),
    )
    assert len(sample.selected) == 50
    assert sample.per_decile_count == [40, 40, 40, 40, 40]
    assert sample.per_decile_drawn == [10, 10, 10, 10, 10]
    assert all(0 <= s.decile < 5 for s in sample.selected)


def test_modulo_remainder_in_last_bin():
    sample = stratified_sample(
        _records(n=203),  # 203 % 5 = 3 → last bin gets 3 extra
        sample_size=50,
        n_bins=5,
        seed=0,
        exclude_pks=frozenset(),
        known_intruder_pks=frozenset(),
    )
    # First 4 bins: 40 each. Last: 43.
    assert sample.per_decile_count == [40, 40, 40, 40, 43]


def test_intruders_short_circuited():
    intruders = frozenset({3, 17, 100})
    sample = stratified_sample(
        _records(),
        sample_size=50,
        n_bins=5,
        seed=0,
        exclude_pks=frozenset(),
        known_intruder_pks=intruders,
        auto_drop_known_intruders=True,
    )
    auto_pks = {s.record.pk for s in sample.auto_drop}
    selected_pks = {s.record.pk for s in sample.selected}
    # Intruders that were drawn move into auto_drop, not selected.
    assert auto_pks <= intruders
    assert selected_pks.isdisjoint(intruders)
    assert len(sample.selected) + len(sample.auto_drop) == 50


def test_excluded_pks_never_drawn():
    excluded = frozenset({0, 1, 2, 3, 4, 5})
    sample = stratified_sample(
        _records(),
        sample_size=50,
        n_bins=5,
        seed=0,
        exclude_pks=excluded,
        known_intruder_pks=frozenset(),
    )
    drawn = {s.record.pk for s in sample.selected}
    assert drawn.isdisjoint(excluded)


def test_determinism():
    a = stratified_sample(
        _records(),
        sample_size=50,
        n_bins=5,
        seed=42,
        exclude_pks=frozenset(),
        known_intruder_pks=frozenset(),
    )
    b = stratified_sample(
        _records(),
        sample_size=50,
        n_bins=5,
        seed=42,
        exclude_pks=frozenset(),
        known_intruder_pks=frozenset(),
    )
    assert [s.record.pk for s in a.selected] == [s.record.pk for s in b.selected]


def test_overflow_rejected():
    with pytest.raises(RefineValidationError):
        stratified_sample(
            _records(n=100),
            sample_size=200,
            n_bins=10,
            seed=0,
            exclude_pks=frozenset(),
            known_intruder_pks=frozenset(),
        )


def test_non_divisible_rejected():
    with pytest.raises(RefineValidationError):
        stratified_sample(
            _records(),
            sample_size=51,
            n_bins=5,
            seed=0,
            exclude_pks=frozenset(),
            known_intruder_pks=frozenset(),
        )
