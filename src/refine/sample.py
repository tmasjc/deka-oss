"""Stratified-by-decile sampling of the Phase 2 cohort.

Phase 3 judges a small, representative slice of the Phase 2 output
rather than the whole cohort. Stratifying by ``nearest_fit_distance``
(the geometric distance to the closest FIT) ensures the sample
covers the whole boundary — close-in chunks the rubric should KEEP
emphatically, mid-band chunks where the predicate matters, and
far-out chunks where the rubric should DROP cleanly.

The sampler is a pure function over already-loaded Phase 2 records;
it does not touch Milvus or the LLM endpoint.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.search.evidence import PrimaryKey

from .errors import RefineLoadError, RefineValidationError


@dataclass(frozen=True)
class Phase2Record:
    """One row of ``runs/{sid}.phase2.jsonl`` projected to the fields
    Phase 3 needs. Other fields are preserved on the record but not
    typed here.
    """

    pk: PrimaryKey
    nearest_fit_distance: float
    raw: dict[str, Any]  # full original record, for downstream propagation


@dataclass(frozen=True)
class SampledRecord:
    """One drawn record paired with its assigned decile bin."""

    record: Phase2Record
    decile: int  # 0-based bin index


@dataclass(frozen=True)
class StratifiedSample:
    """Result of one stratified draw."""

    selected: list[SampledRecord]
    auto_drop: list[SampledRecord]  # known intruders short-circuited
    decile_boundaries: list[float]  # length == n_bins + 1
    per_decile_count: list[int]  # population size per bin (all eligible records)
    per_decile_drawn: list[
        int
    ]  # actual draws per bin (sums to len(selected) + len(auto_drop))
    excluded_pks: frozenset[PrimaryKey] = field(default_factory=frozenset)


def load_phase2_records(runs_dir: Path, session_id: str) -> list[Phase2Record]:
    """Read ``runs/{session_id}.phase2.jsonl`` and project each line to
    a :class:`Phase2Record`.

    Raises :class:`RefineLoadError` if the sidecar is missing or any
    line is malformed; the operator is meant to re-run Phase 2 in
    that case, not let Phase 3 silently sample a partial input.
    """
    path = runs_dir / f"{session_id}.phase2.jsonl"
    if not path.exists():
        raise RefineLoadError(
            f"Phase 2 sidecar missing: {path}. Run Phase 2 first "
            "(`python -m src.anchor <session_id>`) before Phase 3."
        )

    records: list[Phase2Record] = []
    for line_num, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RefineLoadError(f"{path} line {line_num} malformed: {exc}") from exc
        if "pk" not in entry or "nearest_fit_distance" not in entry:
            raise RefineLoadError(
                f"{path} line {line_num} missing required field "
                "'pk' or 'nearest_fit_distance'"
            )
        records.append(
            Phase2Record(
                pk=entry["pk"],
                nearest_fit_distance=float(entry["nearest_fit_distance"]),
                raw=entry,
            )
        )
    if not records:
        raise RefineLoadError(
            f"{path} is empty — Phase 2 produced no candidates. Phase 3 has "
            "nothing to sample."
        )
    return records


def stratified_sample(
    records: list[Phase2Record],
    *,
    sample_size: int,
    n_bins: int,
    seed: int,
    exclude_pks: frozenset[PrimaryKey],
    known_intruder_pks: frozenset[PrimaryKey],
    auto_drop_known_intruders: bool = True,
) -> StratifiedSample:
    """Draw ``sample_size`` records stratified by decile.

    Records are sorted by ``nearest_fit_distance`` ascending, sliced
    into ``n_bins`` equal-sized bins (the last bin absorbs the modulo
    remainder), and ``sample_size / n_bins`` records are drawn
    uniformly within each bin.

    Records whose PK is in ``exclude_pks`` (Phase-1-rated rows) are
    filtered out before binning — judging them again would leak the
    operator's prior labels into the rubric audit.

    When ``auto_drop_known_intruders`` is true and a drawn record's
    PK is in ``known_intruder_pks``, the record is moved into
    ``auto_drop`` instead of ``selected``; the judge skips it and the
    writer records ``verdict='DROP'`` with reason
    ``'auto_drop_known_intruder'``.

    Determinism: the RNG is seeded with ``seed`` and the per-bin
    population is sorted by ``(nearest_fit_distance, pk_str)`` before
    drawing, so two runs with the same inputs produce identical
    samples.
    """
    if sample_size <= 0:
        raise RefineValidationError("sample_size must be positive")
    if n_bins <= 0:
        raise RefineValidationError("n_bins must be positive")
    if sample_size % n_bins != 0:
        raise RefineValidationError(
            f"sample_size ({sample_size}) must be divisible by n_bins ({n_bins})"
        )

    eligible = [r for r in records if r.pk not in exclude_pks]
    if len(eligible) < sample_size:
        raise RefineValidationError(
            f"Phase 2 cohort has {len(eligible)} eligible records after "
            f"excluding {len(exclude_pks)} Phase-1-rated PKs; need "
            f"sample_size={sample_size}. Lower 'refine.sample_size' or "
            "rate fewer rows in Phase 1."
        )

    eligible_sorted = sorted(
        eligible, key=lambda r: (r.nearest_fit_distance, str(r.pk))
    )
    bins = _slice_into_bins(eligible_sorted, n_bins)
    per_decile_count = [len(b) for b in bins]
    boundaries = _decile_boundaries(eligible_sorted, n_bins)

    rng = random.Random(seed)
    per_bin_draw = sample_size // n_bins

    selected: list[SampledRecord] = []
    auto_drop: list[SampledRecord] = []
    per_decile_drawn = [0] * n_bins
    for idx, bucket in enumerate(bins):
        if len(bucket) < per_bin_draw:
            raise RefineValidationError(
                f"decile {idx} has only {len(bucket)} records but "
                f"sample requires {per_bin_draw} per bin. Lower "
                "'refine.sample_size' or 'refine.n_bins'."
            )
        drawn = rng.sample(bucket, per_bin_draw)
        per_decile_drawn[idx] = len(drawn)
        for rec in drawn:
            sampled = SampledRecord(record=rec, decile=idx)
            if auto_drop_known_intruders and rec.pk in known_intruder_pks:
                auto_drop.append(sampled)
            else:
                selected.append(sampled)

    return StratifiedSample(
        selected=selected,
        auto_drop=auto_drop,
        decile_boundaries=boundaries,
        per_decile_count=per_decile_count,
        per_decile_drawn=per_decile_drawn,
        excluded_pks=exclude_pks,
    )


def _slice_into_bins(
    sorted_records: list[Phase2Record], n_bins: int
) -> list[list[Phase2Record]]:
    """Slice a sorted list into ``n_bins`` near-equal buckets.

    The last bin absorbs the modulo remainder so ``concat(bins) ==
    sorted_records`` exactly. Empty input yields ``n_bins`` empty
    buckets.
    """
    n = len(sorted_records)
    if n == 0:
        return [[] for _ in range(n_bins)]
    base = n // n_bins
    bins: list[list[Phase2Record]] = []
    for i in range(n_bins):
        start = i * base
        end = (i + 1) * base if i < n_bins - 1 else n
        bins.append(sorted_records[start:end])
    return bins


def _decile_boundaries(sorted_records: list[Phase2Record], n_bins: int) -> list[float]:
    """Return ``n_bins + 1`` distance boundaries.

    boundary[0] is the min distance, boundary[n_bins] is the max,
    intermediate boundaries are the first record of each subsequent
    bin's distance. Empty input returns ``[0.0] * (n_bins + 1)`` —
    callers should guard against the empty-cohort case before this
    is meaningful.
    """
    if not sorted_records:
        return [0.0] * (n_bins + 1)
    bins = _slice_into_bins(sorted_records, n_bins)
    boundaries: list[float] = [sorted_records[0].nearest_fit_distance]
    for bucket in bins[1:]:
        if bucket:
            boundaries.append(bucket[0].nearest_fit_distance)
        else:
            boundaries.append(boundaries[-1])
    boundaries.append(sorted_records[-1].nearest_fit_distance)
    return boundaries


def load_known_intruder_pks(runs_dir: Path, session_id: str) -> frozenset[PrimaryKey]:
    """Read ``runs/{session_id}.phase2.meta.json`` and pull the
    intruder PK list.

    Returns an empty frozenset if the file is absent or carries no
    intruder block — Phase 3 should still run, just without the
    auto-DROP optimisation.
    """
    path = runs_dir / f"{session_id}.phase2.meta.json"
    if not path.exists():
        return frozenset()
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return frozenset()
    intrusion = meta.get("not_fit_intrusion") or {}
    pks = intrusion.get("intruder_pks") or []
    return frozenset(pks)
