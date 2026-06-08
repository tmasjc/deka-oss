"""Harvest configuration — Phase 1 convergence + Phase 2 retrieval.

The ``harvest:`` YAML section owns the whole Phase 1 → Phase 2
lifecycle: convergence dual gate (``min_fit`` + ``precision_at_k``)
that decides when Phase 1 can hand off, and Phase 2 retrieval
defaults (``batch_size``, ``max_k``).

Callers that pass ``batch_size=None`` / ``max_k=None`` to
:func:`src.anchor.runner.run_anchor` resolve against these values;
explicit kwargs/CLI flags still win.

``batch_size`` is the ``search_iterator`` page size; it must stay
under Milvus's per-page cap (``pymilvus.orm.constants.MAX_BATCH_SIZE``
== 16384). ``max_k`` is the user's safety cap on per-FIT cumulative
hits — the iterator halts at ``max_k`` even if the K-th distance is
still ≤ T'. Budget exhaustion is reported in the sidecar, never
aborts the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.config_loader import ConfigFileError, load_section

from .errors import AnchorError

_SECTION = "harvest"
_MILVUS_PAGE_CAP = 16384
_REQUIRED_KEYS = frozenset(
    {
        "min_fit",
        "min_not_fit",
        "precision_at_k",
        "batch_size",
        "max_k",
        "s2c_outlier_multiple",
        "radius_scheme",
        "anchor_frequency_gate",
    }
)


class RadiusScheme(str, Enum):
    """Phase 2 threshold scheme for the main-pass retrieval.

    ``per_fit`` — legacy. Each FIT uses its own ``T'_i = T + δ_i`` as
    the Milvus filter/stopping rule and as ``threshold_T_prime`` in
    the output JSONL. A drifted FIT widens its own radius.

    ``decoupled`` — issue #20. LOO still uses per-FIT ``T'_i``, but
    the main pass uses the session-wide ``T'_out = T + min(δ)`` for
    every FIT. ``min`` (not median) is the default aggregator based
    on reference-session evidence: it matches cohort-consistency
    under the widest range of anchor drifts and keeps output size
    bounded by the tightest natural spread in the cohort.
    """

    PER_FIT = "per_fit"
    DECOUPLED = "decoupled"


@dataclass(frozen=True)
class HarvestConfig:
    # Phase 1 convergence gate
    min_fit: int
    # Minimum cumulative unique NOT_FIT pks required for convergence.
    # Phase 3 needs negatives to derive a discriminating rubric; without
    # this floor a session can converge with zero NOT_FITs and starve
    # the meta-prompt of contrastive examples.
    min_not_fit: int
    precision_at_k: float
    # Phase 2 retrieval defaults
    batch_size: int
    max_k: int
    # Phase 2 anchor-quality gate: drop a FIT when its δ_i exceeds
    # ``s2c_outlier_multiple · median(δ)``. The complementary
    # ``δ_i > T`` check is always on regardless of this knob.
    s2c_outlier_multiple: float
    # Phase 2 main-pass threshold scheme. See RadiusScheme docstring.
    radius_scheme: RadiusScheme
    # Phase 2 anchor-frequency gate: a chunk is retained only if it
    # falls within the calibrated radius of at least ``f`` distinct
    # FIT anchors. ``f = 1`` disables the gate (legacy single-anchor
    # behaviour); higher values tighten the cohort. Must be ≤ the
    # post-quality-gate FIT count at runtime — exceeding it aborts
    # before the main pass with :class:`AnchorValidationError`.
    anchor_frequency_gate: int


def load_harvest_config(
    path: Path | None = None,
    session_overrides: dict[str, Any] | None = None,
) -> HarvestConfig:
    """Load the harvest section from the unified YAML.

    Every key is required. Raises :class:`AnchorError` on any parse
    or validation failure so callers can funnel config problems into
    the same error surface as the rest of the anchor pipeline.

    ``session_overrides`` (when present) shadows YAML values for this
    load only; see :mod:`src.web_api.schemas.SessionOverrides`.
    """
    try:
        raw = load_section(_SECTION, explicit=path, overrides=session_overrides)
    except ConfigFileError as exc:
        raise AnchorError(str(exc)) from exc

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise AnchorError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}"
        )
    unknown = raw.keys() - _REQUIRED_KEYS
    if unknown:
        raise AnchorError(
            f"config section '{_SECTION}' contains unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_REQUIRED_KEYS)}"
        )

    min_fit = raw["min_fit"]
    if not isinstance(min_fit, int) or isinstance(min_fit, bool) or min_fit <= 0:
        raise AnchorError(
            f"config section '{_SECTION}': 'min_fit' must be a positive integer"
        )

    min_not_fit = raw["min_not_fit"]
    if (
        not isinstance(min_not_fit, int)
        or isinstance(min_not_fit, bool)
        or min_not_fit <= 0
    ):
        raise AnchorError(
            f"config section '{_SECTION}': 'min_not_fit' must be a positive integer"
        )

    precision_at_k = raw["precision_at_k"]
    if (
        isinstance(precision_at_k, bool)
        or not isinstance(precision_at_k, (int, float))
        or not 0.0 <= float(precision_at_k) <= 1.0
    ):
        raise AnchorError(
            f"config section '{_SECTION}': 'precision_at_k' must be a "
            "number in [0.0, 1.0]"
        )

    batch_size = raw["batch_size"]
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise AnchorError(
            f"config section '{_SECTION}': 'batch_size' must be a positive integer"
        )
    if batch_size > _MILVUS_PAGE_CAP:
        raise AnchorError(
            f"config section '{_SECTION}': 'batch_size' ({batch_size}) exceeds "
            f"Milvus per-page cap ({_MILVUS_PAGE_CAP})"
        )

    max_k = raw["max_k"]
    if not isinstance(max_k, int) or isinstance(max_k, bool) or max_k < 1:
        raise AnchorError(
            f"config section '{_SECTION}': 'max_k' must be a positive integer"
        )
    if max_k < batch_size:
        raise AnchorError(
            f"config section '{_SECTION}': 'max_k' ({max_k}) must be >= "
            f"'batch_size' ({batch_size}) — a single page already overshoots the cap."
        )

    s2c_outlier_multiple = raw["s2c_outlier_multiple"]
    if (
        isinstance(s2c_outlier_multiple, bool)
        or not isinstance(s2c_outlier_multiple, (int, float))
        or float(s2c_outlier_multiple) <= 1.0
    ):
        raise AnchorError(
            f"config section '{_SECTION}': 's2c_outlier_multiple' must be a "
            "number > 1.0 (multipliers ≤ 1 would drop every FIT with "
            "above-median δ — almost certainly a misconfig)."
        )

    radius_scheme_raw = raw["radius_scheme"]
    if not isinstance(radius_scheme_raw, str):
        raise AnchorError(
            f"config section '{_SECTION}': 'radius_scheme' must be a string "
            f"(one of {[s.value for s in RadiusScheme]})."
        )
    try:
        radius_scheme = RadiusScheme(radius_scheme_raw)
    except ValueError as exc:
        raise AnchorError(
            f"config section '{_SECTION}': 'radius_scheme' = "
            f"{radius_scheme_raw!r} is not one of "
            f"{[s.value for s in RadiusScheme]}."
        ) from exc

    anchor_frequency_gate = raw["anchor_frequency_gate"]
    if (
        not isinstance(anchor_frequency_gate, int)
        or isinstance(anchor_frequency_gate, bool)
        or anchor_frequency_gate < 1
    ):
        raise AnchorError(
            f"config section '{_SECTION}': 'anchor_frequency_gate' must be "
            "a positive integer >= 1 (1 disables the gate; ≥2 requires that "
            "many distinct FIT anchors to admit a chunk)."
        )

    return HarvestConfig(
        min_fit=min_fit,
        min_not_fit=min_not_fit,
        precision_at_k=float(precision_at_k),
        batch_size=batch_size,
        max_k=max_k,
        s2c_outlier_multiple=float(s2c_outlier_multiple),
        radius_scheme=radius_scheme,
        anchor_frequency_gate=anchor_frequency_gate,
    )
