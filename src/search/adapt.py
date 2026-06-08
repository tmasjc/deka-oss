"""Deterministic adapt step — diagnostics-only since fusion is RRFRanker.

Runs after the Turn-0 probe and before the first fused search. Inspects
the per-path probe stats and raises :class:`AdaptError` when every path
is dead, so the user can pick a different query before the TUI commits
to an unproductive rating cycle. When only one path is active, emits a
flag so the user knows fusion is degenerate for this query.

No weights are mutated — RRFRanker normalizes by rank position, so
score-scale mismatches between paths are irrelevant at fusion time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import AdaptError

if TYPE_CHECKING:
    from .config import SearchConfig
    from .evidence import PathName
    from .search import ProbeResult


@dataclass(frozen=True)
class AdaptedConfig:
    """Output of :func:`adapt_config` — carries the seed config plus diagnostics."""

    config: "SearchConfig"
    rationale: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def adapt_config(seed: "SearchConfig", probe: "ProbeResult") -> AdaptedConfig:
    """Diagnose the probe and gate the turn on at least one active path.

    Raises :class:`AdaptError` when every path returned zero hits — the
    TUI surfaces this as a notification so the user can pick a different
    query. Emits a flag when only one path is active so the agent knows
    fusion is degenerate for this query.
    """

    stats = probe.stats_by_path
    paths: tuple[PathName, ...] = ("dense", "sparse")
    dead = [p for p in paths if _is_dead(stats.get(p))]
    active = [p for p in paths if p not in dead]

    if not active:
        raise AdaptError(
            "Both retrieval paths returned no hits for the probe — "
            "the query is likely uncovered by the corpus."
        )

    rationale = [f"{p} returned no hits in probe" for p in dead]
    flags: list[str] = []
    if len(active) == 1:
        flags.append(
            f"single-path start: only {active[0]} returned hits; "
            "fusion is degenerate until additional paths produce candidates"
        )

    return AdaptedConfig(config=seed, rationale=rationale, flags=flags)


def _is_dead(path_stats: dict | None) -> bool:
    if path_stats is None:
        return True
    if path_stats.get("skipped"):
        return True
    return path_stats.get("hit_count", 0) == 0
