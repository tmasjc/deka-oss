"""Hybrid search executor for the tuning harness.

Public surface:

* :func:`run_search` — one call that produces an :class:`EvidenceTable`.
* :class:`SearchConfig` — tunable parameter set.
* :func:`load_default_config` — read ``defaults.yaml`` + env vars.
* :func:`with_overrides` — reflection-step helper for turn-to-turn edits.
* :class:`EvidenceRow`, :class:`EvidenceTable` — output data structures.
* :func:`compute_breakdown` — post-rating per-path tally.
"""

from .adapt import AdaptedConfig, AdaptError, adapt_config
from .config import SearchConfig, load_default_config, with_overrides
from .errors import ConfigError, EmbeddingServiceError, MilvusSearchError
from .evidence import CandidateRow, EvidenceRow, EvidenceTable, compute_breakdown
from .search import ProbeResult, probe_only, run_search

__all__ = [
    "AdaptError",
    "AdaptedConfig",
    "CandidateRow",
    "ConfigError",
    "EmbeddingServiceError",
    "EvidenceRow",
    "EvidenceTable",
    "MilvusSearchError",
    "ProbeResult",
    "SearchConfig",
    "adapt_config",
    "compute_breakdown",
    "load_default_config",
    "probe_only",
    "run_search",
    "with_overrides",
]
