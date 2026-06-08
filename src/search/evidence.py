"""Evidence-table data structures.

Canonical spec lives in ``harness/schemas/evidence.md``. ``run_search``
produces an :class:`EvidenceTable` with ``rating=None`` on every row;
the rater step fills ratings in place via :meth:`EvidenceTable.set_rating`,
then :func:`compute_breakdown` tallies the per-path breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .config import SearchConfig

Rating = Literal["FIT", "NOT_FIT", "DISCARD"]
PathName = Literal["dense", "sparse"]

ALL_PATHS: tuple[PathName, ...] = ("dense", "sparse")


PrimaryKey = int | str


@dataclass
class EvidenceRow:
    """One row of the evidence table — one chunk in the fused result list."""

    rank: int
    pk: PrimaryKey  # Milvus primary key — opaque join key, may be INT64 or VARCHAR
    chunk_id: str  # display form derived from sample_id + pk
    chunk_content: str  # verbatim, never modified
    sample_id: str
    counselor_id: str
    term: str
    source_paths: list[PathName]
    scores: dict[PathName, float]
    rating: Rating | None = None
    span_line_indices: list[int] = field(default_factory=list)
    span_text: str = ""


@dataclass
class CandidateRow:
    """One per-path candidate — a chunk in path's top-3 NOT in the fused top-K.

    Surfaced separately from :class:`EvidenceRow` so the agent can see whether
    a single path's strong candidates were FIT (ranking issue, consider
    RRFRanker) or NOT_FIT (path is genuinely noisy for this query).
    """

    path: PathName
    rank_in_path: int  # 1, 2, or 3
    pk: PrimaryKey
    chunk_id: str
    chunk_content: str
    sample_id: str
    counselor_id: str
    term: str
    score: float  # the path's own score (no fusion)
    rating: Rating | None = None
    span_line_indices: list[int] = field(default_factory=list)
    span_text: str = ""


def _empty_candidates() -> dict[PathName, list[CandidateRow]]:
    return {"dense": [], "sparse": []}


@dataclass
class EvidenceTable:
    """The full evidence payload for one tuning turn."""

    query: str
    config: "SearchConfig"
    rows: list[EvidenceRow] = field(default_factory=list)
    # Per-path top candidates that did NOT survive fusion (deduped against
    # ``rows``). Empty lists are normal — a path's top 3 may all be in the
    # fused top-K, or the path may have returned zero hits.
    per_path_candidates: dict[PathName, list[CandidateRow]] = field(
        default_factory=_empty_candidates
    )
    # Populated by run_search; consumed by the logging hook. None when
    # the table was constructed by hand (e.g. in tests that don't
    # exercise the search path).
    search_diagnostics: dict | None = None
    # Count of short chunks dropped before display this turn.
    # Populated by run_search; zero means no filtering occurred.
    filtered_short_chunk: int = 0
    # Count of duplicate-sample_id chunks dropped before display this turn.
    filtered_duplicate_sample: int = 0
    # Count of chunks dropped because the span extractor errored on them.
    # Zero when no extractor ran or every extraction succeeded.
    dropped_by_extractor: int = 0

    def set_rating(self, rank: int, rating: Rating) -> None:
        """Fill in the human rating for the row at ``rank`` (1-indexed)."""

        for row in self.rows:
            if row.rank == rank:
                row.rating = rating
                return
        raise KeyError(f"No row with rank={rank} in evidence table")

    def set_candidate_rating(
        self, path: PathName, rank_in_path: int, rating: Rating
    ) -> None:
        """Fill in the human rating for a per-path candidate."""

        for cand in self.per_path_candidates.get(path, []):
            if cand.rank_in_path == rank_in_path:
                cand.rating = rating
                return
        raise KeyError(f"No candidate with path={path!r} rank_in_path={rank_in_path}")

    def all_candidates(self) -> list[CandidateRow]:
        """Flatten per-path candidates in dense → sparse order."""
        out: list[CandidateRow] = []
        for path in ALL_PATHS:
            out.extend(self.per_path_candidates.get(path, []))
        return out

    def all_rated(self, include_candidates: bool = False) -> bool:
        """True iff every fused row (and optionally every per-path candidate) is rated.

        Regular turns rate only the fused top-K; per-path candidates are
        materialised by ``run_search`` but never surfaced for rating. Audit
        turns flip ``include_candidates=True`` so the gate also requires
        candidate ratings before the operator can drop a path or finish.
        """
        if any(row.rating is None for row in self.rows):
            return False
        if include_candidates:
            for path in ALL_PATHS:
                for cand in self.per_path_candidates.get(path, []):
                    if cand.rating is None:
                        return False
        return True


def compute_breakdown(table: EvidenceTable) -> dict[str, dict[str, int]]:
    """Tally dense_only / sparse_only / multi_path counts.

    Implements step 4 of ``harness/schemas/evidence.md``. Raises
    :class:`ValueError` if any row is still unrated — unrated rows
    reaching this function indicate an orchestration bug.
    """

    breakdown: dict[str, dict[str, int]] = {
        "dense_only": {"total": 0, "fit": 0, "not_fit": 0, "discard": 0},
        "sparse_only": {"total": 0, "fit": 0, "not_fit": 0, "discard": 0},
        "multi_path": {"total": 0, "fit": 0, "not_fit": 0, "discard": 0},
    }

    for row in table.rows:
        if row.rating is None:
            raise ValueError(
                f"Row rank={row.rank} is unrated; cannot compute breakdown"
            )

        paths = row.source_paths
        if len(paths) == 1:
            key = f"{paths[0]}_only"
        else:
            key = "multi_path"

        breakdown[key]["total"] += 1
        if row.rating == "FIT":
            breakdown[key]["fit"] += 1
        elif row.rating == "NOT_FIT":
            breakdown[key]["not_fit"] += 1
        else:
            breakdown[key]["discard"] += 1

    return breakdown


def sort_paths(paths: set[PathName]) -> list[PathName]:
    """Deterministic path ordering so evidence tables diff cleanly."""

    return [p for p in ALL_PATHS if p in paths]
