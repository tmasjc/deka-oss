"""Shared fixtures for reflection tests."""

from __future__ import annotations

from src.search.config import SearchConfig
from src.search.evidence import CandidateRow, EvidenceRow, EvidenceTable
from src.session.state import SessionState, TurnRecord


def make_config(**overrides: object) -> SearchConfig:
    defaults: dict[str, object] = {
        "rrf_k": 60,
        "per_path_limit": 20,
        "top_k": 10,
    }
    defaults.update(overrides)
    return SearchConfig(**defaults)  # type: ignore[arg-type]


def make_row(
    rank: int,
    rating: str = "FIT",
    source_paths: list[str] | None = None,
    scores: dict[str, float] | None = None,
) -> EvidenceRow:
    return EvidenceRow(
        rank=rank,
        pk=rank,
        chunk_id=f"sample_{rank:03d}_C{rank:07d}",
        chunk_content=f"Chunk content {rank}",
        sample_id=f"sample_{rank:03d}",
        counselor_id="T0001",
        term="2025F",
        source_paths=source_paths or ["dense"],
        scores=scores or {"dense": 0.8},
        rating=rating,
    )


def make_candidate(
    path: str = "sparse",
    rank_in_path: int = 1,
    rating: str | None = None,
    pk: int | str | None = None,
) -> CandidateRow:
    if pk is None:
        pk = f"{path}-cand-{rank_in_path}"
    return CandidateRow(
        path=path,  # type: ignore[arg-type]
        rank_in_path=rank_in_path,
        pk=pk,
        chunk_id=f"sample_{path}_C{rank_in_path:07d}",
        chunk_content=f"{path} candidate {rank_in_path}",
        sample_id=f"sample_{path}",
        counselor_id="T0001",
        term="2025F",
        score=0.5,
        rating=rating,  # type: ignore[arg-type]
    )


def make_table(
    rows: list[EvidenceRow] | None = None,
    config: SearchConfig | None = None,
    query: str = "测试查询",
) -> EvidenceTable:
    if config is None:
        config = make_config()
    if rows is None:
        rows = [make_row(i, "FIT" if i % 2 == 0 else "NOT_FIT") for i in range(1, 4)]
    return EvidenceTable(query=query, config=config, rows=rows)


def make_turn(
    turn_number: int,
    config: SearchConfig | None = None,
    precision: float = 0.6,
    reflection: dict | None = None,
) -> TurnRecord:
    if config is None:
        config = make_config()
    table = make_table(config=config)
    breakdown = {
        "dense_only": {"total": 2, "fit": 1, "not_fit": 1},
        "sparse_only": {"total": 0, "fit": 0, "not_fit": 0},
        "bm25_only": {"total": 0, "fit": 0, "not_fit": 0},
        "multi_path": {"total": 1, "fit": 1, "not_fit": 0},
    }
    return TurnRecord(
        turn_number=turn_number,
        query="测试查询",
        config=config,
        evidence_table=table,
        breakdown=breakdown,
        precision=precision,
        reflection=reflection,
    )


def make_state(
    turns: list[TurnRecord] | None = None,
    config: SearchConfig | None = None,
) -> SessionState:
    if config is None:
        config = make_config()
    state = SessionState(current_config=config)
    if turns:
        state.turns = turns
        state.query = turns[0].query
    return state
