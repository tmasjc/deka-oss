"""Phase-3 session loader — FIT/NOT_FIT enumeration without Milvus.

Phase 3 needs three slices of a converged Phase 1 session:

- FIT exemplars: ``(pk, span_text)`` pairs the meta-prompt cites as
  positive examples.
- NOT_FIT exemplars: ``(pk, chunk_content)`` pairs the meta-prompt
  cites as concept-adjacent negatives. We carry the full chunk
  (not just a span) because Phase 1 does not extract spans on
  NOT_FIT rows — the meta-prompt picks discriminating language out
  of the whole chunk.
- Rated PKs: the union of FIT + NOT_FIT pks across all turns. The
  sampler excludes these from the Phase 2 cohort so judging never
  re-evaluates a row the operator already labelled.

Implementation reuses the dedup logic of
:mod:`src.anchor.loader` — last-write-wins across turns — but strips
the Milvus + embed-service dependencies (Phase 3 only reads the
session log; chunk embeddings are not needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.replay.loader import ReplaySession, load_session
from src.replay.metrics import _cumulative_discard_pks
from src.search.evidence import PrimaryKey

from .errors import RefineLoadError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitExemplar:
    """A Phase-1 FIT chunk reduced to what the meta-prompt cites."""

    pk: PrimaryKey
    chunk_id: str
    span_text: str
    span_line_indices: list[int]


@dataclass(frozen=True)
class NotFitExemplar:
    """A Phase-1 NOT_FIT chunk reduced to what the meta-prompt reads."""

    pk: PrimaryKey
    chunk_id: str
    chunk_content: str


@dataclass(frozen=True)
class Phase3SessionInputs:
    """Everything Phase 3's derive step needs from the source session."""

    session_id: str
    query: str
    fits: list[FitExemplar]
    not_fits: list[NotFitExemplar]
    rated_pks: frozenset[PrimaryKey]
    reflection_diagnoses: list[str]


def load_phase3_session_inputs(
    session_target: str | Path, *, runs_dir: Path | None = None
) -> Phase3SessionInputs:
    """Read the Phase 1 session log and produce Phase 3's derive inputs.

    Skips any FIT row that lacks a usable span — those would be
    extraction misses, not concept exemplars. Drops NOT_FIT rows
    without ``chunk_content`` (only present on pre-replay logs that
    Phase 2 already rejects, kept here as defence-in-depth).

    Raises :class:`RefineLoadError` if the session has fewer than two
    FIT exemplars (the meta-prompt needs at least two positives to
    name a discriminator).
    """
    session = load_session(session_target, runs_dir=runs_dir)

    fits = _collect_fits(session)
    not_fits = _collect_not_fits(session)
    # DISCARDs join FIT+NOT_FIT in ``rated_pks`` so the Phase 3 sampler
    # never re-evaluates them — they were invalidated in Phase 1
    # (issue #46), not "unseen".
    rated_pks = frozenset(
        [f.pk for f in fits]
        + [n.pk for n in not_fits]
        + list(_cumulative_discard_pks(session))
    )
    diagnoses = _collect_reflection_diagnoses(session)

    if len(fits) < 2:
        raise RefineLoadError(
            f"Session {session.session_id} has only {len(fits)} usable FIT "
            "exemplar(s); the meta-prompt needs at least 2 to name a "
            "discriminator. Continue Phase 1 tuning or rate more FITs."
        )

    query = session.turns[0].query if session.turns else ""
    return Phase3SessionInputs(
        session_id=session.session_id,
        query=query,
        fits=fits,
        not_fits=not_fits,
        rated_pks=rated_pks,
        reflection_diagnoses=diagnoses,
    )


def _collect_fits(session: ReplaySession) -> list[FitExemplar]:
    """Walk every turn and collect FIT-rated chunks with usable spans.

    Last-write-wins: a pk that flips FIT→NOT_FIT across turns drops
    out (the operator's later judgment supersedes the earlier one).
    """
    by_pk: dict[PrimaryKey, dict[str, Any]] = {}
    for turn in session.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "FIT":
                by_pk[row.pk] = {
                    "pk": row.pk,
                    "chunk_id": row.chunk_id,
                    "span_text": row.span_text or "",
                    "span_line_indices": list(row.span_line_indices or []),
                    "rating": "FIT",
                }
            elif row.rating == "NOT_FIT" and row.pk in by_pk:
                by_pk[row.pk]["rating"] = "NOT_FIT"
        for cands in turn.evidence_table.per_path_candidates.values():
            for cand in cands:
                if cand.rating == "FIT":
                    by_pk[cand.pk] = {
                        "pk": cand.pk,
                        "chunk_id": cand.chunk_id,
                        "span_text": cand.span_text or "",
                        "span_line_indices": list(cand.span_line_indices or []),
                        "rating": "FIT",
                    }
                elif cand.rating == "NOT_FIT" and cand.pk in by_pk:
                    by_pk[cand.pk]["rating"] = "NOT_FIT"

    out: list[FitExemplar] = []
    for entry in by_pk.values():
        if entry["rating"] != "FIT":
            continue
        if not entry["span_text"] or not entry["span_line_indices"]:
            log.warning(
                "Skipping FIT pk=%s (chunk_id=%s) — empty span",
                entry["pk"],
                entry["chunk_id"],
            )
            continue
        out.append(
            FitExemplar(
                pk=entry["pk"],
                chunk_id=entry["chunk_id"],
                span_text=entry["span_text"],
                span_line_indices=entry["span_line_indices"],
            )
        )
    return out


def _collect_not_fits(session: ReplaySession) -> list[NotFitExemplar]:
    """Walk every turn and collect NOT_FIT-rated chunks.

    Last-write-wins same as FITs. Drops rows without
    ``chunk_content`` (older logs) since the meta-prompt cannot cite
    what it cannot read.
    """
    by_pk: dict[PrimaryKey, dict[str, Any]] = {}
    for turn in session.turns:
        for row in turn.evidence_table.rows:
            if row.rating == "NOT_FIT":
                by_pk[row.pk] = {
                    "pk": row.pk,
                    "chunk_id": row.chunk_id,
                    "chunk_content": row.chunk_content or "",
                    "rating": "NOT_FIT",
                }
            elif row.rating == "FIT" and row.pk in by_pk:
                by_pk[row.pk]["rating"] = "FIT"
        for cands in turn.evidence_table.per_path_candidates.values():
            for cand in cands:
                if cand.rating == "NOT_FIT":
                    by_pk[cand.pk] = {
                        "pk": cand.pk,
                        "chunk_id": cand.chunk_id,
                        "chunk_content": cand.chunk_content or "",
                        "rating": "NOT_FIT",
                    }
                elif cand.rating == "FIT" and cand.pk in by_pk:
                    by_pk[cand.pk]["rating"] = "FIT"

    out: list[NotFitExemplar] = []
    for entry in by_pk.values():
        if entry["rating"] != "NOT_FIT":
            continue
        if not entry["chunk_content"]:
            log.warning(
                "Skipping NOT_FIT pk=%s (chunk_id=%s) — empty chunk_content",
                entry["pk"],
                entry["chunk_id"],
            )
            continue
        out.append(
            NotFitExemplar(
                pk=entry["pk"],
                chunk_id=entry["chunk_id"],
                chunk_content=entry["chunk_content"],
            )
        )
    return out


def _collect_reflection_diagnoses(session: ReplaySession) -> list[str]:
    """Concatenate non-empty `diagnose` strings from each turn's reflection."""
    out: list[str] = []
    for turn in session.turns:
        if not turn.reflection:
            continue
        diagnose = turn.reflection.get("diagnose")
        if isinstance(diagnose, str) and diagnose.strip():
            out.append(diagnose.strip())
    return out
