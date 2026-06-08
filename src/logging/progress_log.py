"""Per-session JSONL progress logger.

Writes two append-only files per session:

- ``runs/{session_id}.jsonl``: canonical progress log matching
  ``harness/schemas/progress_log.md`` field-for-field.
- ``runs/{session_id}.details.jsonl``: sidecar carrying full search
  and reflection diagnostics for auditability and debugging.

Both files are opened in append mode and flushed after every write so
a crash mid-session leaves readable, line-complete JSONL.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.search.config import SearchConfig
    from src.search.evidence import EvidenceTable
    from src.session.state import SessionState, TurnRecord

log = logging.getLogger(__name__)


class ProgressLogWriter:
    """JSONL writer for per-turn progress records."""

    def __init__(self, session_id: str, runs_dir: Path) -> None:
        self._session_id = session_id
        self._runs_dir = runs_dir
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._canonical_path = runs_dir / f"{session_id}.jsonl"
        self._details_path = runs_dir / f"{session_id}.details.jsonl"

    @property
    def canonical_path(self) -> Path:
        return self._canonical_path

    @property
    def details_path(self) -> Path:
        return self._details_path

    def log_turn(self, state: "SessionState", turn: "TurnRecord") -> None:
        try:
            self._write_canonical(state, turn)
        except Exception as exc:  # noqa: BLE001
            log.warning("canonical progress-log write failed: %s", exc)
        try:
            self._write_details(state, turn)
        except Exception as exc:  # noqa: BLE001
            log.warning("details progress-log write failed: %s", exc)
        # Auto-emit the convergence marker after each post-convergence
        # turn. Re-emitting (rather than once-per-session) keeps the
        # canonical jsonl's last line a converged event whenever the
        # session is in a converged state — the resume classifier in
        # src/web_api/resume.py reads the file's last line to decide
        # POST_TUNING in O(1).
        if state.is_converged:
            try:
                self.log_converged(turn=turn.turn_number)
            except Exception as exc:  # noqa: BLE001
                log.warning("converged-marker write failed: %s", exc)

    def log_converged(self, *, turn: int) -> None:
        """Append a single ``{"event": "converged", "turn", "ts"}`` row
        to the canonical jsonl.

        Idempotent at the row level — repeated calls just append more
        rows. The classifier in :mod:`src.web_api.resume` only inspects
        the file's last line, so duplicates do not affect routing.
        """
        entry: dict[str, Any] = {
            "event": "converged",
            "turn": turn,
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        _append_jsonl(self._canonical_path, entry)

    def log_event(self, *, turn: int, kind: str, **payload: Any) -> None:
        """Append a free-form event entry to ``.details.jsonl``.

        The canonical jsonl is one-record-per-turn with a fixed shape
        (matching ``harness/schemas/progress_log.md``); post-hoc
        events like ``path_drop_recommendation_decision`` go to the
        sidecar instead.
        """
        entry: dict[str, Any] = {
            "turn": turn,
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "session_id": self._session_id,
            "kind": kind,
            **payload,
        }
        try:
            _append_jsonl(self._details_path, entry)
        except Exception as exc:  # noqa: BLE001
            log.warning("details event write failed: %s", exc)

    def _write_canonical(self, state: "SessionState", turn: "TurnRecord") -> None:
        entry = {
            "turn": turn.turn_number,
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "session_id": state.session_id,
            "scope": state.scope,
            "query": turn.query,
            "config": _config_to_log_shape(turn.config),
            "audit_turn": turn.audit_turn,
            "results": _results_from_table(turn.evidence_table),
            "per_path_candidates": _candidates_from_table(turn.evidence_table),
            "metrics": _metrics_from_turn(turn),
            "progress_state": state.progress_state,
            "reflection": _reflection_for_canonical(turn.reflection),
        }
        _append_jsonl(self._canonical_path, entry)

    def _write_details(self, state: "SessionState", turn: "TurnRecord") -> None:
        reflection = turn.reflection or {}
        diagnostics = reflection.get("_diagnostics")
        entry = {
            "turn": turn.turn_number,
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "session_id": state.session_id,
            "query": turn.query,
            "search": _search_diagnostics(turn.evidence_table),
            "reflection": diagnostics,
            "config_diff": (
                diagnostics.get("validation", {}).get("config_diff")
                if isinstance(diagnostics, dict) and diagnostics.get("validation")
                else None
            ),
        }
        _append_jsonl(self._details_path, entry)


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False, default=_json_default))
        fp.write("\n")
        fp.flush()


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _config_to_log_shape(config: "SearchConfig") -> dict[str, Any]:
    """Flat shape matching progress_log.md; fusion is always RRFRanker."""
    return {
        "rrf_k": config.rrf_k,
        "per_path_limit": config.per_path_limit,
        "top_k": config.top_k,
        "active_paths": sorted(config.active_paths),
    }


def _results_from_table(table: "EvidenceTable") -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in table.rows:
        results.append(
            {
                "rank": row.rank,
                "pk": _json_safe_pk(row.pk),
                "chunk_id": row.chunk_id,
                "chunk_content": row.chunk_content,
                "sample_id": row.sample_id,
                "counselor_id": row.counselor_id,
                "term": row.term,
                "rating": row.rating,
                "source_paths": list(row.source_paths),
                "scores": {
                    "dense": row.scores.get("dense", 0.0),
                    "sparse": row.scores.get("sparse", 0.0),
                },
                "span_line_indices": list(row.span_line_indices),
                "span_text": row.span_text,
            }
        )
    return results


def _candidates_from_table(table: "EvidenceTable") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"dense": [], "sparse": []}
    for path in ("dense", "sparse"):
        for cand in table.per_path_candidates.get(path, []):
            out[path].append(
                {
                    "rank_in_path": cand.rank_in_path,
                    "pk": _json_safe_pk(cand.pk),
                    "chunk_id": cand.chunk_id,
                    "chunk_content": cand.chunk_content,
                    "sample_id": cand.sample_id,
                    "counselor_id": cand.counselor_id,
                    "term": cand.term,
                    "score": cand.score,
                    "rating": cand.rating,
                    "span_line_indices": list(cand.span_line_indices),
                    "span_text": cand.span_text,
                }
            )
    return out


def _metrics_from_turn(turn: "TurnRecord") -> dict[str, Any]:
    total = len(turn.evidence_table.rows)
    fit_count = sum(1 for r in turn.evidence_table.rows if r.rating == "FIT")
    not_fit_count = sum(1 for r in turn.evidence_table.rows if r.rating == "NOT_FIT")
    discard_count = sum(1 for r in turn.evidence_table.rows if r.rating == "DISCARD")
    diagnostics = turn.evidence_table.search_diagnostics or {}
    return {
        "total": total,
        "fit_count": fit_count,
        "not_fit_count": not_fit_count,
        "discard_count": discard_count,
        "precision_at_k": turn.precision,
        "per_path_breakdown": turn.breakdown,
        "seen_set_size": int(diagnostics.get("seen_set_size", 0)),
    }


def _reflection_for_canonical(
    reflection: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip the leading-underscore ``_diagnostics`` key for the canonical log."""
    if reflection is None:
        return None
    return {k: v for k, v in reflection.items() if not k.startswith("_")}


def _search_diagnostics(table: "EvidenceTable") -> dict[str, Any] | None:
    return table.search_diagnostics


def _json_safe_pk(pk: Any) -> Any:
    if isinstance(pk, (int, str)):
        return pk
    return str(pk)
