"""Reconstruct a past session from its JSONL files.

The canonical log (`runs/<id>.jsonl`) carries the per-turn config,
fused results with chunk_content, per-path candidates, metrics, and
the agent's reflection block. The sidecar (`runs/<id>.details.jsonl`)
carries the per-path probe stats (hit counts, score ranges, top-3
entities) needed to repopulate the probe banner.

Joining is by `turn` number. Sessions logged before the writer began
persisting `chunk_content` (pre-replay-mode) cannot be replayed
faithfully — the loader raises ``ReplayLoadError`` in that case rather
than silently rendering empty cards.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.search.config import SearchConfig
from src.search.evidence import (
    ALL_PATHS,
    CandidateRow,
    EvidenceRow,
    EvidenceTable,
    PathName,
)
from src.search.search import ProbeResult

log = logging.getLogger(__name__)


class ReplayLoadError(Exception):
    """Raised when a session's logs are missing, malformed, or pre-replay-era."""


@dataclass(frozen=True)
class ReplayTurn:
    """One turn reconstructed from the JSONL pair, ready to feed the TUI."""

    turn_number: int
    timestamp: str
    query: str
    config: SearchConfig
    evidence_table: EvidenceTable
    breakdown: dict[str, dict[str, int]]
    precision: float
    reflection: dict[str, Any] | None
    probe: ProbeResult | None


@dataclass(frozen=True)
class ReplaySession:
    """All turns of one past session, ordered by turn number.

    The ``load_phase{2,3}_*`` accessors below read sidecars that this
    loader does not parse during ``load_session`` — they exist so the
    resume-hydration path can rebuild Phase 2 / Phase 3 state from disk
    without re-implementing JSON I/O. Each raises
    :class:`ReplayLoadError` when its sidecar is missing or malformed,
    matching the rest of the loader's contract.
    """

    session_id: str
    canonical_path: Path
    details_path: Path
    turns: list[ReplayTurn] = field(default_factory=list)

    def _sidecar_path(self, suffix: str) -> Path:
        return self.canonical_path.parent / f"{self.session_id}{suffix}"

    @property
    def phase2_meta_path(self) -> Path:
        return self._sidecar_path(".phase2.meta.json")

    @property
    def phase2_jsonl_path(self) -> Path:
        return self._sidecar_path(".phase2.jsonl")

    @property
    def phase3_rubric_path(self) -> Path:
        return self._sidecar_path(".phase3.rubric.json")

    @property
    def phase3_evidence_path(self) -> Path:
        return self._sidecar_path(".phase3.evidence.jsonl")

    @property
    def phase3_meta_path(self) -> Path:
        return self._sidecar_path(".phase3.meta.json")

    @property
    def phase3_prompt_path(self) -> Path:
        return self._sidecar_path(".phase3.prompt.md")

    def load_phase2_meta(self) -> dict[str, Any]:
        return _load_json_object(self.phase2_meta_path)

    def load_phase2_kept(self) -> list[dict[str, Any]]:
        path = self.phase2_jsonl_path
        if not path.exists():
            raise ReplayLoadError(f"Phase 2 jsonl sidecar missing: {path}")
        return _read_jsonl(path)

    def load_phase3_rubric(self) -> dict[str, Any]:
        return _load_json_object(self.phase3_rubric_path)

    def load_phase3_evidence(self) -> list[dict[str, Any]]:
        path = self.phase3_evidence_path
        if not path.exists():
            raise ReplayLoadError(f"Phase 3 evidence sidecar missing: {path}")
        return _read_jsonl(path)

    def load_phase3_meta(self) -> dict[str, Any]:
        return _load_json_object(self.phase3_meta_path)

    def load_phase3_prompt(self) -> str:
        path = self.phase3_prompt_path
        if not path.exists():
            raise ReplayLoadError(f"Phase 3 prompt sidecar missing: {path}")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReplayLoadError(
                f"Could not read Phase 3 prompt {path}: {exc}"
            ) from exc


def load_session(target: str | Path, runs_dir: Path | None = None) -> ReplaySession:
    """Load a session for replay.

    ``target`` accepts a session id (``"918aa8240313"``) or a path to the
    canonical ``.jsonl`` file. ``runs_dir`` defaults to ``./runs`` and is
    only consulted when ``target`` is a bare session id.
    """

    canonical, details = _resolve_paths(target, runs_dir)
    if not canonical.exists():
        raise ReplayLoadError(f"Canonical log not found: {canonical}")
    if not details.exists():
        raise ReplayLoadError(f"Details sidecar not found: {details}")

    canonical_entries = _read_jsonl(canonical)
    details_entries = _read_jsonl(details)

    if not canonical_entries:
        raise ReplayLoadError(f"Canonical log is empty: {canonical}")

    details_by_turn: dict[int, dict[str, Any]] = {}
    for entry in details_entries:
        turn = entry.get("turn")
        if isinstance(turn, int):
            details_by_turn[turn] = entry

    session_id = str(canonical_entries[0].get("session_id", canonical.stem))
    turns: list[ReplayTurn] = []
    for entry in canonical_entries:
        # Skip event rows (e.g. ``{"event": "converged", ...}``) — they
        # are session-scoped markers, not turn-shaped objects, and
        # ``_build_turn`` would reject them.
        if "event" in entry:
            continue
        turn_no = entry.get("turn")
        if not isinstance(turn_no, int):
            raise ReplayLoadError(f"Canonical entry missing integer 'turn': {entry!r}")
        details_entry = details_by_turn.get(turn_no)
        turns.append(_build_turn(entry, details_entry))

    return ReplaySession(
        session_id=session_id,
        canonical_path=canonical,
        details_path=details,
        turns=turns,
    )


def _resolve_paths(target: str | Path, runs_dir: Path | None) -> tuple[Path, Path]:
    target_path = Path(target)
    if target_path.suffix == ".jsonl" and target_path.name.endswith(".jsonl"):
        if target_path.name.endswith(".details.jsonl"):
            canonical = target_path.with_name(
                target_path.name.replace(".details.jsonl", ".jsonl")
            )
            details = target_path
        else:
            canonical = target_path
            details = target_path.with_name(target_path.stem + ".details.jsonl")
        return canonical, details

    base = runs_dir or Path("runs")
    return base / f"{target}.jsonl", base / f"{target}.details.jsonl"


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON file expected to contain a single object.

    Used by the ReplaySession sidecar accessors so each path produces a
    consistent ``ReplayLoadError`` on missing/malformed input.
    """
    if not path.exists():
        raise ReplayLoadError(f"Sidecar missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayLoadError(f"Sidecar {path} unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayLoadError(
            f"Sidecar {path} expected a JSON object, got {type(payload).__name__}"
        )
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReplayLoadError(
                    f"Malformed JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(entry, dict):
                raise ReplayLoadError(
                    f"Expected JSON object at {path}:{line_no}, got {type(entry).__name__}"
                )
            entries.append(entry)
    return entries


def _build_turn(
    canonical: dict[str, Any], details: dict[str, Any] | None
) -> ReplayTurn:
    turn_no = int(canonical["turn"])
    query = str(canonical.get("query", ""))
    config = _build_config(canonical.get("config") or {}, turn_no)

    rows = _build_rows(canonical.get("results") or [], turn_no)
    candidates = _build_candidates(canonical.get("per_path_candidates") or {}, turn_no)
    table = EvidenceTable(
        query=query,
        config=config,
        rows=rows,
        per_path_candidates=candidates,
    )

    metrics = canonical.get("metrics") or {}
    breakdown = metrics.get("per_path_breakdown") or _empty_breakdown()
    precision = float(metrics.get("precision_at_k", 0.0))

    reflection = _reflection_with_diagnostics(
        canonical.get("reflection"),
        details,
    )

    probe = _build_probe(query, config, details) if details else None

    return ReplayTurn(
        turn_number=turn_no,
        timestamp=str(canonical.get("timestamp", "")),
        query=query,
        config=config,
        evidence_table=table,
        breakdown=breakdown,
        precision=precision,
        reflection=reflection,
        probe=probe,
    )


def _build_config(raw: dict[str, Any], turn_no: int) -> SearchConfig:
    raw_paths = raw.get("active_paths") or []
    if "bm25" in raw_paths:
        raise ReplayLoadError(_legacy_bm25_message(turn_no, "active_paths"))
    try:
        return SearchConfig(
            rrf_k=int(raw["rrf_k"]),
            per_path_limit=int(raw["per_path_limit"]),
            top_k=int(raw["top_k"]),
            active_paths=frozenset(raw_paths or ALL_PATHS),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayLoadError(
            f"Turn {turn_no}: invalid config block ({exc}): {raw!r}"
        ) from exc


def _build_rows(raw_results: list[dict[str, Any]], turn_no: int) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for entry in raw_results:
        if "chunk_content" not in entry:
            raise ReplayLoadError(
                f"Turn {turn_no}: result row missing 'chunk_content' "
                "(session predates the replay-era writer fix and cannot be replayed)"
            )
        scores = entry.get("scores") or {}
        if "bm25" in scores:
            raise ReplayLoadError(_legacy_bm25_message(turn_no, "scores.bm25"))
        rows.append(
            EvidenceRow(
                rank=int(entry["rank"]),
                pk=entry["pk"],
                chunk_id=str(entry.get("chunk_id", "")),
                chunk_content=str(entry["chunk_content"]),
                sample_id=str(entry.get("sample_id", "")),
                counselor_id=str(entry.get("counselor_id", "")),
                term=str(entry.get("term", "")),
                source_paths=list(entry.get("source_paths") or []),
                scores={
                    "dense": float(scores.get("dense", 0.0)),
                    "sparse": float(scores.get("sparse", 0.0)),
                },
                rating=entry.get("rating"),
                span_line_indices=[
                    int(i) for i in entry.get("span_line_indices") or []
                ],
                span_text=str(entry.get("span_text", "")),
            )
        )
    return rows


def _build_candidates(
    raw: dict[str, Any], turn_no: int
) -> dict[PathName, list[CandidateRow]]:
    if "bm25" in raw:
        raise ReplayLoadError(_legacy_bm25_message(turn_no, "per_path_candidates.bm25"))
    out: dict[PathName, list[CandidateRow]] = {p: [] for p in ALL_PATHS}
    for path in ALL_PATHS:
        for entry in raw.get(path) or []:
            if "chunk_content" not in entry:
                raise ReplayLoadError(
                    f"Turn {turn_no}: per-path candidate (path={path}) missing "
                    "'chunk_content' (session predates the replay-era writer fix)"
                )
            out[path].append(
                CandidateRow(
                    path=path,
                    rank_in_path=int(entry["rank_in_path"]),
                    pk=entry["pk"],
                    chunk_id=str(entry.get("chunk_id", "")),
                    chunk_content=str(entry["chunk_content"]),
                    sample_id=str(entry.get("sample_id", "")),
                    counselor_id=str(entry.get("counselor_id", "")),
                    term=str(entry.get("term", "")),
                    score=float(entry.get("score", 0.0)),
                    rating=entry.get("rating"),
                    span_line_indices=[
                        int(i) for i in entry.get("span_line_indices") or []
                    ],
                    span_text=str(entry.get("span_text", "")),
                )
            )
    return out


def _empty_breakdown() -> dict[str, dict[str, int]]:
    return {
        bucket: {"total": 0, "fit": 0, "not_fit": 0, "discard": 0}
        for bucket in ("dense_only", "sparse_only", "multi_path")
    }


_LEGACY_BM25_CUTOVER_SHA = "7237be0"


def _legacy_bm25_message(turn_no: int, field: str) -> str:
    """One-line guard message for pre-removal logs.

    BM25 was dropped from Phase 1 after commit ``7237be0``; old logs
    that name ``bm25`` in any path-keyed field can no longer be
    replayed by current builds. The SHA is referenced verbatim so
    operators can git-checkout the cutover point.
    """
    return (
        f"Turn {turn_no}: legacy 'bm25' in '{field}' is no longer supported "
        f"by Phase 1; replay this session with a build at or before commit "
        f"{_LEGACY_BM25_CUTOVER_SHA}."
    )


def _reflection_with_diagnostics(
    canonical_reflection: Any,
    details: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(canonical_reflection, dict):
        return None
    merged = dict(canonical_reflection)
    if details is None:
        return merged
    diag = details.get("reflection")
    if isinstance(diag, dict):
        # The modal reads ``raw_response`` from ``_diagnostics``; preserve
        # the rest of the diagnostic envelope for parity with live runs.
        merged["_diagnostics"] = diag
    return merged


def _build_probe(
    query: str, config: SearchConfig, details: dict[str, Any]
) -> ProbeResult | None:
    search = details.get("search") or {}
    probes = search.get("probes")
    if not isinstance(probes, dict):
        return None
    stats_by_path: dict[PathName, dict[str, Any]] = {}
    for path in ALL_PATHS:
        path_stats = probes.get(path)
        if isinstance(path_stats, dict):
            stats_by_path[path] = dict(path_stats)
    return ProbeResult(
        query=query,
        config=config,
        dense_vec=[],
        sparse_vec={},
        provenance={},
        scores_by_path={p: {} for p in ALL_PATHS},
        stats_by_path=stats_by_path,
        embed_diagnostics={},
    )
