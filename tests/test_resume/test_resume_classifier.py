"""Exhaustive coverage of ``src.web_api.resume.classify``.

The classifier is the source of truth for which on-disk states map
to which resume target. Each branch (and the abandoned fallthrough)
gets a focused test; ordering tests verify the priority chain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.web_api.resume import ResumeTarget, classify


def _write(path: Path, content: str | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        path.write_text(json.dumps(content) + "\n", encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")


def _converged_row(turn: int = 3) -> str:
    return (
        json.dumps({"event": "converged", "turn": turn, "ts": "2026-01-01T00:00:00Z"})
        + "\n"
    )


def _turn_row(turn: int = 1) -> str:
    return (
        json.dumps({"turn": turn, "session_id": "abcd", "query": "q"})
        + "\n"
    )


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    user_dir = tmp_path / "alice"
    user_dir.mkdir()
    return user_dir


# ---------------------------------------------------------------------------
# Per-branch
# ---------------------------------------------------------------------------


def test_classify_done_view_when_phase3_meta_says_agree(runs_dir: Path) -> None:
    sid = "abcd"
    _write(
        runs_dir / f"{sid}.phase3.meta.json", {"operator_decision": "agree"}
    )
    # Stage A files coexist after finalise — the priority chain still
    # picks DONE_VIEW because meta.json wins.
    _write(runs_dir / f"{sid}.phase3.rubric.json", {"version": 1})
    _write(runs_dir / f"{sid}.phase3.evidence.jsonl", "{}\n")
    # Phase 4 labels present → the session is fully finalised, not
    # APPLY_PENDING. Without this the classifier returns APPLY_PENDING
    # (the additive Phase 4 cohort apply step is still available).
    _write(runs_dir / f"{sid}.phase4.labels.jsonl", "{}\n")
    assert classify(sid, runs_dir) == ResumeTarget.DONE_VIEW


def test_classify_post_rubric_when_stage_a_files_present_without_meta(
    runs_dir: Path,
) -> None:
    sid = "abcd"
    _write(runs_dir / f"{sid}.phase3.rubric.json", {"version": 1})
    _write(runs_dir / f"{sid}.phase3.evidence.jsonl", "{}\n")
    # phase2 files might exist too (finalise hasn't fired yet).
    _write(runs_dir / f"{sid}.phase2.meta.json", {"verdict": "PASSED"})
    assert classify(sid, runs_dir) == ResumeTarget.POST_RUBRIC


def test_classify_post_harvest_when_only_phase2_meta(runs_dir: Path) -> None:
    sid = "abcd"
    _write(runs_dir / f"{sid}.phase2.meta.json", {"verdict": "PASSED"})
    _write(runs_dir / f"{sid}.jsonl", _converged_row())
    assert classify(sid, runs_dir) == ResumeTarget.POST_HARVEST


def test_classify_post_tuning_when_canonical_last_line_is_converged(
    runs_dir: Path,
) -> None:
    sid = "abcd"
    _write(
        runs_dir / f"{sid}.jsonl", _turn_row(1) + _turn_row(2) + _converged_row(2)
    )
    assert classify(sid, runs_dir) == ResumeTarget.POST_TUNING


def test_classify_post_tuning_when_marker_re_emits_after_extra_turn(
    runs_dir: Path,
) -> None:
    """Layout: turn1, marker, turn2 (post-convergence), marker. The
    classifier must still recognise the file as POST_TUNING because
    the marker is the last line."""
    sid = "abcd"
    _write(
        runs_dir / f"{sid}.jsonl",
        _turn_row(1) + _converged_row(1) + _turn_row(2) + _converged_row(2),
    )
    assert classify(sid, runs_dir) == ResumeTarget.POST_TUNING


def test_classify_abandoned_when_no_marker_and_no_phase_sidecars(
    runs_dir: Path,
) -> None:
    sid = "abcd"
    _write(runs_dir / f"{sid}.jsonl", _turn_row(1) + _turn_row(2))
    assert classify(sid, runs_dir) is None


def test_classify_abandoned_when_user_dir_empty(runs_dir: Path) -> None:
    assert classify("ghost", runs_dir) is None


def test_classify_abandoned_when_user_dir_missing(tmp_path: Path) -> None:
    """A user with no sessions yet (no runs/<user_id>/ dir at all) is
    handled gracefully — listing such a user just returns []."""
    assert classify("ghost", tmp_path / "noone") is None


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_priority_done_view_beats_post_rubric(runs_dir: Path) -> None:
    sid = "abcd"
    _write(runs_dir / f"{sid}.phase3.rubric.json", {"version": 1})
    _write(runs_dir / f"{sid}.phase3.evidence.jsonl", "{}\n")
    _write(
        runs_dir / f"{sid}.phase3.meta.json", {"operator_decision": "agree"}
    )
    # Phase 4 labels present → DONE_VIEW wins over POST_RUBRIC and
    # APPLY_PENDING.
    _write(runs_dir / f"{sid}.phase4.labels.jsonl", "{}\n")
    assert classify(sid, runs_dir) == ResumeTarget.DONE_VIEW


def test_priority_post_rubric_beats_post_harvest(runs_dir: Path) -> None:
    sid = "abcd"
    _write(runs_dir / f"{sid}.phase2.meta.json", {"verdict": "PASSED"})
    _write(runs_dir / f"{sid}.phase3.rubric.json", {"version": 1})
    _write(runs_dir / f"{sid}.phase3.evidence.jsonl", "{}\n")
    assert classify(sid, runs_dir) == ResumeTarget.POST_RUBRIC


def test_priority_post_harvest_beats_post_tuning(runs_dir: Path) -> None:
    sid = "abcd"
    _write(runs_dir / f"{sid}.jsonl", _converged_row())
    _write(runs_dir / f"{sid}.phase2.meta.json", {"verdict": "PASSED"})
    assert classify(sid, runs_dir) == ResumeTarget.POST_HARVEST


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_phase3_meta_with_decision_other_than_agree_does_not_pick_done_view(
    runs_dir: Path,
) -> None:
    """Future audit-trail abort modes might write meta.json with a
    different ``operator_decision``. Only ``"agree"`` qualifies as
    finalised — anything else falls through to the next branch."""
    sid = "abcd"
    _write(runs_dir / f"{sid}.phase3.meta.json", {"operator_decision": "abort"})
    _write(runs_dir / f"{sid}.phase3.rubric.json", {"version": 1})
    _write(runs_dir / f"{sid}.phase3.evidence.jsonl", "{}\n")
    assert classify(sid, runs_dir) == ResumeTarget.POST_RUBRIC


def test_corrupt_phase3_meta_falls_through_gracefully(runs_dir: Path) -> None:
    """A malformed phase3.meta.json must not raise; we fall back to the
    next branch (POST_RUBRIC if the stage A pair exists)."""
    sid = "abcd"
    (runs_dir / f"{sid}.phase3.meta.json").write_text(
        "{not json", encoding="utf-8"
    )
    _write(runs_dir / f"{sid}.phase3.rubric.json", {"version": 1})
    _write(runs_dir / f"{sid}.phase3.evidence.jsonl", "{}\n")
    assert classify(sid, runs_dir) == ResumeTarget.POST_RUBRIC


def test_canonical_with_corrupt_last_line_does_not_match_marker(
    runs_dir: Path,
) -> None:
    sid = "abcd"
    (runs_dir / f"{sid}.jsonl").write_text(
        _turn_row(1) + "{not json}\n", encoding="utf-8"
    )
    assert classify(sid, runs_dir) is None


def test_classifier_handles_large_canonical_via_tail_window(
    runs_dir: Path,
) -> None:
    """A canonical JSONL larger than the tail window must still pick up
    the converged marker on its last line. Build a file with many
    padding turns + the marker; classifier reads only the tail."""
    sid = "abcd"
    padding = "".join(_turn_row(i) for i in range(1, 500))
    (runs_dir / f"{sid}.jsonl").write_text(
        padding + _converged_row(499), encoding="utf-8"
    )
    assert classify(sid, runs_dir) == ResumeTarget.POST_TUNING
