"""Tests for prompt assembly."""

from __future__ import annotations

from src.reflection.prompt import (
    build_messages,
    _format_config,
    _render_per_path_candidates,
    _render_probe_stats,
    _render_turn_full,
    _render_turn_summary,
)
from src.search.evidence import CandidateRow

from .conftest import make_config, make_state, make_table, make_turn


class TestBuildMessages:
    def test_single_turn_structure(self) -> None:
        turn = make_turn(1)
        state = make_state(turns=[turn])
        msgs = build_messages(state, "system prompt", "instructions")

        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "system prompt"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == "instructions"

    def test_single_turn_has_no_previous_turns(self) -> None:
        turn = make_turn(1)
        state = make_state(turns=[turn])
        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]
        assert "No previous turns" in context

    def test_multi_turn_all_in_full(self) -> None:
        turns = [make_turn(i, precision=0.3 + i * 0.1) for i in range(1, 5)]
        state = make_state(turns=turns)
        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        # All previous turns (1-3) should be in full detail
        assert "Turn 1:" in context
        assert "Turn 2:" in context
        assert "Turn 3:" in context
        # Current turn (4) in evidence section
        assert "CURRENT TURN (Turn 4)" in context

    def test_compression_for_long_sessions(self) -> None:
        # 10 turns total — turns 1-4 compressed, 5-9 full, turn 10 current
        turns = [make_turn(i, precision=0.3 + i * 0.02) for i in range(1, 11)]
        state = make_state(turns=turns)
        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        # Early turns should be summary format (one-liners without "Query:")
        for i in range(1, 5):
            assert f"Turn {i}: Config=" in context

        # Recent turns should have full format with "Query:"
        for i in range(5, 10):
            assert f"Turn {i}:" in context

        # Current turn
        assert "CURRENT TURN (Turn 10)" in context

    def test_evidence_table_rows(self) -> None:
        turn = make_turn(1)
        state = make_state(turns=[turn])
        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        # Evidence table should have header + 3 rows (from make_table)
        assert "| rank |" in context
        assert "sample_001" in context

    def test_precision_delta_on_second_turn(self) -> None:
        t1 = make_turn(1, precision=0.3)
        t2 = make_turn(2, precision=0.5)
        state = make_state(turns=[t1, t2])
        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "previous turn: 0.30" in context
        assert "+0.20" in context
        assert "improved" in context


class TestProbeStatsRendering:
    def _diag(self, **probes: dict) -> dict:
        return {"probes": probes}

    def test_omits_section_when_no_diagnostics(self) -> None:
        assert _render_probe_stats(None) == ""
        assert _render_probe_stats({}) == ""
        assert _render_probe_stats({"other_key": 1}) == ""

    def test_renders_active_paths(self) -> None:
        diag = self._diag(
            dense={
                "skipped": False,
                "hit_count": 20,
                "score_min": 0.781,
                "score_max": 0.801,
                "score_mean": 0.792,
                "top3_pks": [],
                "latency_ms": 1.0,
            },
            sparse={
                "skipped": False,
                "hit_count": 20,
                "score_min": 0.204,
                "score_max": 0.217,
                "score_mean": 0.211,
                "top3_pks": [],
                "latency_ms": 1.0,
            },
        )

        rendered = _render_probe_stats(diag)

        assert "Per-path probe" in rendered
        assert "hits=20" in rendered
        assert "0.781" in rendered
        assert "mean=0.792" in rendered

    def test_renders_skipped_path(self) -> None:
        diag = self._diag(
            dense={
                "skipped": False,
                "hit_count": 5,
                "score_min": 0.5,
                "score_max": 0.6,
                "score_mean": 0.55,
            },
            sparse={"skipped": True, "hit_count": 0},
        )

        rendered = _render_probe_stats(diag)

        assert "skipped" in rendered

    def test_evidence_table_with_diagnostics_includes_probe_block(self) -> None:
        diag = {
            "probes": {
                "dense": {
                    "skipped": False,
                    "hit_count": 10,
                    "score_min": 0.7,
                    "score_max": 0.8,
                    "score_mean": 0.75,
                    "top3_pks": [],
                },
                "sparse": {
                    "skipped": False,
                    "hit_count": 5,
                    "score_min": 0.2,
                    "score_max": 0.3,
                    "score_mean": 0.25,
                    "top3_pks": [],
                },
            }
        }
        table = make_table()
        table.search_diagnostics = diag
        # build_messages needs a turn whose evidence_table carries the diag.
        turn = make_turn(1)
        turn.evidence_table.search_diagnostics = diag
        state = make_state(turns=[turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Per-path probe" in context
        assert "hits=10" in context

    def test_evidence_table_without_diagnostics_omits_probe_block(self) -> None:
        # Default make_turn fixture produces an EvidenceTable with no diagnostics.
        turn = make_turn(1)
        assert turn.evidence_table.search_diagnostics is None
        state = make_state(turns=[turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Per-path probe" not in context


class TestPerPathCandidatesRendering:
    def _candidate(
        self,
        path: str,
        rank: int,
        rating: str | None,
        chunk_id: str | None = None,
    ) -> CandidateRow:
        return CandidateRow(
            path=path,  # type: ignore[arg-type]
            rank_in_path=rank,
            pk=rank * 100,
            chunk_id=chunk_id or f"S1_C{rank * 100:07d}",
            chunk_content="x",
            sample_id="S1",
            counselor_id="T1",
            term="T",
            score=0.5,
            rating=rating,  # type: ignore[arg-type]
        )

    def test_omits_block_when_no_candidates(self) -> None:
        table = make_table()
        # Default fixture has no per_path_candidates populated.
        rendered = _render_per_path_candidates(table)
        assert rendered == ""

    def test_renders_active_paths_with_summary(self) -> None:
        table = make_table()
        table.per_path_candidates = {
            "dense": [],
            "sparse": [
                self._candidate("sparse", 1, "FIT", "S1_C1"),
                self._candidate("sparse", 2, "NOT_FIT", "S1_C2"),
                self._candidate("sparse", 3, "FIT", "S1_C3"),
            ],
        }

        rendered = _render_per_path_candidates(table)

        assert "Per-path top-3 candidates" in rendered
        assert "sparse" in rendered
        assert "2/3 FIT" in rendered
        assert "S1_C1" in rendered and "FIT" in rendered
        # dense path gets the "(none …)" placeholder.
        assert rendered.count("(none") == 1

    def test_renders_with_unrated_count(self) -> None:
        table = make_table()
        table.per_path_candidates = {
            "dense": [self._candidate("dense", 1, None)],
            "sparse": [],
        }
        rendered = _render_per_path_candidates(table)
        assert "0/1 FIT" in rendered
        assert "(1 unrated)" in rendered

    def test_build_messages_includes_candidates_block(self) -> None:
        turn = make_turn(1)
        turn.evidence_table.per_path_candidates = {
            "dense": [],
            "sparse": [self._candidate("sparse", 1, "FIT")],
        }
        state = make_state(turns=[turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Per-path top-3 candidates" in context

    def test_labels_inactive_path(self) -> None:
        cfg = make_config(active_paths=frozenset({"dense"}))
        table = make_table(config=cfg)
        table.per_path_candidates = {
            "dense": [self._candidate("dense", 1, "FIT")],
            "sparse": [self._candidate("sparse", 1, "NOT_FIT")],
        }

        rendered = _render_per_path_candidates(table)

        assert "sparse " in rendered
        assert "(inactive)" in rendered
        # dense is active → no (inactive) label on its line
        dense_line = [ln for ln in rendered.splitlines() if "dense" in ln][0]
        assert "(inactive)" not in dense_line


class TestSeenSetRendering:
    """The reflection prompt surfaces the seen-set size actually filtered
    at search time so the agent can distinguish exhausted paths from dead
    paths. The source is ``evidence_table.search_diagnostics``, not
    ``state.seen_pks`` (which grows after ratings land)."""

    def test_renders_seen_set_line_when_nonzero(self) -> None:
        turn = make_turn(1)
        turn.evidence_table.search_diagnostics = {"seen_set_size": 3}
        state = make_state(turns=[turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Seen set: 3 chunks excluded" in context

    def test_omits_seen_set_line_when_zero(self) -> None:
        turn = make_turn(1)
        state = make_state(turns=[turn])
        # No diagnostics on turn 1 — no filter was applied.

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Seen set" not in context

    def test_omits_seen_set_line_on_turn_1_despite_post_turn_seen_pks(self) -> None:
        """Regression: `state.seen_pks` grows during `complete_turn` before
        reflection runs. The prompt must NOT claim exclusion on turn 1
        just because the state's seen-set has accumulated this turn's
        ratings — the turn's search itself ran unfiltered."""
        turn = make_turn(1)
        # Diagnostics reflect what was actually filtered at search time: 0.
        turn.evidence_table.search_diagnostics = {"seen_set_size": 0}
        state = make_state(turns=[turn])
        # Post-turn state: the rated PKs have landed in `seen_pks`.
        state.seen_pks = {"pk-1", "pk-2", "pk-3"}

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Seen set" not in context

    def test_seen_set_line_placed_after_config(self) -> None:
        turn = make_turn(1)
        turn.evidence_table.search_diagnostics = {"seen_set_size": 2}
        state = make_state(turns=[turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        config_idx = context.index("Config used:")
        seen_idx = context.index("Seen set:")
        results_idx = context.index("Results with human ratings:")
        assert config_idx < seen_idx < results_idx

    def test_probe_path_marks_exhausted_when_filtered(self) -> None:
        from src.reflection.prompt import _render_probe_path

        stats = {
            "skipped": False,
            "hit_count": 0,
            "filtered_by_seen": 47,
        }
        rendered = _render_probe_path(stats)
        assert "seen=47 filtered" in rendered
        assert "exhausted" in rendered

    def test_probe_path_0_hits_without_filter_unchanged(self) -> None:
        from src.reflection.prompt import _render_probe_path

        stats = {"skipped": False, "hit_count": 0, "filtered_by_seen": 0}
        rendered = _render_probe_path(stats)
        assert rendered == "0 hits"


class TestDepthAccounting:
    """After dedup kicks in (turn 2+), the prompt must expose the per-path
    new-candidate arithmetic so reflection can reason about depth dynamics
    rather than invent them. Numbers come from ``search_diagnostics``."""

    def _diag(
        self,
        seen_set_size: int,
        per_path_limit: int = 20,
        dense: dict | None = None,
        sparse: dict | None = None,
    ) -> dict:
        def _probe(spec: dict | None) -> dict:
            base = {
                "skipped": False,
                "hit_count": 0,
                "filtered_by_seen": 0,
                "score_min": None,
                "score_max": None,
                "score_mean": None,
            }
            if spec:
                base.update(spec)
            # Populate plausible scores when the probe actually returned
            # hits — `_render_probe_path` assumes scores are not None in
            # that branch.
            if base["hit_count"] > 0 and base["score_min"] is None:
                base["score_min"] = 0.5
                base["score_max"] = 0.7
                base["score_mean"] = 0.6
            return base

        return {
            "seen_set_size": seen_set_size,
            "per_path_limit": per_path_limit,
            "probes": {
                "dense": _probe(dense),
                "sparse": _probe(sparse),
            },
        }

    def test_renders_block_when_seen_set_nonzero(self) -> None:
        turn = make_turn(2)
        turn.evidence_table.search_diagnostics = self._diag(
            seen_set_size=10,
            per_path_limit=20,
            dense={"hit_count": 10, "filtered_by_seen": 10},
            sparse={"hit_count": 10, "filtered_by_seen": 10},
        )
        state = make_state(turns=[make_turn(1), turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Depth accounting" in context
        # Active path with new material at tier 2.
        assert "per_path_limit=20" in context
        assert "filtered=10" in context
        assert "new=10 at ranks 11–20" in context

    def test_omits_block_when_seen_set_zero(self) -> None:
        turn = make_turn(1)
        turn.evidence_table.search_diagnostics = self._diag(
            seen_set_size=0,
            per_path_limit=20,
            dense={"hit_count": 20, "filtered_by_seen": 0},
            sparse={"hit_count": 20, "filtered_by_seen": 0},
        )
        state = make_state(turns=[turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Depth accounting" not in context

    def test_renders_exhausted_path(self) -> None:
        turn = make_turn(3)
        turn.evidence_table.search_diagnostics = self._diag(
            seen_set_size=40,
            per_path_limit=20,
            dense={"hit_count": 0, "filtered_by_seen": 20},
            sparse={"hit_count": 10, "filtered_by_seen": 10},
        )
        state = make_state(turns=[make_turn(1), make_turn(2), turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Depth accounting" in context
        # Exhausted path annotation.
        assert "exhausted" in context

    def test_renders_empty_path(self) -> None:
        turn = make_turn(2)
        turn.evidence_table.search_diagnostics = self._diag(
            seen_set_size=10,
            per_path_limit=20,
            dense={"hit_count": 10, "filtered_by_seen": 10},
            sparse={"hit_count": 0, "filtered_by_seen": 0},
        )
        state = make_state(turns=[make_turn(1), turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        assert "Depth accounting" in context
        assert "path empty for this query" in context

    def test_block_placed_between_seen_set_line_and_evidence_table(self) -> None:
        turn = make_turn(2)
        turn.evidence_table.search_diagnostics = self._diag(
            seen_set_size=10,
            per_path_limit=20,
            dense={"hit_count": 10, "filtered_by_seen": 10},
        )
        state = make_state(turns=[make_turn(1), turn])

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        seen_idx = context.index("Seen set:")
        depth_idx = context.index("Depth accounting")
        results_idx = context.index("Results with human ratings:")
        assert seen_idx < depth_idx < results_idx


class TestAuditTurnRendering:
    """An audit-flagged past turn is marked in both full and compressed
    progress-log forms so reflection can reason about whether the
    operator's last drop helped or hurt."""

    def _reflection(self) -> dict:
        return {
            "observe": "observe text",
            "diagnose": "diagnose text",
            "hypothesis": "If the next turn shows X, then Y",
            "previous_hypothesis_verdict": None,
        }

    def test_full_render_marks_audit_turn(self) -> None:
        turn = make_turn(2, reflection=self._reflection())
        turn.audit_turn = True

        rendered = _render_turn_full(turn)
        assert "Audit turn: true" in rendered

    def test_full_render_marks_non_audit_turn(self) -> None:
        turn = make_turn(2, reflection=self._reflection())
        rendered = _render_turn_full(turn)
        assert "Audit turn: false" in rendered

    def test_summary_render_marks_audit_turn(self) -> None:
        turn = make_turn(2, reflection=self._reflection())
        turn.audit_turn = True

        rendered = _render_turn_summary(turn)
        assert "AUDIT" in rendered

    def test_summary_render_omits_audit_marker_on_regular_turn(self) -> None:
        turn = make_turn(2, reflection=self._reflection())
        rendered = _render_turn_summary(turn)
        assert "AUDIT" not in rendered

    def test_audit_marker_survives_compression(self) -> None:
        """Compression kicks in at 9+ turns — ensure the AUDIT marker
        survives the summary form used for early turns."""
        turns = []
        for i in range(1, 11):
            t = make_turn(i, precision=0.4, reflection=self._reflection())
            if i == 3:
                t.audit_turn = True
            turns.append(t)
        state = make_state(turns=turns)

        msgs = build_messages(state, "sys", "instr")
        context = msgs[1]["content"]

        summary_lines = [
            ln for ln in context.splitlines() if ln.startswith("Turn 3: Config=")
        ]
        assert summary_lines, "expected compressed summary for turn 3"
        assert "AUDIT" in summary_lines[0]


class TestLoadedPromptContents:
    """Regression: critical rules must be reachable inside the loaded prompt
    (not stranded in a doc-only fenced block the loader skips)."""

    def test_system_prompt_fixes_ranker_to_rrf(self) -> None:
        from src.reflection.prompt import load_system_prompt

        sys_prompt = load_system_prompt()
        # Fusion is locked to RRFRanker; weights + score scales no longer apply.
        assert "RRFRanker" in sys_prompt
        assert "rrf_k" in sys_prompt

    def test_reflection_instructions_describe_path_drop(self) -> None:
        from src.reflection.prompt import load_reflection_instructions

        instr = load_reflection_instructions()
        # Reflection's only "lever" is recommending a path drop; the
        # rules for when that recommendation is appropriate must reach
        # the loaded prompt (not be stranded in a doc-only block).
        assert "drop" in instr.lower()
        assert "operator" in instr.lower()

    def test_reflection_instructions_describe_three_phases(self) -> None:
        from src.reflection.prompt import load_reflection_instructions

        instr = load_reflection_instructions()
        # Phases stay observe → diagnose → hypothesize. Prescribe is gone.
        assert "OBSERVE" in instr
        assert "DIAGNOSE" in instr
        assert "HYPOTHESIZE" in instr
        assert "PRESCRIBE" not in instr

    def test_json_schema_appendix_forbids_prescribe(self) -> None:
        from src.reflection.prompt import load_reflection_instructions

        instr = load_reflection_instructions()
        # Schema appendix must explicitly tell the LLM not to emit a
        # prescribe block — reflection no longer tunes.
        assert "prescribe" in instr
        assert "DO NOT include" in instr

    def test_reflection_instructions_describe_path_drop_recommendation_field(
        self,
    ) -> None:
        from src.reflection.prompt import load_reflection_instructions

        instr = load_reflection_instructions()
        # The structured field, the cumulative-evidence standard, the
        # early-turn null guarantee, and the Rule B1 self-check must
        # all reach the loaded prompt — they're the load-bearing
        # rules for the direct-apply field.
        assert "path_drop_recommendation" in instr
        assert "PATH-DROP RECOMMENDATION" in instr
        assert "cumulative" in instr.lower()
        assert "turn 2" in instr.lower()
        # Rule B1 self-check is now the agent's responsibility since
        # the apply site no longer enforces it.
        assert "rule b1" in instr.lower()
        # Confidence levels must be documented.
        assert '"low"' in instr or "low:" in instr.lower()
        assert '"high"' in instr or "high:" in instr.lower()


class TestStatelessRecommendationFeedback:
    """The agent must not see its own past path_drop_recommendation
    values rendered back into the progress log — that risks teaching
    it to suppress recommendations to avoid being declined.
    """

    def test_render_turn_full_omits_path_drop_recommendation(self) -> None:
        turn = make_turn(
            2,
            reflection={
                "diagnose": "dense path is noisy",
                "hypothesis": "if dense stays NOT_FIT then drop",
                "previous_hypothesis_verdict": "CONFIRMED",
                "path_drop_recommendation": {
                    "path": "dense",
                    "reason": "2 turns of NOT_FIT-only dense contribution",
                    "confidence": "high",
                },
            },
        )
        rendered = _render_turn_full(turn)
        # The recommendation's reason text must not surface into the
        # context the agent reads on the next turn.
        assert "2 turns of NOT_FIT-only dense contribution" not in rendered
        assert "path_drop_recommendation" not in rendered
        # Other reflection fields still render.
        assert "dense path is noisy" in rendered
        assert "CONFIRMED" in rendered


class TestFormatConfig:
    def test_rrf_ranker(self) -> None:
        cfg = make_config(rrf_k=60)
        result = _format_config(cfg)
        assert "RRFRanker(k=60)" in result
        assert "per_path_limit=20" in result
        assert "top_k=10" in result
        assert "active_paths=[dense,sparse]" in result

    def test_reduced_active_paths(self) -> None:
        cfg = make_config(active_paths=frozenset({"dense"}))
        result = _format_config(cfg)
        assert "active_paths=[dense]" in result
