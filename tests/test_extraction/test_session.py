"""Tests for ``src.extraction.session`` — the per-turn closure helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.extraction.cache import SpanCache
from src.extraction.extractor import SpanExtractor
from src.extraction.schema import SpanExtractionResult
from src.extraction.session import build_extract_span, gather_prior_fit_spans
from src.search.config import SearchConfig
from src.search.evidence import CandidateRow, EvidenceRow, EvidenceTable
from src.session.state import SessionState, TurnRecord


def _config() -> SearchConfig:
    return SearchConfig(rrf_k=60, per_path_limit=20, top_k=3)


def _row(rank: int, rating: str | None, span_text: str = "") -> EvidenceRow:
    return EvidenceRow(
        rank=rank,
        pk=rank,
        chunk_id=f"abc_C{rank:07d}",
        chunk_content=f"老师：row {rank}",
        sample_id="abc",
        counselor_id="T1",
        term="2025F",
        source_paths=["dense"],
        scores={"dense": 0.8, "sparse": 0.0, "bm25": 0.0},
        rating=rating,  # type: ignore[arg-type]
        span_text=span_text,
    )


def _candidate(rating: str | None, span_text: str = "") -> CandidateRow:
    return CandidateRow(
        path="sparse",
        rank_in_path=1,
        pk="cand",
        chunk_id="xyz_C0000001",
        chunk_content="老师：c",
        sample_id="xyz",
        counselor_id="T1",
        term="2025F",
        score=0.3,
        rating=rating,  # type: ignore[arg-type]
        span_text=span_text,
    )


def _turn(
    rows: list[EvidenceRow], candidates: list[CandidateRow] | None = None
) -> TurnRecord:
    table = EvidenceTable(
        query="q",
        config=_config(),
        rows=rows,
        per_path_candidates={
            "dense": [],
            "sparse": candidates or [],
            "bm25": [],
        },
    )
    return TurnRecord(
        turn_number=1,
        query=table.query,
        config=table.config,
        evidence_table=table,
        breakdown={
            "dense_only": {"total": 0, "fit": 0, "not_fit": 0},
            "sparse_only": {"total": 0, "fit": 0, "not_fit": 0},
            "bm25_only": {"total": 0, "fit": 0, "not_fit": 0},
            "multi_path": {"total": 0, "fit": 0, "not_fit": 0},
        },
        precision=0.0,
        reflection=None,
    )


class TestGatherPriorFitSpans:
    def test_empty_state_returns_empty(self) -> None:
        state = SessionState(current_config=_config())
        assert gather_prior_fit_spans(state) == []

    def test_collects_fit_rows_only(self) -> None:
        state = SessionState(current_config=_config())
        state.turns.append(
            _turn(
                [
                    _row(1, "FIT", "老师：good span"),
                    _row(2, "NOT_FIT", "not collected"),
                    _row(3, "FIT", "家长：another"),
                ]
            )
        )
        spans = gather_prior_fit_spans(state)
        assert spans == ["老师：good span", "家长：another"]

    def test_collects_fit_candidates_too(self) -> None:
        state = SessionState(current_config=_config())
        state.turns.append(
            _turn(
                [_row(1, "NOT_FIT", "skip me")],
                candidates=[_candidate("FIT", "老师：cand span")],
            )
        )
        assert gather_prior_fit_spans(state) == ["老师：cand span"]

    def test_skips_fit_rows_with_empty_span_text(self) -> None:
        state = SessionState(current_config=_config())
        state.turns.append(_turn([_row(1, "FIT", "")]))
        assert gather_prior_fit_spans(state) == []

    def test_caps_at_last_n(self) -> None:
        state = SessionState(current_config=_config())
        state.turns.append(_turn([_row(i, "FIT", f"span {i}") for i in range(1, 11)]))
        spans = gather_prior_fit_spans(state, cap=5)
        assert spans == [f"span {i}" for i in range(6, 11)]


class TestBuildExtractSpan:
    def test_closure_calls_extractor_with_query_and_prior(self, tmp_path: Path) -> None:
        client = MagicMock()
        import json as _json

        message = MagicMock()
        message.content = _json.dumps({"span_line_indices": [0], "reason": "ok"})
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        extractor = SpanExtractor(client=client, cache=SpanCache(tmp_path / "cache"))
        state = SessionState(current_config=_config())
        state.turns.append(_turn([_row(1, "FIT", "老师：prior")]))

        call = build_extract_span(extractor, state, "my query")
        indices, text = call("老师：first line\n家长：second")

        assert indices == [0]
        assert text == "老师：first line"

        sent_messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert "my query" in sent_messages[1]["content"]
        assert "老师：prior" in sent_messages[1]["content"]

    def test_closure_derives_span_text_from_chunk(self, tmp_path: Path) -> None:
        """``span_text`` is derived server-side — we don't trust the LLM to echo."""
        client = MagicMock()
        import json as _json

        message = MagicMock()
        # Return indices but no text field (server-side derivation).
        message.content = _json.dumps({"span_line_indices": [1], "reason": "ok"})
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        extractor = SpanExtractor(client=client, cache=SpanCache(tmp_path / "cache"))
        state = SessionState(current_config=_config())

        call = build_extract_span(extractor, state, "q")
        indices, text = call("老师：A\n家长：B\n老师：C")

        assert indices == [1]
        assert text == "家长：B"

    def test_closure_exposes_meta_attributes(self, tmp_path: Path) -> None:
        """model / prompt_version / latency_ms_total / cache_hits are delta-
        accounted against the extractor's running totals."""
        client = MagicMock()
        import json as _json

        message = MagicMock()
        message.content = _json.dumps({"span_line_indices": [0], "reason": "ok"})
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        extractor = SpanExtractor(client=client, cache=SpanCache(tmp_path / "cache"))
        state = SessionState(current_config=_config())

        call = build_extract_span(extractor, state, "q")
        assert call.model == extractor.model
        assert call.prompt_version == extractor.prompt_version
        # Before any extractions, counts are zero for this turn.
        assert call.cache_hits == 0
        assert call.latency_ms_total == 0.0

        call("老师：x\n家长：y")
        # One LLM call, no cache hit for this turn.
        assert call.cache_hits == 0

        # Second call on same content is a cache hit — surfaces in stats.
        call("老师：x\n家长：y")
        assert call.cache_hits == 1

    def test_closure_returns_none_on_extraction_error(
        self, tmp_path: Path, caplog
    ) -> None:
        """An ExtractionError on one chunk must not kill the turn — the
        closure returns ``None`` so ``run_search`` drops the chunk entirely
        (rather than surfacing a half-evaluated card)."""
        import logging

        client = MagicMock()
        import json as _json

        # Both attempts (initial + the one automatic retry) return spans
        # that exceed the 3-line cap, so validation fails twice and the
        # closure must surface failure as None — simulating an LLM that
        # stays off-script even after the retry.
        def _bad_response() -> MagicMock:
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = _json.dumps(
                {"span_line_indices": [0, 1, 2, 3], "reason": "too long"}
            )
            return r

        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message.content = _json.dumps(
            {"span_line_indices": [1], "reason": "clear"}
        )
        # Chunk 1: both attempts fail (initial + retry). Chunk 2: succeeds first try.
        client.chat.completions.create.side_effect = [
            _bad_response(),
            _bad_response(),
            good_response,
        ]

        extractor = SpanExtractor(client=client, cache=SpanCache(tmp_path / "cache"))
        state = SessionState(current_config=_config())
        call = build_extract_span(extractor, state, "q")

        with caplog.at_level(logging.WARNING, logger="src.extraction.session"):
            outcome = call("老师：a\n家长：b\n老师：c\n家长：d\n老师：e")
        assert outcome is None
        assert call.failures == 1
        assert any("Span extraction failed" in r.message for r in caplog.records)

        # Second chunk succeeds — the closure is still usable.
        outcome2 = call("老师：A\n家长：B\n老师：C")
        assert outcome2 == ([1], "家长：B")
        assert call.failures == 1

    def test_closure_reports_progress_callback(self, tmp_path: Path) -> None:
        """``on_progress`` fires with the running processed-count after each
        extraction attempt, success or failure."""
        client = MagicMock()
        import json as _json

        good = MagicMock()
        good.choices = [MagicMock()]
        good.choices[0].message.content = _json.dumps(
            {"span_line_indices": [0], "reason": "ok"}
        )
        bad = MagicMock()
        bad.choices = [MagicMock()]
        bad.choices[0].message.content = _json.dumps(
            {"span_line_indices": [0, 5], "reason": "bad"}
        )
        client.chat.completions.create.side_effect = [good, bad, good]

        extractor = SpanExtractor(client=client, cache=SpanCache(tmp_path / "cache"))
        state = SessionState(current_config=_config())
        seen: list[int] = []
        call = build_extract_span(
            extractor, state, "q", on_progress=lambda n: seen.append(n)
        )

        call("a\nb")
        call("c\nd\ne\nf\ng\nh")  # triggers the bad response + failure
        call("i\nj")

        assert seen == [1, 2, 3]

    def test_empty_span_closure(self, tmp_path: Path) -> None:
        """Empty indices -> empty span_text."""
        client = MagicMock()

        # Pre-populate the cache so we skip the LLM entirely.
        cache = SpanCache(tmp_path / "cache")
        extractor = SpanExtractor(client=client, cache=cache)
        from src.extraction.cache import CacheKey

        cache.put(
            CacheKey(
                model_id=extractor.model,
                prompt_version=extractor.prompt_version,
                query="q",
                chunk_content="老师：x",
            ),
            SpanExtractionResult(span_line_indices=[], reason="none"),
        )
        state = SessionState(current_config=_config())

        call = build_extract_span(extractor, state, "q")
        indices, text = call("老师：x")
        assert indices == []
        assert text == ""
        # Cache hit -> no LLM call.
        client.chat.completions.create.assert_not_called()
