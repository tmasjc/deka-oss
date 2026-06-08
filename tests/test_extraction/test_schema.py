"""Tests for ``SpanExtractionResult`` validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.extraction.schema import SpanExtractionResult


class TestSpanExtractionResult:
    def test_empty_span_is_valid(self) -> None:
        r = SpanExtractionResult(span_line_indices=[], reason="no clean span")
        assert r.span_line_indices == []

    def test_single_index_is_valid(self) -> None:
        r = SpanExtractionResult(span_line_indices=[3], reason="ok")
        assert r.span_line_indices == [3]

    def test_two_contiguous_indices_valid(self) -> None:
        r = SpanExtractionResult(span_line_indices=[3, 4], reason="ok")
        assert r.span_line_indices == [3, 4]

    def test_three_contiguous_indices_valid(self) -> None:
        r = SpanExtractionResult(span_line_indices=[0, 1, 2], reason="ok")
        assert r.span_line_indices == [0, 1, 2]

    def test_four_indices_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds 3 lines"):
            SpanExtractionResult(span_line_indices=[0, 1, 2, 3], reason="bad")

    def test_non_contiguous_sorted_indices_valid(self) -> None:
        # Contiguity is no longer required — disjoint highlights are fine.
        r = SpanExtractionResult(span_line_indices=[3, 5], reason="ok")
        assert r.span_line_indices == [3, 5]

    def test_out_of_order_indices_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sorted and unique"):
            SpanExtractionResult(span_line_indices=[5, 4], reason="bad")

    def test_duplicate_indices_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sorted and unique"):
            SpanExtractionResult(span_line_indices=[2, 2], reason="bad")

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValidationError, match="negative"):
            SpanExtractionResult(span_line_indices=[-1, 0], reason="bad")

    def test_reason_required(self) -> None:
        with pytest.raises(ValidationError):
            SpanExtractionResult(span_line_indices=[])  # type: ignore[call-arg]

    def test_json_round_trip(self) -> None:
        r = SpanExtractionResult(span_line_indices=[2, 3], reason="ok")
        dumped = r.model_dump_json()
        reloaded = SpanExtractionResult.model_validate_json(dumped)
        assert reloaded == r
