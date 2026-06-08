"""Pydantic schema for span-extractor LLM responses.

The extractor returns 0-3 sorted, unique line indices into the chunk.
Lines need not be contiguous — the rater sees the full chunk with the
selected lines highlighted, so disjoint highlights are fine. Sorted +
unique is still required so ``derive_span_text`` produces a stable
join order.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SpanExtractionResult(BaseModel):
    """One extraction outcome for one chunk."""

    span_line_indices: list[int] = Field(default_factory=list)
    reason: str

    @field_validator("span_line_indices")
    @classmethod
    def _check_length_and_ordering(cls, v: list[int]) -> list[int]:
        if len(v) > 3:
            raise ValueError(f"span exceeds 3 lines: {v}")
        if v:
            if any(i < 0 for i in v):
                raise ValueError(f"negative line index: {v}")
            if sorted(set(v)) != v:
                raise ValueError(f"indices must be sorted and unique: {v}")
        return v
