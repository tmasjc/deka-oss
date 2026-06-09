"""Pydantic models for Phase 3 artifacts.

Two shapes live here:

- :class:`RubricMetadata` is the structured companion to the rubric
  prompt's markdown rendering. The rubric prompt and this metadata
  co-vary by construction: the metadata renders to the prompt; the
  prompt re-parses into the metadata. See
  :mod:`src.refine.derive` for the round-trip.
- :func:`make_judge_verdict_model` returns a per-judging-call Pydantic
  model factory that closes over the rubric's declared
  ``failed_check`` enum and the candidate chunk's line count, so a
  single ``model_validate_json`` call enforces both the closed-enum
  contract and the line-index range from the proposal.

Both models reject unknown fields (``extra="forbid"``); a stray field
in derive output or judge output is a hard parse error, not a silent
pass-through.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Rubric metadata (rubric.json sidecar)
# ---------------------------------------------------------------------------


class RubricCheck(BaseModel):
    """One named predicate the judge applies to a chunk."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1)
    required: bool = True


class RubricFitExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pk: str | int
    span_text: str = Field(..., min_length=1)


class RubricNotFitExample(BaseModel):
    """A NOT_FIT exemplar plus the check id(s) it fails."""

    model_config = ConfigDict(extra="forbid")

    pk: str | int
    span_text: str = Field(..., min_length=1)
    fails: list[str] = Field(..., min_length=1)


class RubricMetadata(BaseModel):
    """Structured view of a rubric prompt — exact JSON shape persisted
    as ``runs/{sid}.phase3.rubric.json``.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    source_session_id: str = Field(..., min_length=1)
    derive_model_id: str = Field(..., min_length=1)
    meta_prompt_path: str
    meta_prompt_sha256: str = Field(..., min_length=64, max_length=64)
    checks: list[RubricCheck] = Field(..., min_length=1)
    fit_examples: list[RubricFitExample] = Field(..., min_length=1)
    not_fit_examples: list[RubricNotFitExample]
    prompt_path: str
    prompt_sha256: str = Field(..., min_length=64, max_length=64)
    version: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> "RubricMetadata":
        ids = {c.id for c in self.checks}
        if len(ids) != len(self.checks):
            raise ValueError(
                "RubricMetadata.checks contains duplicate ids: "
                f"{[c.id for c in self.checks]}"
            )
        for example in self.not_fit_examples:
            unknown = set(example.fails) - ids
            if unknown:
                raise ValueError(
                    f"NOT_FIT example pk={example.pk} references unknown "
                    f"check id(s) {sorted(unknown)}; declared ids: {sorted(ids)}"
                )
        return self

    @property
    def allowed_check_ids(self) -> frozenset[str]:
        return frozenset(c.id for c in self.checks)


# ---------------------------------------------------------------------------
# Derive LLM output (JSON shape — the harness renders markdown from this)
# ---------------------------------------------------------------------------


class DeriveCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1)


class DeriveNotFitAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pk: str | int
    fails: list[str] = Field(..., min_length=1)


class DeriveLLMOutput(BaseModel):
    """Strict-JSON shape the derive LLM emits.

    The harness combines this with the original session's FIT and
    NOT_FIT chunks to produce a :class:`RubricMetadata`. Decoupling
    the LLM's job (identify the discriminators) from the harness's
    job (canonical markdown rendering) makes the meta-prompt much
    more reliable across LLMs — JSON mode is universally supported,
    while structured-markdown emission is not.
    """

    model_config = ConfigDict(extra="forbid")

    checks: list[DeriveCheck] = Field(..., min_length=2, max_length=4)
    not_fit_annotations: list[DeriveNotFitAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> "DeriveLLMOutput":
        ids = {c.id for c in self.checks}
        if len(ids) != len(self.checks):
            raise ValueError(
                "derive output: checks contain duplicate ids: "
                f"{[c.id for c in self.checks]}"
            )
        for ann in self.not_fit_annotations:
            unknown = set(ann.fails) - ids
            if unknown:
                raise ValueError(
                    f"derive output: not_fit_annotations[pk={ann.pk}] references "
                    f"unknown check id(s) {sorted(unknown)}; declared: {sorted(ids)}"
                )
        return self


# ---------------------------------------------------------------------------
# Judge verdict (one row per judged chunk)
# ---------------------------------------------------------------------------


def make_judge_verdict_model(
    *, allowed_checks: frozenset[str], chunk_line_count: int
) -> type[BaseModel]:
    """Return a Pydantic model class for one judge verdict.

    The factory closes over the per-call constraints — the rubric's
    declared ``failed_check`` enum and the line range of the chunk
    being judged — so a single ``model_validate_json(raw)`` enforces
    everything the proposal calls out:

    - ``verdict ∈ {KEEP, DROP}``
    - ``KEEP`` requires non-empty ``evidence_line_indices`` and
      ``failed_check is None``
    - ``DROP`` requires ``failed_check`` ∈ ``allowed_checks``
    - ``evidence_line_indices`` are 1-based, ascending, unique,
      length 1–3, all within ``[1, chunk_line_count]``
    - ``reason`` ≤ 500 chars
    - Unknown top-level keys hard-fail.

    The closed-over constants are baked into per-call validators so
    Pydantic emits one error path, not two; the alternative —
    post-parse validation — would need a second error surface.
    """
    if chunk_line_count <= 0:
        raise ValueError("chunk_line_count must be a positive integer")

    allowed_checks = frozenset(allowed_checks)

    class JudgeVerdict(BaseModel):
        model_config = ConfigDict(extra="forbid")

        verdict: Literal["KEEP", "DROP"]
        evidence_line_indices: list[int] = Field(default_factory=list)
        failed_check: str | None = None
        reason: Annotated[str, Field(max_length=500)] = ""

        @model_validator(mode="after")
        def _enforce_contract(self) -> "JudgeVerdict":
            indices = self.evidence_line_indices
            if not (1 <= len(indices) <= 3):
                raise ValueError(
                    f"evidence_line_indices must have length 1-3, got {len(indices)}"
                )
            for idx in indices:
                if not isinstance(idx, int) or isinstance(idx, bool):
                    raise ValueError(
                        f"evidence_line_indices must contain ints, got {idx!r}"
                    )
                if not 1 <= idx <= chunk_line_count:
                    raise ValueError(
                        f"evidence_line_index {idx} out of range "
                        f"[1, {chunk_line_count}]"
                    )
            if list(indices) != sorted(set(indices)):
                raise ValueError(
                    f"evidence_line_indices must be ascending and unique; got {indices}"
                )

            if self.verdict == "KEEP":
                if self.failed_check is not None:
                    raise ValueError(
                        "KEEP verdicts must have failed_check=null; "
                        f"got {self.failed_check!r}"
                    )
            else:  # DROP
                if self.failed_check is None:
                    raise ValueError(
                        "DROP verdicts must declare a failed_check from "
                        f"{sorted(allowed_checks)}"
                    )
                if self.failed_check not in allowed_checks:
                    raise ValueError(
                        f"failed_check {self.failed_check!r} is not in the "
                        f"rubric's declared enum {sorted(allowed_checks)}"
                    )
            return self

    JudgeVerdict.__qualname__ = "JudgeVerdict"
    return JudgeVerdict
