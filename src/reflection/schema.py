"""Pydantic model for structured LLM reflection output.

Reflection is a reasoning-trace, not a tuning controller. The session
config (``rrf_k``, ``top_k``, ``active_paths``, ``per_path_limit``) is
locked once turn 1 starts; the only mid-session config change is a
path drop. ``extra="forbid"`` ensures a stray prescription from an
off-policy LLM hard-fails rather than being silently dropped.

The optional ``path_drop_recommendation`` field is the agent's
sanctioned channel for nominating a path drop when cumulative
evidence warrants it. The operator confirms via a one-keystroke
prompt at end-of-turn; ``[a]pply`` drops the path immediately, with
no audit step in between. The agent therefore carries the burden of
weighing Rule B1 itself before recommending — it must not nominate a
drop that would lose a sole-sourced FIT chunk visible in the
progress log.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PathDropRecommendation(BaseModel):
    """Structured nomination of a path for direct drop on apply.

    Populated by the agent only when cumulative session evidence (2+
    consecutive turns of consistent NOT_FIT contribution from the
    named path, with the FIT pile sourced elsewhere) warrants it.
    Never populated on turn 1; null on most turns thereafter.

    Apply semantics: ``[a]pply`` calls
    :meth:`SessionState.apply_recommended_drop` and removes the path
    from ``active_paths`` for the remainder of the session — no
    audit, no candidate rating. Rule B is NOT enforced at the apply
    site; the agent is expected to have considered it from the
    progress log before recommending.
    """

    model_config = ConfigDict(extra="forbid")

    path: Literal["dense", "sparse"] = Field(
        description="Active retrieval path the agent recommends dropping."
    )
    reason: str = Field(
        min_length=1,
        description=(
            "Short justification grounded in cumulative session evidence "
            "(per-path counts and FIT distributions across the rendered "
            "progress log) — not single-turn stats. Must also implicitly "
            "address Rule B1: no FIT chunk in the visible log should be "
            "sole-sourced by the path being dropped."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Agent's confidence that the drop is correct and safe. "
            "low: 2 turns of consistent NOT_FIT, FIT pile sourced "
            "elsewhere, but pattern may be situational. medium: 2–3 "
            "turns of consistent NOT_FIT, no Rule B1 risk visible. "
            "high: stable pattern + material FIT-pile impact + no "
            "Rule B1 risk visible from the log."
        )
    )


class ReflectionOutput(BaseModel):
    """Three-phase reflection output from the LLM.

    The phases are ``observe`` → ``diagnose`` → ``hypothesize``. An
    optional structured ``path_drop_recommendation`` carries a
    direct-apply path-drop nomination when cumulative evidence
    warrants it; the operator confirms via the one-keystroke prompt.
    ``status`` is retained because ``CONTINUE`` vs ``CONVERGED`` is
    still a meaningful judgment the agent contributes against the
    dual gate.
    """

    model_config = ConfigDict(extra="forbid")

    observe: str = Field(description="Raw facts about the current turn")
    diagnose: str = Field(description="Root-cause interpretation")
    hypothesis: str = Field(
        description=(
            "One falsifiable hypothesis about the diagnosis, in "
            "'If the next turn shows … then … because …' form."
        )
    )
    previous_hypothesis_verdict: Literal["CONFIRMED", "REFUTED"] | None = Field(
        default=None
    )
    path_drop_recommendation: PathDropRecommendation | None = Field(
        default=None,
        description=(
            "Optional path-drop nomination grounded in cumulative "
            "evidence. Null on most turns; populated only when 2+ "
            "consecutive turns show consistent NOT_FIT contribution "
            "from the named path AND no Rule B1 risk is visible in "
            "the log."
        ),
    )
    status: Literal["CONTINUE", "CONVERGED"] = "CONTINUE"
    turns_to_converge: int | None = Field(
        default=None,
        description="Set when status is CONVERGED",
    )

    def to_log_dict(self) -> dict[str, Any]:
        """Reshape to the nested format expected by ``progress_log.md``."""
        result: dict[str, Any] = {
            "observe": self.observe,
            "diagnose": self.diagnose,
            "hypothesis": self.hypothesis,
            "previous_hypothesis_verdict": self.previous_hypothesis_verdict,
            "status": self.status,
        }
        if self.path_drop_recommendation is not None:
            result["path_drop_recommendation"] = (
                self.path_drop_recommendation.model_dump()
            )
        if self.turns_to_converge is not None:
            result["turns_to_converge"] = self.turns_to_converge
        return result
