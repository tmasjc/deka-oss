"""Tests for ReflectionAgent with mocked OpenAI client.

Reflection is a reasoning-trace, not a tuning controller. The agent
emits observe / diagnose / hypothesis / status; prescription validation
(Rule A oscillation, Rule B sole-source / candidate-FIT) lives in
``SessionState.apply_path_drop`` and is exercised by web-API audit-flow
tests.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.reflection.agent import ReflectionAgent, _extract_json
from src.reflection.errors import LLMCallError

from .conftest import make_state, make_turn


def _mock_response(data: dict[str, Any]) -> MagicMock:
    """Build a mock ChatCompletion response."""
    message = MagicMock()
    message.content = json.dumps(data)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _valid_reflection(**overrides: Any) -> dict[str, Any]:
    """Return a valid ReflectionOutput JSON dict."""
    base: dict[str, Any] = {
        "observe": "3 FIT, 2 NOT_FIT. Precision improved.",
        "diagnose": (
            "Dense path is the primary noise source; consider an audit if "
            "the pattern persists for another turn."
        ),
        "hypothesis": (
            "If the next turn surfaces the same dense-NOT_FIT pattern, the "
            "diagnosis is correct because score magnitudes are stable."
        ),
        "previous_hypothesis_verdict": None,
        "status": "CONTINUE",
        "turns_to_converge": None,
    }
    base.update(overrides)
    return base


def _make_agent(client: MagicMock) -> ReflectionAgent:
    return ReflectionAgent(client=client, api_key="test-key")


class TestReflectHappyPath:
    def test_returns_valid_dict(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            _valid_reflection()
        )
        agent = _make_agent(client)
        turn = make_turn(1, precision=0.6)
        state = make_state(turns=[turn])

        result = agent.reflect(state)

        assert result is not None
        assert result["observe"] == "3 FIT, 2 NOT_FIT. Precision improved."
        assert result["status"] == "CONTINUE"
        assert "prescribe" not in result
        assert "validation_error" not in result

    def test_calls_openai_with_json_format(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            _valid_reflection()
        )
        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        agent.reflect(state)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert len(call_kwargs["messages"]) == 3

    def test_converged_passthrough(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            _valid_reflection(status="CONVERGED", turns_to_converge=4)
        )
        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)

        assert result is not None
        assert result["status"] == "CONVERGED"
        assert result["turns_to_converge"] == 4


class TestSchemaEnforcement:
    def test_rejects_prescribe_in_response(self) -> None:
        """A drifting LLM that re-emits a prescribe block must fail to
        parse rather than silently slipping past — the schema is the
        single contract enforcing that reflection no longer tunes."""
        client = MagicMock()
        data = _valid_reflection()
        data["prescribe"] = {
            "rrf_k": 120,
            "top_k": 10,
            "active_paths": ["dense", "sparse"],
        }
        client.chat.completions.create.return_value = _mock_response(data)
        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)

        # Parse failure path: the result carries only the diagnostics
        # trail with parse_error populated.
        assert result is not None
        assert set(result.keys()) == {"_diagnostics"}
        assert result["_diagnostics"]["parse_error"] is not None


class TestErrorHandling:
    def test_api_error_raises_llm_call_error(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("connection failed")

        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        with pytest.raises(LLMCallError, match="connection failed"):
            agent.reflect(state)

    def test_parse_failure_returns_diagnostics_only(self) -> None:
        client = MagicMock()
        message = MagicMock()
        message.content = "not valid json at all"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)
        assert result is not None
        assert set(result.keys()) == {"_diagnostics"}
        diag = result["_diagnostics"]
        assert diag["raw_response"] == "not valid json at all"
        assert diag["parse_error"] is not None

    def test_empty_turns_returns_none(self) -> None:
        client = MagicMock()
        agent = _make_agent(client)
        state = make_state(turns=[])

        result = agent.reflect(state)
        assert result is None
        client.chat.completions.create.assert_not_called()

    def test_empty_content_returns_diagnostics_only(self) -> None:
        client = MagicMock()
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)
        assert result is not None
        assert set(result.keys()) == {"_diagnostics"}
        assert result["_diagnostics"]["raw_response"] is None


class TestRecommendationSuppression:
    """If the LLM recommends dropping a path that's no longer active
    (e.g. dropped on an earlier turn but the agent reasoned from the
    historical progress log), suppress the recommendation so the
    operator never sees a modal it can't act on. The suppressed value
    lands on ``_diagnostics`` for post-hoc analysis.
    """

    def _agent_for_response(
        self, recommendation: dict[str, Any]
    ) -> tuple[ReflectionAgent, MagicMock]:
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_response(
            _valid_reflection(path_drop_recommendation=recommendation)
        )
        return _make_agent(client), client

    def test_strips_recommendation_for_inactive_path(self) -> None:
        from src.search.config import with_overrides

        agent, _ = self._agent_for_response(
            {"path": "dense", "reason": "noisy", "confidence": "high"}
        )
        # Build a state where dense was dropped before this reflection.
        state = make_state(turns=[make_turn(1)])
        state.current_config = with_overrides(
            state.current_config, active_paths=frozenset({"sparse"})
        )

        result = agent.reflect(state)

        assert result is not None
        # The modal/banner field is gone — the operator never sees a
        # recommendation it can't act on.
        assert "path_drop_recommendation" not in result
        # The suppressed value lands in diagnostics for analytics.
        diag = result["_diagnostics"]
        assert "suppressed_recommendation" in diag
        suppressed = diag["suppressed_recommendation"]
        assert suppressed["reason"] == "path_inactive"
        assert suppressed["recommendation"]["path"] == "dense"
        assert suppressed["active_paths"] == ["sparse"]

    def test_passes_through_recommendation_for_active_path(self) -> None:
        agent, _ = self._agent_for_response(
            {
                "path": "dense",
                "reason": "2 turns of NOT_FIT",
                "confidence": "medium",
            }
        )
        # Default state has both paths active.
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)

        assert result is not None
        rec = result.get("path_drop_recommendation")
        assert rec is not None
        assert rec["path"] == "dense"
        # No suppression marker when the recommendation is valid.
        assert "suppressed_recommendation" not in result["_diagnostics"]

    def test_strips_recommendation_when_drop_already_applied(self) -> None:
        """Session-level safeguard: once an apply has landed, no
        further recommendations surface — even for paths that are
        still active. Each session is entitled to one
        agent-recommended drop.
        """
        agent, _ = self._agent_for_response(
            {
                "path": "sparse",
                "reason": "now sparse looks bad too",
                "confidence": "medium",
            }
        )
        # Session has both paths active (operator hasn't dropped any
        # via the recommendation flow yet from a config standpoint),
        # but the flag is set: a previous apply landed earlier.
        state = make_state(turns=[make_turn(1)])
        state.recommended_drop_applied = True

        result = agent.reflect(state)

        assert result is not None
        # Recommendation suppressed even though sparse is still active.
        assert "path_drop_recommendation" not in result
        diag = result["_diagnostics"]
        assert "suppressed_recommendation" in diag
        suppressed = diag["suppressed_recommendation"]
        assert suppressed["reason"] == "recommended_drop_already_applied"
        assert suppressed["recommendation"]["path"] == "sparse"
        assert suppressed["recommended_drop_applied"] is True


class TestExtractJson:
    def test_plain_json_passthrough(self) -> None:
        raw = '{"key": "value"}'
        assert _extract_json(raw) == raw

    def test_think_tags_stripped(self) -> None:
        raw = '<think>\nLet me analyze...\n</think>\n{"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_multiple_think_blocks(self) -> None:
        raw = '<think>first</think><think>second</think>{"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_prose_wrapper_extracted(self) -> None:
        raw = 'Here is the JSON:\n{"key": "value"}\nDone.'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_no_json_returns_cleaned(self) -> None:
        raw = "<think>hmm</think>no json here"
        assert _extract_json(raw) == "no json here"


class TestThinkingModelParsing:
    def test_think_wrapped_response_parses(self) -> None:
        client = MagicMock()
        data = _valid_reflection()
        content = (
            f"<think>\nLet me analyze the results...\n</think>\n{json.dumps(data)}"
        )
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)
        assert result is not None
        assert result["observe"] == data["observe"]
        assert result["status"] == data["status"]

    def test_prose_wrapped_response_parses(self) -> None:
        client = MagicMock()
        data = _valid_reflection()
        content = f"Here is my reflection:\n{json.dumps(data)}"
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        agent = _make_agent(client)
        state = make_state(turns=[make_turn(1)])

        result = agent.reflect(state)
        assert result is not None
