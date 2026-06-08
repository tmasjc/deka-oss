"""Tests for reflection schema models."""

import pytest
from pydantic import ValidationError

from src.reflection.schema import PathDropRecommendation, ReflectionOutput


class TestReflectionOutput:
    def test_round_trip(self) -> None:
        data = {
            "observe": "3 FIT, 2 NOT_FIT",
            "diagnose": "Dense path is noisy; consider an audit.",
            "hypothesis": (
                "If the next turn shows the same dense-NOT_FIT pattern, "
                "the diagnosis is correct because the path's score "
                "magnitudes are stable across turns."
            ),
            "previous_hypothesis_verdict": "CONFIRMED",
            "status": "CONTINUE",
            "turns_to_converge": None,
        }
        output = ReflectionOutput.model_validate(data)
        assert output.observe == "3 FIT, 2 NOT_FIT"
        assert output.previous_hypothesis_verdict == "CONFIRMED"
        assert output.status == "CONTINUE"
        assert output.turns_to_converge is None

    def test_minimal_continue(self) -> None:
        output = ReflectionOutput.model_validate(
            {
                "observe": "obs",
                "diagnose": "diag",
                "hypothesis": "hypo",
            }
        )
        assert output.status == "CONTINUE"
        assert output.previous_hypothesis_verdict is None
        assert output.turns_to_converge is None

    def test_to_log_dict_omits_turns_to_converge_on_continue(self) -> None:
        output = ReflectionOutput(observe="o", diagnose="d", hypothesis="h")
        d = output.to_log_dict()
        assert d["observe"] == "o"
        assert d["diagnose"] == "d"
        assert d["hypothesis"] == "h"
        assert d["status"] == "CONTINUE"
        assert "turns_to_converge" not in d
        assert "prescribe" not in d

    def test_to_log_dict_carries_turns_to_converge_on_converged(self) -> None:
        output = ReflectionOutput(
            observe="o",
            diagnose="d",
            hypothesis="session converged",
            status="CONVERGED",
            turns_to_converge=5,
        )
        d = output.to_log_dict()
        assert d["status"] == "CONVERGED"
        assert d["turns_to_converge"] == 5
        assert "prescribe" not in d

    def test_rejects_prescribe_field(self) -> None:
        # ``prescribe`` was removed when reflection became a pure
        # reasoning-trace. The schema must hard-fail rather than silently
        # accept a stray emission so the LLM doesn't drift back into
        # tuning behaviour without anyone noticing.
        with pytest.raises(ValidationError):
            ReflectionOutput.model_validate(
                {
                    "observe": "o",
                    "diagnose": "d",
                    "hypothesis": "h",
                    "prescribe": {
                        "rrf_k": 60,
                        "top_k": 10,
                        "active_paths": ["dense", "sparse"],
                    },
                }
            )

    def test_rejects_unknown_top_level_field(self) -> None:
        with pytest.raises(ValidationError):
            ReflectionOutput.model_validate(
                {
                    "observe": "o",
                    "diagnose": "d",
                    "hypothesis": "h",
                    "bogus_extra": 1,
                }
            )

    def test_path_drop_recommendation_round_trip(self) -> None:
        data = {
            "observe": "o",
            "diagnose": "d",
            "hypothesis": "h",
            "path_drop_recommendation": {
                "path": "sparse",
                "reason": "sparse contributed only NOT_FIT for 2 turns",
                "confidence": "medium",
            },
        }
        output = ReflectionOutput.model_validate(data)
        assert output.path_drop_recommendation is not None
        assert output.path_drop_recommendation.path == "sparse"
        assert output.path_drop_recommendation.confidence == "medium"
        d = output.to_log_dict()
        assert d["path_drop_recommendation"] == {
            "path": "sparse",
            "reason": "sparse contributed only NOT_FIT for 2 turns",
            "confidence": "medium",
        }

    def test_to_log_dict_omits_path_drop_recommendation_when_none(self) -> None:
        output = ReflectionOutput(observe="o", diagnose="d", hypothesis="h")
        d = output.to_log_dict()
        assert "path_drop_recommendation" not in d

    def test_path_drop_recommendation_defaults_to_none(self) -> None:
        output = ReflectionOutput.model_validate(
            {"observe": "o", "diagnose": "d", "hypothesis": "h"}
        )
        assert output.path_drop_recommendation is None


class TestPathDropRecommendation:
    def test_rejects_path_outside_enum(self) -> None:
        with pytest.raises(ValidationError):
            PathDropRecommendation.model_validate(
                {"path": "lexical", "reason": "r", "confidence": "low"}
            )

    def test_rejects_confidence_outside_enum(self) -> None:
        with pytest.raises(ValidationError):
            PathDropRecommendation.model_validate(
                {"path": "dense", "reason": "r", "confidence": "very_high"}
            )

    def test_rejects_empty_reason(self) -> None:
        with pytest.raises(ValidationError):
            PathDropRecommendation.model_validate(
                {"path": "dense", "reason": "", "confidence": "low"}
            )

    def test_rejects_extra_field(self) -> None:
        # ``extra="forbid"`` on the sub-model: a stray key (e.g. an LLM
        # adding ``priority``) hard-fails rather than being silently
        # accepted, mirroring ``ReflectionOutput`` policy.
        with pytest.raises(ValidationError):
            PathDropRecommendation.model_validate(
                {
                    "path": "dense",
                    "reason": "r",
                    "confidence": "low",
                    "priority": 1,
                }
            )
