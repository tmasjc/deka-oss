"""Tests for src.apply.load_session — sidecar parsing + guardrail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.apply.errors import ApplyGuardrailError, ApplyLoadError
from src.apply.load_session import (
    RubricPin,
    load_phase4_session_inputs,
    verify_rubric_pin,
)


def _write_session(tmp_path: Path, sid: str = "abc") -> None:
    rubric = {
        "version": 1,
        "prompt_sha256": "0" * 64,
        # Plus the rest of the rubric metadata fields; loader only
        # reads version + prompt_sha256, so we don't need to ship a
        # valid RubricMetadata here.
    }
    (tmp_path / f"{sid}.phase3.rubric.json").write_text(
        json.dumps(rubric), encoding="utf-8"
    )
    details = {
        "search": {
            "collection": "test_collection",
            "embed_url": "http://localhost:8000",
            "embed_model_id": "bge-m3-test",
        }
    }
    (tmp_path / f"{sid}.details.jsonl").write_text(
        json.dumps(details) + "\n", encoding="utf-8"
    )
    evidence = [
        {
            "pk": "k1",
            "nearest_fit_distance": 0.1,
            "decile": 0,
            "verdict": "KEEP",
        },
        {
            "pk": "k2",
            "nearest_fit_distance": 0.2,
            "decile": 1,
            "verdict": "DROP",
        },
        {
            "pk": "k3",
            "nearest_fit_distance": 0.3,
            "decile": 2,
            "verdict": "ERROR",
        },
    ]
    (tmp_path / f"{sid}.phase3.evidence.jsonl").write_text(
        "\n".join(json.dumps(r) for r in evidence) + "\n",
        encoding="utf-8",
    )
    cohort = [
        {"pk": "c1", "nearest_fit_distance": 0.11},
        {"pk": "c2", "nearest_fit_distance": 0.22},
    ]
    (tmp_path / f"{sid}.phase2.jsonl").write_text(
        "\n".join(json.dumps(r) for r in cohort) + "\n", encoding="utf-8"
    )


def test_happy_path(tmp_path):
    _write_session(tmp_path)
    inputs = load_phase4_session_inputs("abc", runs_dir=tmp_path)
    assert inputs.rubric.rubric_version == 1
    assert inputs.rubric.prompt_sha256 == "0" * 64
    assert inputs.search.collection == "test_collection"
    assert inputs.search.embed_model_id == "bge-m3-test"
    # ERROR row filtered out, two labels remain.
    assert len(inputs.labels) == 2
    assert {label.verdict for label in inputs.labels} == {"KEEP", "DROP"}
    assert len(inputs.cohort) == 2


def test_missing_rubric_raises(tmp_path):
    _write_session(tmp_path)
    (tmp_path / "abc.phase3.rubric.json").unlink()
    with pytest.raises(ApplyLoadError):
        load_phase4_session_inputs("abc", runs_dir=tmp_path)


def test_empty_labels_after_filter_raises(tmp_path):
    _write_session(tmp_path)
    evidence = [
        {"pk": "k1", "nearest_fit_distance": 0.1, "decile": 0, "verdict": "ERROR"}
    ]
    (tmp_path / "abc.phase3.evidence.jsonl").write_text(
        json.dumps(evidence[0]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ApplyLoadError):
        load_phase4_session_inputs("abc", runs_dir=tmp_path)


def test_verify_rubric_pin_happy():
    pin = RubricPin(rubric_version=1, prompt_sha256="a" * 64)
    verify_rubric_pin(classifier_pin=pin, session_pin=pin)


def test_verify_rubric_pin_sha_mismatch_raises():
    a = RubricPin(rubric_version=1, prompt_sha256="a" * 64)
    b = RubricPin(rubric_version=1, prompt_sha256="b" * 64)
    with pytest.raises(ApplyGuardrailError):
        verify_rubric_pin(classifier_pin=a, session_pin=b)


def test_verify_rubric_pin_version_mismatch_raises():
    a = RubricPin(rubric_version=1, prompt_sha256="c" * 64)
    b = RubricPin(rubric_version=2, prompt_sha256="c" * 64)
    with pytest.raises(ApplyGuardrailError):
        verify_rubric_pin(classifier_pin=a, session_pin=b)
