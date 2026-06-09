"""Unit tests for :func:`_build_artifact_zip`.

Exercises the pure builder that turns ``<sid>.phase4.labels.jsonl`` +
the session's Postgres binding into a (`merged.csv`, `metadata.json`)
zip. The Postgres fetcher is stubbed — no app, no auth, no live DB.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from src.web_api.app import _build_artifact_zip


class _StubFetcher:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls: list[list[str]] = []

    def fetch_originals(self, pks: list[str]) -> dict[str, str]:
        self.calls.append(list(pks))
        return {pk: self._mapping[pk] for pk in pks if pk in self._mapping}


def _seed(
    tmp_path: Path,
    sid: str,
    *,
    labels: list[dict],
    phase4_meta: dict,
    phase2_meta: dict | None = None,
    canonical_first_row: dict | None = None,
) -> None:
    (tmp_path / f"{sid}.phase4.labels.jsonl").write_text(
        "\n".join(json.dumps(row) for row in labels) + "\n",
        encoding="utf-8",
    )
    (tmp_path / f"{sid}.phase4.meta.json").write_text(
        json.dumps(phase4_meta), encoding="utf-8"
    )
    if phase2_meta is not None:
        (tmp_path / f"{sid}.phase2.meta.json").write_text(
            json.dumps(phase2_meta), encoding="utf-8"
        )
    if canonical_first_row is not None:
        (tmp_path / f"{sid}.jsonl").write_text(
            json.dumps(canonical_first_row) + "\n", encoding="utf-8"
        )


def _read_zip(payload: bytes) -> tuple[list[dict[str, str]], dict]:
    buf = io.BytesIO(payload)
    with zipfile.ZipFile(buf) as zf:
        assert set(zf.namelist()) == {"merged.csv", "metadata.json"}
        csv_text = zf.read("merged.csv").decode("utf-8")
        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    return rows, meta


def test_builds_csv_and_metadata_for_a_typical_session(tmp_path: Path) -> None:
    sid = "s1"
    _seed(
        tmp_path,
        sid,
        labels=[
            {"pk": "64094054-30006713-14-pre-p", "verdict": "KEEP", "p_keep": 0.99},
            {"pk": "58672771-30003755-7-pre-p", "verdict": "KEEP", "p_keep": 0.98},
            {"pk": "11111111-22222222-3-in-p", "verdict": "DROP", "p_keep": 0.10},
        ],
        phase4_meta={
            "ts": "2026-05-19T08:45:53Z",
            "threshold": 0.8,
            "eval_metrics": {"precision_at_threshold": 0.803279},
        },
        canonical_first_row={
            "turn": 1,
            "timestamp": "2026-05-18T12:34:55Z",
            "session_id": sid,
        },
    )
    fetcher = _StubFetcher(
        {
            "64094054-30006713-14-pre-p": "电话泄漏的内容",
            "58672771-30003755-7-pre-p": "another transcript",
            "11111111-22222222-3-in-p": "drop row content",
        }
    )

    payload = _build_artifact_zip(tmp_path, sid, "Parent Full", "complaint", fetcher)

    rows, meta = _read_zip(payload)
    assert [r["user_id"] for r in rows] == ["64094054", "58672771", "11111111"]
    assert [r["counselor_id"] for r in rows] == ["30006713", "30003755", "22222222"]
    assert [r["label"] for r in rows] == ["KEEP", "KEEP", "DROP"]
    assert rows[0]["original_content"] == "电话泄漏的内容"

    assert meta["session_id"] == sid
    assert meta["scope"] == "Parent Full"
    assert meta["query"] == "complaint"
    assert meta["session_start"] == "2026-05-18T12:34:55Z"
    assert meta["session_end"] == "2026-05-19T08:45:53Z"
    assert meta["n_chunks_total"] == 3
    assert meta["n_keep"] == 2
    assert meta["n_drop"] == 1
    assert meta["threshold"] == 0.8
    assert meta["precision_at_threshold"] == 0.803279
    assert meta["n_missing_content"] == 0


def test_missing_postgres_rows_emit_empty_content_and_count(tmp_path: Path) -> None:
    sid = "s2"
    _seed(
        tmp_path,
        sid,
        labels=[
            {"pk": "a-b-1-pre-p", "verdict": "KEEP", "p_keep": 0.9},
            {"pk": "c-d-2-pre-p", "verdict": "KEEP", "p_keep": 0.9},
            {"pk": "e-f-3-pre-p", "verdict": "DROP", "p_keep": 0.1},
        ],
        phase4_meta={"ts": "2026-01-01T00:00:00Z", "threshold": 0.5},
    )
    fetcher = _StubFetcher({"a-b-1-pre-p": "found"})

    payload = _build_artifact_zip(tmp_path, sid, "Foo", "q", fetcher)
    rows, meta = _read_zip(payload)

    assert rows[0]["original_content"] == "found"
    assert rows[1]["original_content"] == ""
    assert rows[2]["original_content"] == ""
    assert meta["n_missing_content"] == 2
    assert meta["n_keep"] == 2
    assert meta["n_drop"] == 1


def test_falls_back_to_phase2_ts_when_canonical_jsonl_missing(tmp_path: Path) -> None:
    sid = "s3"
    _seed(
        tmp_path,
        sid,
        labels=[{"pk": "u-c-1-pre-p", "verdict": "KEEP", "p_keep": 0.9}],
        phase4_meta={"ts": "2026-02-02T02:02:02Z"},
        phase2_meta={"ts": "2026-02-01T00:00:00Z", "query": "ignored-here"},
    )
    fetcher = _StubFetcher({"u-c-1-pre-p": "x"})

    _rows, meta = _read_zip(_build_artifact_zip(tmp_path, sid, "Foo", "q", fetcher))

    assert meta["session_start"] == "2026-02-01T00:00:00Z"
    assert meta["session_end"] == "2026-02-02T02:02:02Z"


def test_malformed_pk_emits_empty_id_columns(tmp_path: Path) -> None:
    sid = "s4"
    _seed(
        tmp_path,
        sid,
        labels=[{"pk": "no-dashes-shape", "verdict": "KEEP", "p_keep": 0.9}],
        phase4_meta={"ts": "2026-01-01T00:00:00Z"},
    )
    fetcher = _StubFetcher({"no-dashes-shape": "content"})

    rows, _meta = _read_zip(_build_artifact_zip(tmp_path, sid, None, "", fetcher))

    assert rows[0]["user_id"] == ""
    assert rows[0]["counselor_id"] == ""
    assert rows[0]["original_content"] == "content"
