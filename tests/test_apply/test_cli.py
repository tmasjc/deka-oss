"""Smoke tests for the python -m src.apply CLI."""

from __future__ import annotations

import json
from pathlib import Path


from src.apply import __main__ as cli
from tests.test_apply.test_runner import _make_fetcher, _seed_session, _test_cfg


def test_cli_requires_auto_accept_on_train_path(tmp_path: Path, monkeypatch, capsys):
    _seed_session(tmp_path)
    rc = cli.main(["demo", "--runs-dir", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "auto-accept" in err


def test_cli_train_path_with_auto_accept(tmp_path: Path, monkeypatch, capsys):
    _seed_session(tmp_path)
    cfg = _test_cfg()
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(cli, "run_apply_train", _bind_runner_with_cfg(cfg, fetcher))
    monkeypatch.setattr(cli, "finalize_apply", _bind_finalize_with_fetcher(fetcher))

    rc = cli.main(["demo", "--runs-dir", str(tmp_path), "--auto-accept"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "labels:" in out
    assert (tmp_path / "demo.phase4.labels.jsonl").exists()


def test_cli_reuse_path_refuses_on_sha_mismatch(tmp_path: Path, monkeypatch, capsys):
    _seed_session(tmp_path)
    # First produce a real classifier on disk.
    from src.apply.runner import finalize_apply, run_apply_train

    fetcher = _make_fetcher(tmp_path)
    state = run_apply_train(
        "demo",
        runs_dir=tmp_path,
        cfg=_test_cfg(),
        embeddings_fetcher=fetcher,
    )
    finalize_apply(state, runs_dir=tmp_path, embeddings_fetcher=fetcher)
    classifier_path = tmp_path / "demo.phase4.classifier.json"
    payload = json.loads(classifier_path.read_text())
    payload["prompt_sha256"] = "f" * 64
    classifier_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _test_cfg()
    monkeypatch.setattr(cli, "run_apply_reuse", _bind_reuse_with_cfg(cfg, fetcher))

    rc = cli.main(
        [
            "demo",
            "--runs-dir",
            str(tmp_path),
            "--classifier",
            str(classifier_path),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "Rubric" in err or "rubric" in err


def test_cli_missing_classifier_path(tmp_path: Path, capsys):
    _seed_session(tmp_path)
    rc = cli.main(
        [
            "demo",
            "--runs-dir",
            str(tmp_path),
            "--classifier",
            str(tmp_path / "does-not-exist.json"),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "does not exist" in err


# ---------------------------------------------------------------------------
# Helpers — wrap the runner entry points so CLI tests don't need a real
# Milvus or config.yaml.
# ---------------------------------------------------------------------------


def _bind_runner_with_cfg(cfg, fetcher):
    from src.apply.runner import run_apply_train

    def wrapper(*args, **kwargs):
        kwargs.setdefault("cfg", cfg)
        kwargs.setdefault("embeddings_fetcher", fetcher)
        return run_apply_train(*args, **kwargs)

    return wrapper


def _bind_finalize_with_fetcher(fetcher):
    from src.apply.runner import finalize_apply

    def wrapper(*args, **kwargs):
        kwargs.setdefault("embeddings_fetcher", fetcher)
        return finalize_apply(*args, **kwargs)

    return wrapper


def _bind_reuse_with_cfg(cfg, fetcher):
    from src.apply.runner import run_apply_reuse

    def wrapper(*args, **kwargs):
        kwargs.setdefault("cfg", cfg)
        kwargs.setdefault("embeddings_fetcher", fetcher)
        return run_apply_reuse(*args, **kwargs)

    return wrapper
