"""``python -m src.refine <session_id>`` — Phase 3 entry point.

Headless smoke / batch runner. Useful for iterating on the
meta-prompt (``--derive-only``) and for running Phase 3 from CI
without the TUI's review loop (``--auto-accept``).

Real reviews go through the TUI / web — those expose the editor and
the verdict-review panel that this CLI deliberately omits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .errors import RefineError
from .runner import (
    _open_default_fetcher,
    finalize_refine,
    run_refine_derive,
    run_refine_judge,
)


def _print_progress(done: int, total: int) -> None:
    pct = (done / total * 100.0) if total else 0.0
    print(
        f"\rrefine.judge: {done}/{total} ({pct:5.1f}%)",
        end="",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(prog="python -m src.refine")
    parser.add_argument(
        "session",
        help="Session id (e.g. 13252e21...) or path to runs/<id>.jsonl",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing session sidecars (default: runs)",
    )
    parser.add_argument(
        "--derive-only",
        action="store_true",
        help=(
            "Run derive only; print the rubric markdown to stdout, do "
            "not call the judge. Use this to iterate on the meta-prompt."
        ),
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help=(
            "Skip the operator review step — derive, judge, then finalise "
            "without a human in the loop. Useful for CI / smoke tests."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable INFO-level logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runs_dir = Path(args.runs_dir)

    try:
        state = run_refine_derive(args.session, runs_dir=runs_dir)
    except RefineError as exc:
        print(f"refine.derive: {exc}", file=sys.stderr)
        return 2

    derive = state.derive_result
    assert derive is not None  # run_refine_derive guarantees this
    print(
        f"refine.derive: session={state.session_id} "
        f"checks={[c.id for c in derive.metadata.checks]} "
        f"attempts={derive.attempts} "
        f"latency={derive.latency_ms:.1f}ms",
        file=sys.stderr,
    )

    if args.derive_only:
        # Dump the rubric markdown to stdout. Use this in pipelines
        # like `python -m src.refine <sid> --derive-only > rubric.md`.
        sys.stdout.write(derive.rubric_text)
        if not derive.rubric_text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if not args.auto_accept:
        print(
            "refine: this CLI does not run the review loop. Re-run with "
            "--auto-accept to ship without review, or use the TUI / web "
            "for the full operator workflow.",
            file=sys.stderr,
        )
        return 1

    try:
        fetcher = _open_default_fetcher()
    except RefineError as exc:
        print(f"refine.judge: {exc}", file=sys.stderr)
        return 2

    try:
        state = run_refine_judge(
            state,
            runs_dir=runs_dir,
            fetcher=fetcher,
            progress=_print_progress,
        )
        print()  # newline after progress
        result = state.judge_result
        assert result is not None
        print(
            f"refine.judge: verdicts={len(result.verdicts)} "
            f"parse_errors={result.parse_error_count} "
            f"api_errors={result.api_error_count} "
            f"latency={result.total_latency_ms:.0f}ms",
            file=sys.stderr,
        )

        state = finalize_refine(state, runs_dir=runs_dir, operator_decision="agree")
    except RefineError as exc:
        print(f"refine: {exc}", file=sys.stderr)
        return 2
    finally:
        close = getattr(fetcher, "close", None)
        if callable(close):
            close()

    write = state.write_result
    assert write is not None
    print(f"  prompt:  {write.prompt_path}")
    print(f"  rubric:  {write.rubric_path}")
    print(f"  evidence:{write.evidence_path}")
    print(f"  meta:    {write.meta_path}")
    print(f"  details: {write.details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
