"""``python -m src.apply <session_id>`` — Phase 4 entry point.

Headless smoke / batch runner. Two paths:

- Plain: ``python -m src.apply <sid>`` — train + apply, using
  ``apply.confidence_threshold`` from config. Requires
  ``--auto-accept`` since this CLI does not run a threshold review.
- Reuse: ``python -m src.apply <sid> --classifier <path>`` — load a
  persisted classifier, verify the rubric pin, and apply to the
  session's current Phase 2 cohort.

Real reviews go through the web UI's calibration screen, which
exposes the threshold slider and the PR curve.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .errors import ApplyError
from .runner import finalize_apply, run_apply_reuse, run_apply_train


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(prog="python -m src.apply")
    parser.add_argument(
        "session",
        help="Session id (e.g. 629f9dc3-ec24-...) — Phase 3 must have finalised.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing session sidecars (default: runs)",
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help=(
            "Path to a persisted phase4.classifier.json. Triggers the "
            "reuse path: rubric-pin-verified apply against the current "
            "Phase 2 cohort, no retraining."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the threshold (default: apply.confidence_threshold "
            "from config, or the persisted classifier's threshold on the "
            "reuse path)."
        ),
    )
    parser.add_argument(
        "--auto-accept",
        action="store_true",
        help=(
            "Train, evaluate, and apply without an operator threshold "
            "review. Required for the train path; ignored on --classifier."
        ),
    )
    parser.add_argument(
        "--allow-low-precision",
        action="store_true",
        help=(
            "Proceed even if eval precision is below the configured "
            "apply.min_precision bar. Records operator_decision='override_low_precision'."
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

    if args.classifier is not None:
        return _run_reuse(args, runs_dir=runs_dir)
    return _run_train(args, runs_dir=runs_dir)


def _run_train(args: argparse.Namespace, *, runs_dir: Path) -> int:
    if not args.auto_accept:
        print(
            "apply: this CLI does not run the threshold review loop. "
            "Re-run with --auto-accept to apply at apply.confidence_threshold, "
            "or use the web UI for the full operator workflow.",
            file=sys.stderr,
        )
        return 1
    try:
        state = run_apply_train(args.session, runs_dir=runs_dir)
        state = finalize_apply(
            state,
            runs_dir=runs_dir,
            threshold=args.threshold,
            allow_low_precision=args.allow_low_precision,
        )
    except ApplyError as exc:
        print(f"apply: {exc}", file=sys.stderr)
        return 2
    return _print_summary(state)


def _run_reuse(args: argparse.Namespace, *, runs_dir: Path) -> int:
    classifier_path = Path(args.classifier)
    if not classifier_path.exists():
        print(
            f"apply: --classifier path does not exist: {classifier_path}",
            file=sys.stderr,
        )
        return 2
    try:
        state = run_apply_reuse(
            args.session,
            runs_dir=runs_dir,
            classifier_path=classifier_path,
            threshold=args.threshold,
            allow_low_precision=args.allow_low_precision,
        )
    except ApplyError as exc:
        print(f"apply: {exc}", file=sys.stderr)
        return 2
    return _print_summary(state)


def _print_summary(state) -> int:
    write = state.write_result
    metadata = state.classifier_metadata
    assert write is not None and metadata is not None
    print(f"  classifier: {write.classifier_path}")
    print(f"  eval:       {write.eval_path}")
    print(f"  labels:     {write.labels_path}")
    print(f"  meta:       {write.meta_path}")
    print(f"  details:    {write.details_path}")
    print(
        f"  threshold={metadata.threshold:.3f} "
        f"eval_precision={metadata.eval_metrics.precision_at_threshold:.3f} "
        f"decision={state.operator_decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
