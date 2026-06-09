"""``python -m src.anchor <session_id>`` — Phase 2 entry point.

Runs the span-anchored retrieval end-to-end: load, calibrate, LOO
gate, retrieve (iterator widening), write.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.config_loader import ConfigFileError, load_session_overrides

from .config import RadiusScheme
from .errors import AnchorError
from .runner import run_anchor

log = logging.getLogger(__name__)

_BOLD_RED = "\033[1;31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _resolve_overrides_location(session_arg: str, runs_dir: Path) -> tuple[str, Path]:
    """Mirror :func:`src.replay.loader._resolve_paths` for the overrides
    sidecar. Returns ``(session_id, effective_runs_dir)``.

    When the CLI receives a path, the sidecar lives next to the
    canonical jsonl. When it receives a bare id, ``--runs-dir`` is used
    as-is — same convention as the canonical-log lookup.
    """
    target = Path(session_arg)
    name = target.name
    if name.endswith(".details.jsonl"):
        return name[: -len(".details.jsonl")], target.parent
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")], target.parent
    return session_arg, runs_dir


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(prog="python -m src.anchor")
    parser.add_argument(
        "session",
        help="Session id (e.g. 13252e21...) or path to runs/<id>.jsonl",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing canonical session JSONL files (default: runs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "search_iterator page size. Must stay ≤ 16384 "
            "(Milvus MAX_BATCH_SIZE). Omit to use harvest.batch_size "
            "from config.yaml."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help=(
            "Per-FIT safety cap on cumulative hits. The iterator normally "
            "stops when the page's last hit passes T'; max_k guards "
            "against a corpus denser than expected. Omit to use "
            "harvest.max_k from config.yaml."
        ),
    )
    parser.add_argument(
        "--radius-scheme",
        choices=[s.value for s in RadiusScheme],
        default=None,
        help=(
            "Main-pass threshold scheme. 'per_fit' (legacy) uses each "
            "FIT's own T'_i = T + δ_i; 'decoupled' (issue #20) uses the "
            "session-wide T'_out = T + min(δ). LOO recovery always "
            "uses per-FIT T'_i regardless. Omit to use "
            "harvest.radius_scheme from config.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute T, δ, T', LOO; skip main retrieve pass + sidecars",
    )
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help=(
            "Skip the Phase 1 dual-gate convergence check. Default is to "
            "abort when the session has not converged (latest P@K below "
            "the threshold or cumulative FIT count below the floor); "
            "pass this flag to run Phase 2 anyway — useful for replaying "
            "historical sessions or experimenting with thin cohorts."
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

    scheme_override = RadiusScheme(args.radius_scheme) if args.radius_scheme else None

    runs_dir = Path(args.runs_dir)
    session_id, overrides_runs_dir = _resolve_overrides_location(args.session, runs_dir)
    try:
        overrides = load_session_overrides(session_id, overrides_runs_dir, user_id=None)
    except ConfigFileError as exc:
        log.warning(
            "session overrides sidecar unreadable for %s: %s "
            "(proceeding with YAML defaults)",
            session_id,
            exc,
        )
        overrides = {}

    try:
        result = run_anchor(
            args.session,
            runs_dir=runs_dir,
            batch_size=args.batch_size,
            max_k=args.max_k,
            radius_scheme=scheme_override,
            dry_run=args.dry_run,
            allow_unconverged=args.allow_unconverged,
            harvest_overrides=overrides.get("harvest"),
        )
    except AnchorError as exc:
        print(f"phase2: {exc}", file=sys.stderr)
        return 2

    recovery = result.recovery
    calib = result.calibration
    from src.anchor.threshold import distance_summary

    delta_s = distance_summary(calib.deltas)
    tprime_s = distance_summary(calib.T_primes)
    print(
        f"phase2: session={result.inputs.session_id} "
        f"scheme={result.radius_scheme.value} "
        f"FITs={recovery.total} T={calib.T:.4f} "
        f"δ=[{delta_s['min']:.4f}–{delta_s['median']:.4f}–{delta_s['max']:.4f}] "
        f"T'=[{tprime_s['min']:.4f}–{tprime_s['median']:.4f}–{tprime_s['max']:.4f}] "
        f"T'_out={calib.T_prime_out:.4f} "
        f"recovered={recovery.recovered}/{recovery.total} ({recovery.verdict}) "
        f"retrieved={len(result.retrieval.candidates)} "
        f"intrusions={result.not_fit_intrusions}"
    )
    if recovery.verdict == "FLAGGED":
        missed = [(p.fit_pk, p.fit_chunk_id) for p in recovery.missed_fits]
        print(
            f"  WARNING: LOO verdict FLAGGED — review missed FITs: {missed}",
            file=sys.stderr,
        )
    dropped = result.quality_gate_dropped
    if dropped:
        n_orig = len(dropped) + result.calibration.n_fit
        print(
            f"{_YELLOW}QUALITY GATE dropped {len(dropped)}/{n_orig} FIT(s):{_RESET}",
            file=sys.stderr,
        )
        for drop_rec in dropped:
            print(
                f"  {drop_rec['fit_chunk_id']}  δ={drop_rec['delta']:.4f}  "
                f"[{','.join(drop_rec['reasons'])}]",
                file=sys.stderr,
            )
    missing_cohort = [
        r for r in result.cohort_consistency if not r["own_chunk_retained"]
    ]
    if missing_cohort:
        print(
            f"{_YELLOW}COHORT CONSISTENCY: {len(missing_cohort)}/"
            f"{len(result.cohort_consistency)} FIT(s) own chunk absent "
            f"from output:{_RESET}",
            file=sys.stderr,
        )
        for miss in missing_cohort:
            print(
                f"  {miss['fit_chunk_id']}  (pk={miss['fit_pk']})",
                file=sys.stderr,
            )
    exhausted = [p for p in result.retrieval.per_fit_pages if p.budget_exhausted]
    if exhausted:
        names = ", ".join(p.fit_chunk_id for p in exhausted)
        print(
            f"{_BOLD_RED}BUDGET EXHAUSTED on {len(exhausted)} FIT(s) "
            f"({names}) — raise harvest.max_k and re-run to converge.{_RESET}",
            file=sys.stderr,
        )
    if result.write is not None:
        print(f"  jsonl:   {result.write.jsonl_path}")
        print(f"  meta:    {result.write.meta_path}")
        print(f"  details: {result.write.details_path}")
    else:
        print("  (dry-run — no sidecars written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
