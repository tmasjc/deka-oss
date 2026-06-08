"""Prompt assembly: convert SessionState into OpenAI chat messages.

Follows the assembly order in ``harness/prompts/REFLECTION.md``:
system prompt, progress log, current-turn evidence, reflection
instructions with JSON output schema.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.paths import prompt_path

from .errors import PromptAssemblyError

if TYPE_CHECKING:
    from src.search.config import SearchConfig
    from src.search.evidence import EvidenceRow, EvidenceTable
    from src.session.state import SessionState, TurnRecord

_COMPRESS_THRESHOLD = 8
_RECENT_WINDOW = 5

# JSON output schema appended to the reflection instructions.
# Reflection is a reasoning trace, not a tuning controller — there is
# no ``prescribe`` block. The schema is enforced by ``ReflectionOutput``
# with ``extra="forbid"``: a stray ``prescribe`` field will hard-fail
# parse. The optional ``path_drop_recommendation`` carries a structured
# direct-apply path-drop nomination grounded in cumulative session
# evidence; on operator ``[a]pply`` the drop is applied immediately.
_JSON_SCHEMA_APPENDIX = """

## Output format

You MUST respond with a single JSON object (no markdown fences, no
extra text):

```json
{
  "observe": "<raw facts>",
  "diagnose": "<root-cause interpretation>",
  "hypothesis": "If the next turn shows [pattern], then [diagnosis correct/wrong], because [reasoning]",
  "previous_hypothesis_verdict": "CONFIRMED" | "REFUTED" | null,
  "path_drop_recommendation": {
    "path": "dense" | "sparse",
    "reason": "<short cumulative-evidence justification>",
    "confidence": "low" | "medium" | "high"
  } | null,
  "status": "CONTINUE",
  "turns_to_converge": null
}
```

- `previous_hypothesis_verdict`: null on the first turn; otherwise
  CONFIRMED or REFUTED based on whether this turn's evidence matches
  the previous turn's hypothesis.
- `path_drop_recommendation`: null on most turns. Populate ONLY when
  the PATH-DROP RECOMMENDATION criteria above hold (turn 2 or later,
  2+ consecutive turns of consistent NOT_FIT contribution from the
  named path, no FIT row sole-sourced by that path in the visible
  log, path is currently active and not exhausted). The apply is
  mechanical — do NOT also restate the recommendation as prose
  inside `diagnose`.
- `status`: set to ``"CONVERGED"`` only when both halves of the dual
  gate hold (Precision@K ≥ harvest.precision_at_k AND cumulative
  unique FIT PKs ≥ harvest.min_fit). Else ``"CONTINUE"``.
- `turns_to_converge`: set to the turn count when status is
  ``"CONVERGED"``, else null.
- DO NOT include a ``prescribe`` block, ``rrf_k``, ``top_k``,
  ``per_path_limit``, ``active_paths``, or any other config fields —
  the session config is locked; extra keys will be rejected.
"""


def load_prompt_block(path: Path) -> str:
    """Extract the first triple-backtick code block from a Markdown file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptAssemblyError(f"Prompt file not found: {path}") from exc

    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match is None:
        raise PromptAssemblyError(f"No code block found in {path}")
    return match.group(1).strip()


def load_system_prompt(repo_root: Path | None = None) -> str:
    path = (
        repo_root / "harness" / "prompts" / "SYSTEM.md"
        if repo_root is not None
        else prompt_path("SYSTEM.md")
    )
    return load_prompt_block(path)


def load_reflection_instructions(repo_root: Path | None = None) -> str:
    path = (
        repo_root / "harness" / "prompts" / "REFLECTION.md"
        if repo_root is not None
        else prompt_path("REFLECTION.md")
    )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptAssemblyError(f"Prompt file not found: {path}") from exc

    # Extract the last code block (the three-phase instructions)
    blocks = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if len(blocks) < 4:
        raise PromptAssemblyError(
            f"Expected at least 4 code blocks in {path}, found {len(blocks)}"
        )
    return blocks[3].strip() + _JSON_SCHEMA_APPENDIX


def build_messages(
    state: "SessionState",
    system_prompt: str,
    reflection_instructions: str,
) -> list[dict[str, str]]:
    """Build the 3-message list for the OpenAI chat completions API."""
    context = _build_context(state)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
        {"role": "user", "content": reflection_instructions},
    ]


def _build_context(state: "SessionState") -> str:
    """Render the dynamic context: progress log + current-turn evidence."""
    parts: list[str] = []

    # Progress log (all turns except the latest, which is the current turn)
    if len(state.turns) > 1:
        parts.append(_render_progress_log(state.turns[:-1]))
    else:
        parts.append("## PROGRESS LOG (read-only)\n\nNo previous turns.")

    # Current turn evidence
    current = state.turns[-1]
    parts.append(_render_current_turn(current, state))

    return "\n\n".join(parts)


def _render_progress_log(turns: list["TurnRecord"]) -> str:
    lines = ["## PROGRESS LOG (read-only — do not modify past entries)\n"]

    if len(turns) + 1 > _COMPRESS_THRESHOLD:
        # Compress early turns
        cutoff = len(turns) - _RECENT_WINDOW
        for turn in turns[:cutoff]:
            lines.append(_render_turn_summary(turn))
        lines.append("")
        for turn in turns[cutoff:]:
            lines.append(_render_turn_full(turn))
    else:
        for turn in turns:
            lines.append(_render_turn_full(turn))

    return "\n".join(lines)


def _render_turn_summary(turn: "TurnRecord") -> str:
    cfg = _format_config(turn.config)
    verdict = "N/A"
    if turn.reflection and turn.reflection.get("previous_hypothesis_verdict"):
        verdict = turn.reflection["previous_hypothesis_verdict"]
    audit_marker = ", AUDIT" if turn.audit_turn else ""
    return (
        f"Turn {turn.turn_number}: Config={cfg}, "
        f"P@K={turn.precision:.2f}, Verdict={verdict}{audit_marker}"
    )


def _render_turn_full(turn: "TurnRecord") -> str:
    cfg = _format_config(turn.config)
    bd = turn.breakdown
    lines = [
        f"Turn {turn.turn_number}:",
        f'  Query: "{turn.query}"',
        f"  Config: {cfg}",
        f"  Results: {_total_from_breakdown(bd)} chunks returned, "
        f"{_fit_from_breakdown(bd)} FIT, {_not_fit_from_breakdown(bd)} NOT_FIT",
        f"  Precision@K: {turn.precision:.2f}",
        "  Per-path breakdown:",
    ]
    for key in ("dense_only", "sparse_only", "multi_path"):
        entry = bd.get(key, {"total": 0, "fit": 0, "not_fit": 0})
        lines.append(
            f"    {key}: {entry['total']} ({entry['fit']} FIT, {entry['not_fit']} NOT_FIT)"
        )

    if turn.reflection:
        r = turn.reflection
        lines.append(f"  Diagnosis: {r.get('diagnose', 'N/A')}")
        lines.append(f"  Hypothesis: {r.get('hypothesis', 'N/A')}")
        verdict = r.get("previous_hypothesis_verdict", "N/A") or "N/A"
        lines.append(f"  Verdict on previous hypothesis: {verdict}")
        # Deliberately omitted: ``path_drop_recommendation``. Feeding
        # the agent its own past recommendations alongside the operator's
        # apply/ignore signal would risk teaching it to suppress
        # recommendations to avoid being declined. The recommendation
        # rests on the rated turn history, not on prior recommendation
        # outcomes — see audit_recommendation_field.md (the spec doc
        # named for the predecessor feature).
    else:
        lines.append("  Diagnosis: (no reflection recorded — converged or manual)")
    lines.append(f"  Audit turn: {'true' if turn.audit_turn else 'false'}")

    return "\n".join(lines) + "\n"


def _render_current_turn(turn: "TurnRecord", state: "SessionState") -> str:
    cfg = _format_config(turn.config)
    table = turn.evidence_table
    total = len(table.rows)
    fit_count = sum(1 for r in table.rows if r.rating == "FIT")
    not_fit_count = sum(1 for r in table.rows if r.rating == "NOT_FIT")
    discard_count = sum(1 for r in table.rows if r.rating == "DISCARD")

    prev_precision = state.turns[-2].precision if len(state.turns) >= 2 else None
    delta = turn.precision - prev_precision if prev_precision is not None else None

    lines = [
        f"## CURRENT TURN (Turn {turn.turn_number})\n",
        f'Query: "{turn.query}"\n',
        f"Config used: {cfg}\n",
    ]
    # Read what was actually filtered at search time — `state.seen_pks` is
    # populated by `complete_turn` *before* reflection runs, so its size
    # reflects post-turn state (includes this turn's ratings) and would
    # wrongly claim "N chunks excluded" on turn 1. The diagnostics snapshot
    # taken inside run_search is the ground truth for this turn's pool.
    seen_size = 0
    if table.search_diagnostics:
        seen_size = table.search_diagnostics.get("seen_set_size", 0)
    if seen_size > 0:
        lines.append(
            f"Seen set: {seen_size} chunks excluded from this turn's candidate "
            "pool (dedup across prior turns).\n"
        )
        depth_block = _render_depth_accounting(table.search_diagnostics)
        if depth_block:
            lines.append(depth_block)
            lines.append("")
    lines.extend(
        [
            "Results with human ratings:\n",
            "| rank | chunk_id | rating | source_paths | dense_score | sparse_score |",
            "|------|----------|--------|--------------|-------------|--------------|",
        ]
    )
    for row in table.rows:
        lines.append(_render_evidence_row(row))

    lines.append("")
    lines.append("Summary:")
    discard_part = f", DISCARD: {discard_count}" if discard_count else ""
    lines.append(
        f"  Total: {total}, FIT: {fit_count}, NOT_FIT: {not_fit_count}{discard_part}"
    )

    if prev_precision is not None:
        direction = (
            "improved"
            if delta > 0.005
            else "degraded"
            if delta < -0.005
            else "unchanged"
        )
        lines.append(
            f"  Precision@K: {turn.precision:.2f} (previous turn: {prev_precision:.2f})"
        )
        lines.append(f"  Delta: {delta:+.2f} ({direction})")
    else:
        lines.append(f"  Precision@K: {turn.precision:.2f} (first turn)")

    probe_block = _render_probe_stats(table.search_diagnostics)
    if probe_block:
        lines.append("")
        lines.append(probe_block)

    candidate_block = _render_per_path_candidates(table)
    if candidate_block:
        lines.append("")
        lines.append(candidate_block)

    return "\n".join(lines)


def _render_depth_accounting(diagnostics: dict | None) -> str:
    """Per-path arithmetic of the candidate pool after dedup.

    Turns the server-side dedup counts (``filtered_by_seen``) into a
    rank-range statement so reflection can reason about depth without
    inventing the arithmetic. Caller is responsible for gating on
    ``seen_set_size > 0`` — this function renders unconditionally.
    """
    if not diagnostics:
        return ""
    probes = diagnostics.get("probes")
    if not probes:
        return ""
    per_path_limit = diagnostics.get("per_path_limit")
    if not isinstance(per_path_limit, int) or per_path_limit <= 0:
        return ""

    lines = [
        'Depth accounting (candidate pool after dedup — "new" = unseen '
        "chunks each path surfaced this turn):",
    ]
    for path in ("dense", "sparse"):
        stats = probes.get(path, {})
        lines.append(f"  {path:7s}: {_render_depth_path(stats, per_path_limit)}")
    return "\n".join(lines)


def _render_depth_path(stats: dict, per_path_limit: int) -> str:
    if stats.get("skipped"):
        return "skipped (e.g. empty embedding)"
    hit_count = stats.get("hit_count", 0)
    filtered = stats.get("filtered_by_seen", 0)
    if hit_count == 0:
        if filtered > 0:
            return (
                f"per_path_limit={per_path_limit}, filtered={filtered}, "
                f"new=0 (exhausted — all top-{per_path_limit} already rated)"
            )
        return (
            f"per_path_limit={per_path_limit}, filtered=0, "
            "new=0 (path empty for this query)"
        )
    start = filtered + 1
    end = filtered + hit_count
    return (
        f"per_path_limit={per_path_limit}, filtered={filtered}, "
        f"new={hit_count} at ranks {start}–{end}"
    )


def _render_probe_stats(diagnostics: dict | None) -> str:
    """Render the per-path probe summary, or empty string if unavailable."""
    if not diagnostics:
        return ""
    probes = diagnostics.get("probes")
    if not probes:
        return ""
    lines = ["Per-path probe (independent retrieval before fusion):"]
    for path in ("dense", "sparse"):
        stats = probes.get(path, {})
        lines.append(f"  {path:7s}: {_render_probe_path(stats)}")
    return "\n".join(lines)


def _render_per_path_candidates(table: "EvidenceTable") -> str:
    """Render per-path top-3 candidate ratings (those not in the fused top-K).

    Lets the agent distinguish "this path's strong candidates were FIT but
    lost in fusion" (a ranking issue — consider rrf_k) from "this path
    is genuinely noisy for this query" (consider deactivating).

    Inactive paths still surface their candidates — the probe always runs
    on both — but are labelled ``(inactive)`` so the agent knows they
    didn't contribute to the fused top-K. This is how the agent judges
    re-activation: if an inactive path's candidates are FIT, re-adding it
    may help; if they're NOT_FIT, the deactivation is working.
    """
    candidates = table.per_path_candidates or {}
    if not any(candidates.get(p) for p in ("dense", "sparse")):
        return ""

    active_paths = table.config.active_paths
    lines = [
        "Per-path top-3 candidates (NOT in fused results — rated separately):",
    ]
    for path in ("dense", "sparse"):
        label = f"{path:7s}" if path in active_paths else f"{path:7s}(inactive)"
        path_cands = candidates.get(path) or []
        if not path_cands:
            lines.append(f"  {label}: (none — top 3 either in fused or path empty)")
            continue
        fit = sum(1 for c in path_cands if c.rating == "FIT")
        not_fit = sum(1 for c in path_cands if c.rating == "NOT_FIT")
        unrated = len(path_cands) - fit - not_fit
        per_chunk = "  ".join(
            f"#{c.rank_in_path} {c.chunk_id}={c.rating or '—'}" for c in path_cands
        )
        summary = f"{fit}/{len(path_cands)} FIT"
        if unrated:
            summary += f" ({unrated} unrated)"
        lines.append(f"  {label}: {per_chunk}  → {summary}")
    return "\n".join(lines)


def _render_probe_path(stats: dict) -> str:
    if stats.get("skipped"):
        return "skipped (e.g. empty embedding)"
    hit_count = stats.get("hit_count", 0)
    filtered = stats.get("filtered_by_seen", 0)
    if hit_count == 0:
        if filtered > 0:
            # Exhausted: path has no NEW material. Distinguish from a
            # genuinely dead path so reflection doesn't prescribe a drop.
            return f"0 hits (seen={filtered} filtered — path likely exhausted)"
        return "0 hits"
    score_min = stats.get("score_min")
    score_max = stats.get("score_max")
    score_mean = stats.get("score_mean")
    return (
        f"hits={hit_count}  "
        f"scores {score_min:.3f}–{score_max:.3f}  "
        f"mean={score_mean:.3f}"
    )


def _render_evidence_row(row: "EvidenceRow") -> str:
    paths = ", ".join(row.source_paths)
    d = f"{row.scores.get('dense', 0.0):.4f}"
    s = f"{row.scores.get('sparse', 0.0):.4f}"
    return f"| {row.rank} | {row.chunk_id} | {row.rating} | {paths} | {d} | {s} |"


def _format_config(config: "SearchConfig") -> str:
    paths = ",".join(sorted(config.active_paths))
    return (
        f"RRFRanker(k={config.rrf_k}), "
        f"per_path_limit={config.per_path_limit}, top_k={config.top_k}, "
        f"active_paths=[{paths}]"
    )


def _total_from_breakdown(bd: dict[str, dict[str, int]]) -> int:
    return sum(v["total"] for v in bd.values())


def _fit_from_breakdown(bd: dict[str, dict[str, int]]) -> int:
    return sum(v["fit"] for v in bd.values())


def _not_fit_from_breakdown(bd: dict[str, dict[str, int]]) -> int:
    return sum(v["not_fit"] for v in bd.values())
