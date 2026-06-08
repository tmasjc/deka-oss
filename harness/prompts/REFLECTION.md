# Reflection prompt template

This template is assembled fresh each turn. It has three dynamic
sections (progress log, evidence table, computed metrics) and one
static section (the three-phase instructions).

The reflection agent's role is **reasoning trace, not tuning
controller**. The session config is locked once turn 1 starts; the
only mid-session config change is a path drop. When cumulative
session evidence shows a path is consistently unhelpful, the agent
populates the optional structured `path_drop_recommendation` field;
the operator confirms via a one-keystroke prompt at end-of-turn and
the drop is applied immediately (no audit step). The operator can
also trigger a manual audit via the `p` keypress mid-turn — that
flow rates per-path candidates and enforces Rule B before dropping;
it is independent of the agent's recommendation.

---

## Assembly order

```
1. [STATIC]  System prompt           — from harness/prompts/SYSTEM.md
2. [DYNAMIC] Progress log            — from runs/{session_id}.jsonl
3. [DYNAMIC] Current turn evidence   — constructed by orchestrator
4. [STATIC]  Reflection instructions — below
```

## Dynamic section: Progress log injection

```
## PROGRESS LOG (read-only — do not modify past entries)

{for each past turn, render:}

Turn {n}:
  Query: "{query_text}"
  Config: RRFRanker(k={rrf_k}),
          per_path_limit={per_path_limit}, top_k={top_k},
          active_paths=[{paths}]
  Results: {total} chunks returned, {fit_count} FIT, {not_fit_count} NOT_FIT
  Precision@K: {precision}
  Per-path breakdown:
    dense_only:  {count} ({fit} FIT, {not_fit} NOT_FIT)
    sparse_only: {count} ({fit} FIT, {not_fit} NOT_FIT)
    multi_path:  {count} ({fit} FIT, {not_fit} NOT_FIT)
  Diagnosis: {agent's diagnosis text from that turn}
  Hypothesis: {agent's hypothesis text}
  Verdict on previous hypothesis: {CONFIRMED | REFUTED | N/A}
  Audit turn: {true | false}
    — true means per-path candidates were rated this turn and
      possibly a path was dropped via apply_path_drop. The current
      ``Config`` line above already reflects any drop.
```

For sessions longer than 8 turns, compress turns 1 through (n-5) to
summary-only format (config + precision + verdict). Keep the most
recent 5 turns in full detail. This manages context window pressure
while preserving recent reasoning chain.

## Dynamic section: Current turn evidence

```
## CURRENT TURN (Turn {n})

Query: "{query_text}"

Config used: RRFRanker(k={rrf_k}),
             per_path_limit={per_path_limit}, top_k={top_k},
             active_paths=[{paths}]

Seen set: {N} chunks excluded from this turn's candidate pool
          (dedup across prior turns).
          — omitted on turn 1 when N == 0.

Depth accounting (candidate pool after dedup — "new" = unseen chunks
                  each path surfaced this turn):
  dense  : per_path_limit={L}, filtered={F_d}, new={H_d} at ranks {F_d+1}–{F_d+H_d}
  sparse : per_path_limit={L}, filtered={F_s}, new={H_s} at ranks {F_s+1}–{F_s+H_s}
          — exhausted paths (hit_count=0, filtered>0) render as
            "new=0 (exhausted — all top-{L} already rated)".
          — empty paths (hit_count=0, filtered=0) render as
            "new=0 (path empty for this query)".
          — emitted only when the Seen set line is present.

Results with human ratings:

| rank | chunk_id    | rating  | source_paths     | dense_score | sparse_score |
|------|-------------|---------|------------------|-------------|--------------|
{for each result row}
| {i}  | {chunk_id}  | {rating}| {paths joined}   | {d_score}   | {s_score}    |
{end for}

Summary:
  Total: {total}, FIT: {fit_count}, NOT_FIT: {not_fit_count}{, DISCARD: {discard_count} when any}
  Precision@K: {precision} (previous turn: {prev_precision})
  Delta: {delta} ({improved | degraded | unchanged})
```

### How source_paths is determined

A chunk's source_paths lists every retrieval path that returned it
within its per_path_limit, regardless of which path's score ultimately
dominated in the reranker. This is determined by running each path's
AnnSearchRequest independently and recording which chunk IDs appear
in each result set, before fusion.

### How per-path scores are populated

For each chunk in the fused results:
  - dense_score: COSINE similarity from the dense path (0 if not in
    dense path's result set)
  - sparse_score: IP score from the sparse path (0 if not in sparse
    path's result set)

## Static section: Three-phase reflection instructions

```
## YOUR TASK

Analyze the current turn's results and produce your reflection.
Follow these three phases IN ORDER. Do not skip phases. Output using
the exact headers below.

### OBSERVE

State the raw facts. Do not interpret yet.

- How many FIT vs NOT_FIT?
- What is Precision@K compared to last turn? Is the trend improving,
  degrading, or flat?
- Which source_paths appear on FIT chunks? List them.
- Which source_paths appear on NOT_FIT chunks? List them.
- Are there chunks where multiple paths agree (source_paths contains
  more than one)? What were their ratings?
- Are there score patterns visible in the table? For example:
  "FIT chunks average sparse_score=0.74, NOT_FIT chunks average
  sparse_score=0.21."
- If this is turn 2+, how does the per-path breakdown compare to
  last turn?

### DIAGNOSE

Now interpret. Identify the root cause of the NOT_FIT results.

- Which path is the primary source of noise this turn? (The path
  that contributes the most NOT_FIT-only chunks.)
- Is the problem retrieval (wrong chunks surfaced) or ranking (right
  chunks exist in the candidate pool but ranked below top_k)?
- Compare this turn's per-path performance to the previous turn's.
  Did the previous audit (if any) help the intended path? Did it
  cause regression elsewhere?
- If the previous turn stated a hypothesis, was it CONFIRMED or
  REFUTED by this turn's evidence? State this explicitly with
  supporting numbers.

### HYPOTHESIZE

Form ONE falsifiable hypothesis about the diagnosis.

- State it as: "If the next turn shows [specific evidence pattern],
  then [diagnosis is correct/wrong], because [reasoning grounded in
  the diagnosis]."
- The hypothesis must be falsifiable: you must be able to determine
  from the next turn's results whether it was confirmed or refuted.
- The hypothesis is about evidence, not about parameters — you no
  longer prescribe a config change. Example: "If the next turn shows
  the dense-only NOT_FIT rate stays above 70%, the diagnosis (dense
  is the noise source) is confirmed, because score magnitudes for
  dense-only chunks have been stable across the last 3 turns."

### PATH-DROP RECOMMENDATION

Optionally populate the `path_drop_recommendation` field. **Null on
most turns.** This is a structured nomination to **drop** a path
from `active_paths` for the rest of the session. The operator
confirms via a one-keystroke prompt; on `[a]pply` the drop is
applied immediately — no audit, no candidate rating, no Rule B
safety net. The field is the only sanctioned channel for a drop
recommendation — do NOT restate it as prose inside `diagnose`.

Shape (rendered as JSON in the output):

    "path_drop_recommendation": {
      "path":       "dense" | "sparse",   // currently-active path
      "reason":     "<short cumulative-evidence justification>",
      "confidence": "low" | "medium" | "high"
    }

Because the apply is mechanical, the bar is **higher** than for an
audit recommendation. You are committing to a config change, not to
an investigation. Populate only when ALL of the following hold:

- It is turn 2 or later. Turn 1 has no prior history; leave the field
  null.
- The named path has contributed only NOT_FIT (or near-only NOT_FIT)
  to the fused top-K across the **last 2 or more turns**, judged from
  the rendered progress log's per-path breakdowns.
- The session's cumulative FIT pile has grown without that path's
  help — i.e. the FIT chunks were sole-sourced or co-sourced by other
  paths.
- **Rule B1 self-check.** Scan every FIT row in the visible progress
  log. If any FIT row's `source_paths` is exactly `[<the path you'd
  drop>]` (sole-sourced by that path), do NOT recommend the drop —
  removing the path would silently lose that FIT chunk from future
  fusion. The operator can still manually audit if they want; your
  recommendation must be safe by inspection.
- The path is currently active (listed in `active_paths`).
- The path is NOT exhausted. A probe reading `0 hits (seen=N
  filtered — path likely exhausted)` is *not* a candidate; the
  auto-retry will deepen the per-path pool. Recommend drops on
  paths that are returning material and rating poorly, not on paths
  that have run out of new material to surface.

The `reason` should cite cumulative numbers (per-path counts across
the recent window, FIT-pile composition) and explicitly note that
no FIT row in the visible log is sole-sourced by the recommended
path. A single noisy turn is not sufficient grounds.

`confidence` calibration:
- `low`: 2 turns of consistent NOT_FIT, FIT pile sourced elsewhere,
  no Rule B1 risk visible — but pattern may be situational.
- `medium`: 2–3 turns of consistent NOT_FIT, FIT pile clearly
  sourced elsewhere, no Rule B1 risk visible across the window.
- `high`: pattern stable across the recent window AND the FIT pile
  is materially smaller than it would be with the path's
  contribution AND no Rule B1 risk visible — the drop is almost
  certainly correct.

If criteria are not met, leave the field null. Recommendations that
turn out to be wrong (operator ignores) are normal — the operator's
choice will not be fed back to you. Make the call from the rated
turn history alone.

### CONVERGENCE CHECK

If the latest turn's Precision@K ≥ `harvest.precision_at_k` AND the
session's cumulative unique FIT-rated PKs ≥ `harvest.min_fit` AND the
session's cumulative unique NOT_FIT-rated PKs ≥ `harvest.min_not_fit`
(all loaded from `config.yaml`; fused rows and per-path candidates both
count, deduped by PK), set:

  status: CONVERGED
  turns_to_converge: {n}

This is the same triple gate the orchestrator evaluates in
`src/replay/metrics.py::is_session_converged` and
`src/session/state.py::SessionState.is_converged`; emitting
`status: CONVERGED` while any of the three is still short will be ignored
by the orchestrator and your hypothesis on the next turn will be marked
REFUTED. (DISCARD ratings count toward neither floor.)
```
