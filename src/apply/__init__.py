"""Phase 4 — cohort apply via cheap classifier trained on Phase 3 sample.

Trains a logistic regression on the Phase 3 judged sample (KEEP/DROP)
and applies it to the full Phase 2 cohort at zero marginal LLM cost.
See ``proposals/phase4_cohort_apply.md`` for the design.
"""
