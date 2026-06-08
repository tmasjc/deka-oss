# Introduction --- the locked-up corpus

Deka was built around a corpus of customer-conversation transcripts ---
the recorded interactions between a business's sales and service
personnel and its individual customers. Each representative--customer
relationship plays out across chat messages and phone calls, and the
business merges all of it into a single transcript per pair. Taken
together, these transcripts encode the operational knowledge of the
organisation: how a representative answers a customer who balks at the
price, how they steer a hesitant customer toward renewing, which doubts
about the product or service recur, how rapport gets built. The value is
real, and it is almost entirely inaccessible --- the corpus is far too
large to read, and it is in Chinese, a detail that will shape how it can
be searched.

Unlocking it comes down to a deceptively simple capability: ask a
specific question and reliably get back the chunks of conversation that
answer it. Hybrid retrieval over learned embeddings can do this. Each
chunk carries a *dense* vector that captures meaning and a
*learned-sparse* vector that captures lexical association, and a query
can be matched against both. But the two paths are not interchangeable.
A conceptual question --- *how does a representative build trust with a
customer?* --- is served best by the dense semantic path; a colloquial
one --- a customer grumbling that the product is "way too expensive" in
informal register
rather than the formal vocabulary of price sensitivity --- is served
best by the learned-sparse path, which bridges casual and formal
wording. No single blend of the two serves every query type well, so
each query must be tuned.

Tuning is where the cost lives, because deciding whether a retrieved
chunk genuinely answers a query is a judgement only a domain expert can
make. The product owner who drives Deka reads Chinese fluently and
understands the workflow --- the renewal decision, the recurring
customer objections --- well enough to tell a real
match from a superficially similar one. They are not, however, expected
to understand retrieval mathematics: cosine versus inner-product
metrics, reciprocal-rank fusion, index probe widths. That asymmetry sets
the system's central division of labour. The agent owns the mechanics
and explains its reasoning in plain narrative; the human owns the
judgement, and contributes nothing more than a verdict on each chunk ---
FIT, NOT\_FIT, or DISCARD.

The deliverable, for every query the operator cares about, is the set of
*all* chunks in the corpus relevant to it, with as little irrelevant
material mixed in as possible. "Relevant" is defined by the operator's
domain fluency; "all" is an aspiration, because the handful of chunks a
human can rate in a turn never covers a corpus this size. Meeting it
therefore cannot be the work of interactive tuning alone --- it is the
work of the phases that scale the tuned result.

This paper describes how Deka does that. Section 2 sets out the design
philosophy --- the harness --- that holds the system together.
Section 3 describes the data substrate and the overall architecture.
Section 4, the technical core, walks the four phases --- Probe, Harvest,
Refine, and Apply --- each presented as a deliberate transformation
along three axes: precision, cost, and legibility. Section 5 draws out
the recurring design principles and asks what generalises beyond this
particular corpus. Throughout, this is a paper about design and method:
it explains how the mechanisms work and why they are shaped as they are,
not how they score on a benchmark.

# Design philosophy: the harness

Deka is built on a simple equation:
$$\text{Agent} = \text{Model} + \text{Harness}$$
A large language model supplies reasoning, but reasoning alone is not a
system. Left to itself a model will drift, fabricate, lose track of what
it has already tried, and quietly answer a different question than the
one asked. The *harness* is everything around the model that makes its
reasoning reliable: the environment it acts in, the constraints it must
respect, the feedback it receives, and the state it would otherwise have
to hold in its head. Deka's contribution is not a new model or a new
embedding --- both are taken off the shelf --- but the harness that
turns a general-purpose model and a busy human expert into a dependable
retrieval-tuning instrument. Figure \ref{fig:loop} shows the Phase 1
loop in full; three of its elements carry the design.

**Feedforward guides** constrain the action before it happens. Each turn
the agent is handed the query, the locked search configuration, and a
set of hard constraints --- never fabricate a chunk, always cite chunk
IDs, always log the turn. These guides narrow the space of things the
agent *can* do to the things it *should* do, so that a single careless
step cannot corrupt the session.

**Feedback sensors** report what happened after the action. Two kinds of
signal flow back: the human's verdict on each presented chunk (FIT,
NOT\_FIT, or DISCARD) and metrics the harness computes for itself ---
precision at $K$, per-path contribution. The human signal is the ground
truth the system is aligning to; the computed signal is how the agent
reasons about *why* a turn went the way it did.

**Externalised state** keeps the session's memory out of the model.
Every turn is appended to a progress log on disk --- the query, the
configuration, the ratings, the agent's reflection. The model never has
to remember the session; it reads the log. This is what makes a session
resumable, auditable, and reproducible, and it is what lets the agent
reason over the whole history rather than just the last exchange.

Two disciplines hold the loop steady. The first is a **locked
configuration**: the four parameters that define a search --- the
rank-fusion constant, the per-path candidate limit, the per-turn
presentation size, and the set of active retrieval paths --- are fixed
once the first turn begins. The agent's reflection does not propose new
parameter values; it produces a reasoning trace --- *observe, diagnose,
hypothesise* --- about what the evidence shows. Changing a knob
mid-session would confound the very signal the operator is reading, so
the only permitted mid-session change is a tightly gated path drop,
taken after an explicit audit (Section 4.1). The second discipline is
**narrative over metrics**: the progress log is written as
human-readable prose, not a dump of numbers. Because no one can read the
corpus directly, the accumulated reflections become a map of what the
corpus looks like from the angle of each query --- intuition a metric
table would discard.

\begin{figure}[t]
\centering
\input{figures/harness-loop.tex}
\caption{The Phase 1 harness loop. Feedforward guides constrain the agent's hybrid-search action; human ratings and computed metrics feed a structured reflection; the externalized progress log persists each turn until the convergence gate fires, at which point the FIT-anchored harvest is offered.}
\label{fig:loop}
\end{figure}

# System architecture

## The data substrate

Deka retrieves over a Milvus vector collection in which each row is one
chunk of a customer-conversation transcript. Two learned representations sit on
every chunk: a 1024-dimensional *dense* embedding indexed under a cosine
metric, and a *learned-sparse* embedding indexed under inner product,
both produced by the same multilingual embedding model. The dense vector
answers semantic similarity; the learned-sparse vector captures lexical
associations that bridge casual and formal vocabulary --- the reason a
customer's offhand "way too expensive" can still match a query phrased in
the register of price sensitivity.

A query is issued against both fields at once, and the two ranked lists
are combined by **reciprocal-rank fusion**. A candidate's fused score is
the sum over active paths of $1 / (k + r)$, where $r$ is its rank within
that path's list and $k$ is a smoothing constant (`rrf_k`). Fusion uses
rank *position*, not native score: the dense path's cosine scores and
the sparse path's inner-product scores live on incomparable scales, and
rank fusion makes the combination scale-free. Weighting the paths is
therefore not a tunable axis --- the only levers are how many candidates
each path contributes and how flat the rank-decay curve is.

The schema once carried a third path --- a built-in BM25 index over the
raw text --- but it was retired from Phase 1. The default tokenizer does
not segment Chinese: short keyword queries tokenise and score, but the
long, phrase-shaped queries this workbench is built for collapse into
unmatched tokens and return nothing. In practice BM25 contributed almost
no FIT-rated chunks while handing the reflection agent a misleading third
knob to twist, so it was dropped from the loop. The field remains on the
schema, so a future Chinese-aware tokenizer could restore the path
without migrating any data.

A companion PostgreSQL store holds the pre-chunking *original text* of
each conversation, keyed by the same primary key. Retrieval itself runs
entirely on Milvus; Postgres is consulted only when a human --- or, later,
an LLM judge --- needs to read a chunk in its full surrounding context.
Neither the collection nor the table is hard-coded: a small **scope**
registry maps a human-facing name to a `{collection, table}` pair, so one
deployment can serve several corpora and the operator simply picks a
scope when a session begins.

## The four locked axes

Phase 1 exposes exactly four search parameters, and all four are fixed
the moment the first turn begins (Section 2). They are scaffolding for
reaching the goal efficiently, not the goal itself.

| Parameter | Role |
|-----------|------|
| `rrf_k` | Rank-fusion smoothing constant; a larger value flattens the rank-decay curve. |
| `per_path_limit` | How many candidates each path contributes before fusion. |
| `top_k` | How many fused chunks the operator sees per turn --- a presentation budget, **not** a cap on corpus coverage. |
| `active_paths` | Which retrieval paths feed fusion: dense, learned-sparse, or both. |

The `top_k` distinction matters: it bounds the human's per-turn reading
load, but the corpus-wide coverage promised by the deliverable is met
later, by the harvest, not by enlarging what any single turn shows.

## Phases and the workbench

The same substrate is queried by four phases that run in sequence, each
optional and each gated on the operator's confirmation. Phase 1 (Probe)
tunes retrieval interactively; Phase 2 (Harvest) sweeps the corpus around
the validated examples; Phase 3 (Refine) turns that sweep into a readable
rubric; Phase 4 (Apply) scales the rubric across the whole cohort with a
cheap classifier. A session advances through an explicit state machine
(Figure \ref{fig:states}): each phase can be declined by the operator or
disabled by configuration, in which case the session ends early. Only
Phase 1 loops; Phases 2--4 are single forward passes, each with its own
review gate.

The operator works entirely through a web workbench. It presents the
query entry, the per-turn evidence as scrollable chunk cards with FIT /
NOT\_FIT / DISCARD controls, a convergence panel that tracks the session
against its stopping criteria, and a reflection view that surfaces the
agent's observe--diagnose--hypothesise trace. Everything the operator
does is a judgement about content; everything mechanical happens behind
the panel.

\begin{figure}[t]
\centering
\input{figures/state-machine.tex}
\caption{The session phase state machine. Phases~2--4 run sequentially after Phase~1 converges, each gated on operator confirmation and a configuration flag; a declined or disabled phase routes directly to \texttt{DONE}. Failure states (\texttt{ANCHOR\_FAILED}, \texttt{REFINE\_FAILED}, \texttt{APPLY\_FAILED}) are omitted for legibility.}
\label{fig:states}
\end{figure}

# The four-phase method

\begin{figure}[t]
\centering
\input{figures/phase-transform.tex}
\caption{The four phases as a chain of transformations. Each phase moves the operator's question one step further along the axes of precision, cost, and legibility --- from a fuzzy intuition to a labelled cohort applied across the full corpus.}
\label{fig:transform}
\end{figure}

The four phases share a substrate but answer four different questions,
and each hands its successor a more refined object than it received
(Figure \ref{fig:transform}). Read as a sequence they trace a single
arc: a fuzzy intuition becomes a set of rated examples; the examples
become a geometric cohort; the cohort becomes a readable rule; and the
rule becomes a label on every relevant chunk in the corpus. Each step
buys something --- precision, coverage, legibility, or scale --- and the
cost it pays for that gain is what motivates the step that follows.

## Probe --- tuning the retrieval lens

Phase 1 is the only interactive phase and the only one that loops. A
turn is a single circuit of the harness loop in Figure \ref{fig:loop}:
the agent runs the hybrid search under the locked configuration, the
workbench presents the `top_k` fused chunks, and the operator rates
each one. FIT means the chunk genuinely answers the query; NOT\_FIT
means it does not; DISCARD removes a chunk that is defective at the
source --- garbled speech-to-text, broken grammar, off-topic noise ---
so that it never propagates into a later phase. The agent then reflects
in three movements --- *observe* what the evidence shows, *diagnose*
why, *hypothesise* what it implies --- and appends the turn to the
progress log. Rated chunks are filtered out of subsequent searches, so
every turn shows fresh material.

The loop ends at a **triple convergence gate**. A session has converged
when three conditions hold at once: the latest turn's precision among
the presented chunks clears a configured bar; the session has
accumulated at least a configured number of distinct FIT chunks; and it
has accumulated at least a configured number of distinct NOT\_FIT
chunks. The precision bar certifies that the current configuration is
actually surfacing relevant material. The FIT floor guarantees enough
validated examples to define the concept. The NOT\_FIT floor is the
subtle one: it exists because the rubric phase downstream needs
*contrastive negatives* --- examples of what the concept is *not* --- and
a session that converged with zero NOT\_FITs would starve that phase of
the evidence it uses to draw a boundary. The operator may also stop
manually once satisfied.

One controlled exception to the locked configuration lives here. If the
operator suspects a retrieval path is contributing only noise, they can
trigger an *audit*: the workbench surfaces the per-path candidates that
fusion normally hides, the operator rates them, and only if that
evidence clears a specific gate is the path dropped for the remainder of
the session. Nothing else about the configuration can change once tuning
has begun.

This loop is also where exploration happens. A high-quality query is
rarely handed to the system at the outset; it is one the operator
arrives at --- by probing the corpus with different formulations of a
question, reading the reflections each probe returns, and letting that
accumulating picture sharpen both their grasp of the corpus and the
query itself. The downstream phases run only once a query is judged
worth committing to; Deka is deliberately not a single forward pass from
question to labelled cohort. With the search configuration locked, this
is also what the per-turn reflection is chiefly *for*: less a control
signal than the instrument through which the operator comes to
understand the corpus.

What Phase 1 produces is not the tuned configuration --- that is a
by-product --- but a set of human-validated FIT and NOT\_FIT examples:
the first transformation in Figure \ref{fig:transform}, a fuzzy operator
intuition made concrete. Its cost is human attention, paid one turn at a
time, which is precisely why it cannot be the mechanism that covers the
whole corpus.

## FIT-anchored harvest --- from examples to a cohort

The handful of FITs a session accumulates are a small fraction of the
chunks in the corpus that express the same concept. Phase 2 closes that
gap by **reframing the FITs as a query rather than a result** --- used as
concept exemplars, not training labels: given these $n$ human-validated
examples, what other chunks lie in their immediate embedding
neighbourhood? This is the step that delivers on the "all relevant
chunks" promise the per-turn budget could never keep.

Two dense vectors travel with every FIT. The *span* vector $s_i$ embeds
the concept-bearing segment the operator marked during Phase 1 --- the
tightest read on what the concept is, and the search vector Phase 2
launches from. The *chunk* vector $c_i$ is the embedding under which the
chunk is indexed in Milvus --- the FIT's footprint in the searchable
space. The two occupy slightly different regions of the embedding space,
and that gap shapes the calibration below.

### Calibration: a radius derived from the FITs

The metric throughout is cosine distance,
$d(u, v) = 1 - \tfrac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert}$,
matching the index on the dense field. Two quantities set each anchor's
reach. The first is a **session-wide base radius** $T$, derived purely
from the geometry of the FIT spans among themselves: take each span's
leave-one-out nearest-neighbour distance
$\ell_i = \min_{j \neq i} d(s_i, s_j)$ and set
$T = \mathrm{p90}(\{\ell_i\})$ --- the 90th percentile, which absorbs the
bulk of the set's internal spread without letting a single outlying span
inflate the radius. The second is a **per-anchor slack**
$\delta_i = d(s_i, c_i)$ that bridges the span manifold to the chunk
manifold: because the search launches from a span vector while the corpus
is indexed by chunk vectors, even a FIT's own chunk sits at distance
$\delta_i$ from its span. The anchor's effective radius is the sum
$$T'_i \;=\; T \;+\; \delta_i,$$
so one concept-level spread ($T$) serves anchors whose spans are tight
quotations (large $\delta_i$) and near-whole-chunk excerpts (small
$\delta_i$) alike. An **anchor-quality gate** discards any FIT whose
$\delta_i$ is implausibly large --- a degenerate anchor whose radius
would admit chunks well beyond the cohort's spread.

A radius derived from the FITs can also fail quietly: if the examples are
individually fine but collectively incoherent, their neighbourhoods may
not overlap and the cohort becomes a union of disjoint islands. A
**leave-one-out recovery check** guards against this --- each FIT is
hidden in turn and the rest re-calibrated, asking whether the held-out
chunk still surfaces inside some remaining anchor's neighbourhood. A
failing verdict aborts the run, on the grounds that no calibration this
incoherent should be trusted to define a neighbourhood.

### The main pass: a monotonic widening search

The harvest opens one dense $k$-NN search per surviving anchor against
the corpus, using $s_i$ as the query vector. Because Milvus returns hits
in ascending cosine distance, the radius is itself the stopping rule ---
once a page's last hit exceeds $T'_i$, the iterator closes; no
probe-width tuning, no candidate-list cap. (A safety cap $M$ bounds total
hits per anchor against pathological breadth, logged loudly rather than
silently truncated.) The $n$ neighbourhoods are then unioned into one
candidate set, and for each candidate $p$ the system records its
nearest-FIT distance --- a downstream proximity score --- and the full
set of anchors whose ball $B(s_i, T'_i) = \{q : d(s_i, q) \le T'_i\}$
admitted it, the quantity that drives the final filter.

### The anchor-frequency gate

A candidate is kept only if at least $f$ distinct FIT anchors admitted
it (default $f = 2$). The intuition is geometric: the union
$\bigcup_i B(s_i, T'_i)$ is a generous region that absorbs any single
anchor's noise --- chunks near a slightly mis-placed anchor that do not
express the concept --- while the intersection of any two balls is far
tighter, because single-anchor noise rarely lands inside two
independently rated neighbourhoods at once. Concept-bearing chunks,
close to *every* anchor's span, are largely unaffected. In practice,
requiring two-anchor agreement shrinks the candidate cohort several-fold
on dense queries --- often by close to an order of magnitude, though the
factor varies with how diffuse the concept is --- while preserving the
high-confidence core.
Each retained chunk records both its nearest anchor and the full set of
admitting anchors, so the gate can be tightened or relaxed offline
without re-running the search.

This is the second transformation in Figure \ref{fig:transform}: a set
of rated examples becomes a geometric cohort spanning the corpus. But
the cohort is defined entirely by *distance*, and that is its
limitation. Proximity in the embedding space overlaps with --- yet is
not identical to --- "expresses the same concept", and closing that gap
is the reason the next phase exists.

## Rubric refinement --- from geometry to language

A cohort defined by distance is not the same as a cohort defined by
*meaning*, and the gap between them is exactly what the operator cares
about. Phase 2 can establish that a chunk sits close to the validated
examples; it cannot say, in words a person can read and argue with, *why*
a chunk belongs. Phase 3 turns the geometric boundary into an explicit,
language-level one.

Its deliverable is a **rubric prompt**: a self-contained document that any
language model with a JSON mode can apply to any chunk. The prompt names
the checks that define membership in the concept and fixes the structure
of the verdict; it carries FIT exemplars drawn from the session and
NOT\_FIT counter-examples, each annotated with the specific check it
fails. Given the prompt and a chunk, a model returns a structured
judgement. The rubric is therefore portable --- it can be read, edited,
version-controlled, and shipped downstream entirely independently of
Deka.

What the harness contributes is not any single rubric but the
**meta-prompt** that writes one. A strong model runs the meta-prompt
against the converged session and emits a rubric of the agreed shape
every time; the operator may edit any part before locking it. Making
rubric derivation *repeatable* --- rather than a bespoke prompt-engineering
exercise per query --- is the real engineering here, and every derived
artifact is stamped with the SHA-256 of the meta-prompt that produced it,
so any rubric can be traced to its exact generator.

The phase runs in four steps. **Derive** produces the rubric as above.
**Sample** draws a manageable evaluation set from the Phase 2 cohort ---
around five hundred chunks by default --- stratified by distance to the
nearest FIT so that the full range from core to periphery is
represented, not just the easy centre; chunks already rated in Phase 1
are excluded, and any already known to be NOT\_FIT are dropped without
spending a model call. **Judge** applies the locked rubric to each
sampled chunk through a rate-limited asynchronous fan-out, validating
every response against the rubric's declared schema. **Review** puts the
operator in front of the verdicts: they either *finalise* --- writing the
rubric and its judged sample to disk and ending the phase --- or
*discard*, which bumps the rubric version and returns to the editor, so
that each round of judging is always against a single, locked predicate.

This is the third transformation in Figure \ref{fig:transform}: a
geometric cohort becomes a readable, auditable language boundary. The
cost it pays is a few hundred model calls on a sample --- and that
sample, it turns out, is a labelled training set the system would
otherwise throw away.

## Cohort apply --- from sample to scale

The rubric was always meant to travel across the entire Phase 2 cohort,
which can run to hundreds of thousands of chunks. Applying the
language-model judge to every one of them, one call per chunk, would cost
many times the rest of the session combined. But Phase 3 has just
produced something valuable almost as a side effect: several hundred
chunks, each labelled KEEP or DROP by the locked rubric. That is a
training set.

Phase 4 fits a deliberately cheap model --- a logistic-regression
classifier --- on those labels, using each chunk's dense embedding
together with its distance to the nearest FIT anchor as features, with
balanced class weights so the smaller class is not swamped. The rubric
remains the boundary's *definer*; the classifier is only its *applier*,
a fast approximation that can sweep the whole cohort for the cost of a
dot product per chunk.

Because the entire point is to avoid polluting whatever consumes the
cohort downstream, the classifier is tuned for **precision over
recall**: a configured precision floor --- 0.75 by default --- must be met
on the KEEP class, and shipping below it requires an explicit operator
override. The reasoning is asymmetric. A relevant chunk the classifier
wrongly drops can be recovered by a later, looser pass; an irrelevant
chunk it wrongly keeps quietly corrupts everything built on top.
Precision is estimated honestly, by repeated stratified $k$-fold
cross-validation over *all* the labelled rows --- five folds repeated five
times by default --- rather than a single held-out split, so the reported
figure does not hinge on one lucky partition.

The operator's only interaction in this phase is with the **threshold**,
not with individual chunks. A calibration screen shows the
precision--recall curve, the precision and recall at the current
threshold, how the cohort would split into KEEP and DROP, and a carousel
of borderline chunks for a sanity check; the operator drags a slider to
choose the operating point. On confirmation, every chunk in the cohort is
scored once through the persisted classifier and written out with its
verdict and keep-probability. The classifier file records the rubric
version and the meta-prompt hash it was trained under, so that
re-applying it later to fresh data refuses to run if the rubric has since
changed.

This is the final transformation in Figure \ref{fig:transform}:
expensive, per-chunk language-model judging becomes near-zero-cost
classification at full-cohort scale. The arc that began with a fuzzy
intuition ends with a defensible label on every relevant chunk in the
corpus.

# Design principles and discussion

Several principles recur across the four phases, and together they
characterise the system more than any single mechanism does.

**Keep the human on judgement, never on mechanics.** At every phase the
operator's input is a verdict about *content* --- is this chunk relevant,
does this rubric read correctly, is this threshold acceptable --- and
never a retrieval parameter or a piece of mathematics. The machinery
computes; the human decides. This is what lets a product owner with
domain fluency but no retrieval background drive the system to a
trustworthy result.

**Prefer precision, and make the cost of error explicit.** From the
harvest's multi-anchor agreement gate to the classifier's precision
floor, the system is consistently biased against false positives,
because a wrongly-included chunk pollutes everything downstream while a
wrongly-excluded one can be recovered later. Recall is treated as the
cheaper mistake, deliberately.

**Externalise everything, and make it reproducible.** State lives on disk
as narrative logs and versioned artifacts, not in a model's context.
Rubrics are stamped with the hash of the meta-prompt that produced them;
classifiers refuse to run against data whose rubric has drifted.

**What generalises.** Nothing in the four-phase arc is specific to sales
transcripts, or to Chinese. The pattern --- tune an interactive loop until
a human's intuition is captured as validated examples, expand those
examples geometrically, distil the expansion into a readable rule, then
cheapen the rule into a classifier for scale --- applies wherever an
expert holds a fuzzy concept that must be found, agreed upon, and applied
across a corpus too large to read. The substrate would change; the
harness would not.

It is worth being equally clear about what Deka is *not*. It is not a
retrieval-augmented generation system: it never has a model answer
questions over the chunks, only locate and label them. It is not an
automatic optimiser: the agent proposes and explains, but a human
confirms every consequential step. It tunes one query at a time and does
not rank queries against each other, and it takes the embedding model as
fixed rather than evaluating alternatives. These are deliberate scope
choices, not missing features --- each one keeps the human's judgement,
rather than a metric, at the centre of the loop.

# Conclusion

Deka treats a hard problem --- extracting an expert's tacit notion of
relevance from a corpus no one can read --- as a matter of harness design
rather than model capability. A general-purpose model supplies the
reasoning and a domain expert supplies the judgement; the harness
supplies everything that makes the pair reliable, and arranges their work
as a deliberate escalation. Each phase pays down a different cost: Probe
pays for judgement with human attention, Harvest pays for coverage with
geometry, Refine pays for legibility with a sample of model calls, and
Apply pays for scale with a cheap classifier trained on that sample. What
walks out of a session is not merely a set of chunks but a reproducible,
auditable account of a concept --- validated examples, a readable rubric,
and a classifier that carries it across the whole corpus. The lesson
generalises past this corpus: when the model is unreliable and the human
is expensive, the leverage is in the harness between them.

\appendix

# End-to-end workflow

The three figures in the body each capture one slice of the system ---
the Phase 1 harness loop (Figure \ref{fig:loop}), the inter-phase state
machine (Figure \ref{fig:states}), and the precision/cost/legibility
arc the phases trace (Figure \ref{fig:transform}). Figure
\ref{fig:workflow} synthesises them into a single dataflow picture.

\begin{figure}[h]
\centering
\input{figures/workflow.tex}
\caption{End-to-end workflow. Three swim lanes show, left to right, the
\emph{substrate} (the on-disk artifacts each phase writes), the
\emph{agent} (the four phases), and the \emph{operator} (the human's
per-phase touchpoint). Solid thin arrows are writes (agent
$\to$ substrate); bold blue arrows are phase advances down the agent
column; double-headed arrows between agent and operator denote
\emph{present + verdict} --- the agent surfaces evidence, the operator
returns a judgement (a rating, a confirmation, a slider position). The
small curved arrow above Phase~1 marks the only looping phase;
Phases~2--4 are single forward passes.}
\label{fig:workflow}
\end{figure}

Three points are worth reading off the figure directly. First, the
operator's column is contentful at every phase but never carries
mechanics --- only judgements about content, thresholds, or
finalisation. Second, the substrate column makes every inter-phase
handoff an explicit, persisted artifact: a session resumed from disk
re-enters the workflow at the right row without re-running prior
phases. Third, the absence of any arrow from a later phase back to an
earlier one is itself a design statement --- the workflow is a strictly
forward escalation, and rework happens by re-running a phase, not by
splicing back into one already finished.

# A worked session

To make the escalation concrete, here is one converged session carried
through all four phases against the configured collection. It is an
*illustration*, not a benchmark --- one run cannot characterise the
method's accuracy, but it puts the arc and its costs in real numbers.

The session **converged in three Phase 1 turns**, over which the operator
rated twenty-three chunks --- twenty FIT, three NOT\_FIT --- at a per-turn
precision near 0.87. Those twenty FITs are the whole human budget. Phase 2
reframed them as the query: the quality gate kept fifteen anchors,
calibration set a base radius $T = 0.18$, and the per-anchor searches
returned a union of 154,109 candidates that the two-anchor gate ($f = 2$)
cut to a **38,099-chunk cohort** --- a roughly fourfold reduction. The
leave-one-out recovery check came back FLAGGED, not HEALTHY (ten of
fifteen held-out FITs re-surfaced), the signature of a somewhat diffuse
concept; the cohort was judged usable.

Phase 3 judged a 500-chunk stratified sample, splitting **288 KEEP /
211 DROP**, with the KEEP rate falling as distance to the nearest FIT
grew --- geometry and meaning agreeing at the core and parting at the
edge. Phase 4 fit a logistic classifier on that sample and, under
five-by-five cross-validation, reached **precision 0.75 and recall 0.91**
at the operator's threshold ($0.750 \pm 0.003$ across folds). Applied to
the cohort, it labelled **27,148 chunks KEEP**.

Read end to end, the run is the paper's claim in miniature: *three turns
and twenty examples become some twenty-seven thousand labelled chunks* ---
human judgement spent only at the front, every later step paying it down
with geometry, a sample of model calls, and a dot product per chunk.
