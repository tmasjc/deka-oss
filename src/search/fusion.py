"""Pure-Python Reciprocal Rank Fusion for counterfactual previews.

Mirrors pymilvus.RRFRanker semantics so ``rrf_merge`` can simulate
what the fused top-K would look like given an arbitrary subset of
per-path rankings — without an extra Milvus round-trip. Used to
compute drop-impact previews shown to the human when the reflection
agent proposes deactivating a retrieval path.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TypeVar

K = TypeVar("K")


def rrf_merge(
    rankings: dict[str, list[K]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[K]:
    """Reciprocal rank fusion.

    ``rankings`` maps path name to a list of primary keys ordered by
    descending in-path score (rank 1 is first). Keys may repeat across
    paths; the fused score is the sum of ``1 / (rrf_k + rank)`` across
    every path that contains the key. Ties break by first-seen order
    within the input dict (deterministic in Python 3.7+).
    """
    scores: dict[K, float] = defaultdict(float)
    first_seen: dict[K, int] = {}
    counter = 0
    for path_rankings in rankings.values():
        for rank, key in enumerate(path_rankings, start=1):
            scores[key] += 1.0 / (rrf_k + rank)
            if key not in first_seen:
                first_seen[key] = counter
                counter += 1

    return sorted(
        scores.keys(),
        key=lambda k: (-scores[k], first_seen[k]),
    )[:top_k]
