"""Greedy farthest-first selection for diverse exemplar picking.

Used by :mod:`src.refine.derive` to cap how many FIT / NOT_FIT
exemplars are passed to the rubric meta-prompt. When a converged
session carries more ratings than the cap (``max_fit_examples`` /
``max_not_fit_examples`` in ``config.yaml``), this picks the subset
that maximises semantic coverage over the underlying span / chunk
embeddings.

The algorithm — also called *farthest-point sampling* — starts with a
deterministic seed (``items[0]``) and iteratively picks the item
whose minimum cosine distance to the already-selected set is largest.
That keeps the selected subset spread across the embedding manifold
without needing a clustering step.

Pure-Python + numpy; no I/O, no config import. Standalone so the
selection rule is unit-testable with synthetic vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def select_diverse(
    items: Sequence[T],
    embeddings: Sequence[Sequence[float]],
    k: int,
) -> list[T]:
    """Return up to ``k`` items chosen for maximum semantic diversity.

    Items are paired with their embeddings positionally — ``items[i]``
    corresponds to ``embeddings[i]``.

    Returns ``list(items)`` unchanged when ``len(items) <= k``; selection
    only matters when the list overflows the cap.

    Implementation: greedy farthest-first over cosine distance.

    - Seed with ``items[0]`` so the result is deterministic given the
      same input order. Callers that need a different seed can rotate
      the input lists before calling.
    - At each step pick the unselected item whose minimum cosine
      distance to the already-selected set is largest. Ties are broken
      by the first-encountered index (stable).

    Raises ``ValueError`` for ``k <= 0``, mismatched lengths, or
    embeddings that are not 2-D.
    """

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(items) != len(embeddings):
        raise ValueError(
            f"items / embeddings length mismatch: "
            f"{len(items)} vs {len(embeddings)}"
        )
    if not items:
        return []
    if len(items) <= k:
        return list(items)

    mat = np.asarray(embeddings, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D matrix, got shape {mat.shape}"
        )

    # L2-normalise once; cosine distance reduces to 1 - x·y on unit
    # vectors. Guard against zero-norm rows (degenerate embedding —
    # treat as max-distance from everything so it gets picked first
    # after the seed).
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = mat / safe

    selected_idx: list[int] = [0]
    # Cosine distance from every item to the current selected set;
    # initialised against the seed.
    min_dist = 1.0 - unit @ unit[0]
    min_dist[0] = -np.inf  # mark selected so argmax never picks it again

    while len(selected_idx) < k:
        nxt = int(np.argmax(min_dist))
        selected_idx.append(nxt)
        # Update running min-distance with distances to the new pick.
        new_dist = 1.0 - unit @ unit[nxt]
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[nxt] = -np.inf

    return [items[i] for i in selected_idx]
