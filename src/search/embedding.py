"""BGE-M3 embedding client.

Thin wrapper around the ``/embed-all`` endpoint documented in
``docs/INFRA.md``. Ported from ``smoke_tests/hybrid_search.py`` with
two changes: the endpoint is injected as an argument instead of a
module-level constant, and failures are wrapped in
:class:`EmbeddingServiceError`.
"""

from __future__ import annotations

import logging

import requests

from .errors import EmbeddingServiceError

log = logging.getLogger(__name__)


def get_embeddings(
    sentences: list[str],
    embed_url: str,
    *,
    timeout: int,
) -> dict:
    """Call ``/embed-all`` to get dense + sparse vectors for ``sentences``.

    Raises :class:`EmbeddingServiceError` on any HTTP or network failure.
    """

    url = f"{embed_url}/embed-all"
    log.debug("Requesting embeddings for %d sentence(s) from %s", len(sentences), url)
    try:
        resp = requests.post(url, json={"sentences": sentences}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise EmbeddingServiceError(
            f"BGE-M3 embedding request to {url} failed: {exc}"
        ) from exc

    try:
        return resp.json()
    except ValueError as exc:
        raise EmbeddingServiceError(
            f"BGE-M3 response from {url} was not valid JSON"
        ) from exc


def sparse_to_milvus(sparse: dict) -> dict[int, float]:
    """Convert a BGE-M3 sparse dict (string keys) to Milvus form (int keys).

    Milvus rejects string keys with a validation error, so this
    conversion is non-negotiable. See ``docs/INFRA.md`` for the rule.
    """

    return {int(k): float(v) for k, v in sparse.items()}
