"""Feature construction — embeddings ⨯ ``nearest_fit_distance``.

Phase 4 features each chunk as ``[*dense_embedding, nearest_fit_distance]``:

- ``dense_embedding`` is the BGE-M3 vector already living in Milvus
  on the session's collection. We fetch by PK to avoid re-embedding —
  same approach as Phase 2's :func:`src.anchor.loader._fetch_chunk_embeddings`.
- ``nearest_fit_distance`` comes from the Phase 2 cohort sidecar and
  is the only feature we standardise (mean=0, scale=1).

The dense components are left raw — BGE-M3 vectors are unit-normalised
so standardising them would just add noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from src.search.config import load_default_config
from src.search.evidence import PrimaryKey

from .errors import ApplyLoadError
from .load_session import CohortRow, TrainingLabel
from .schema import ScalerParams

log = logging.getLogger(__name__)


class _EmbeddingFetcher(Protocol):
    """Function-shaped Milvus query for unit tests to stub out."""

    def __call__(
        self,
        pks: list[PrimaryKey],
        *,
        collection: str,
    ) -> dict[PrimaryKey, list[float]]: ...


@dataclass(frozen=True)
class FeatureFrame:
    """A feature matrix with row identifiers preserved.

    ``X`` is (n_rows, embedding_dim + 1). Last column is the raw
    ``nearest_fit_distance`` — the trainer applies the
    :class:`ScalerParams` learned at fit time before model.fit().
    """

    pks: list[PrimaryKey]
    nearest_fit_distance: list[float]
    embeddings: list[list[float]]
    labels: list[int] | None  # 1=KEEP, 0=DROP; None for cohort apply
    deciles: list[int] | None  # populated for training rows only

    @property
    def n_rows(self) -> int:
        return len(self.pks)

    @property
    def embedding_dim(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0


def build_training_frame(
    labels: list[TrainingLabel],
    *,
    embeddings: dict[PrimaryKey, list[float]],
) -> FeatureFrame:
    """Join training labels against fetched embeddings.

    Drops any label whose PK has no embedding in ``embeddings`` (logs
    a warning) — Milvus could be missing a row that the cohort sidecar
    still references (e.g. a delete since Phase 2).
    """
    pks: list[PrimaryKey] = []
    distances: list[float] = []
    vectors: list[list[float]] = []
    ys: list[int] = []
    deciles: list[int] = []
    missing: list[PrimaryKey] = []
    for label in labels:
        vec = embeddings.get(label.pk)
        if vec is None:
            missing.append(label.pk)
            continue
        pks.append(label.pk)
        distances.append(label.nearest_fit_distance)
        vectors.append(vec)
        ys.append(1 if label.verdict == "KEEP" else 0)
        deciles.append(label.decile)
    if missing:
        log.warning(
            "build_training_frame: dropped %d label(s) with no Milvus "
            "embedding (first 5: %s)",
            len(missing),
            missing[:5],
        )
    return FeatureFrame(
        pks=pks,
        nearest_fit_distance=distances,
        embeddings=vectors,
        labels=ys,
        deciles=deciles,
    )


def build_cohort_frame(
    cohort: list[CohortRow],
    *,
    embeddings: dict[PrimaryKey, list[float]],
) -> FeatureFrame:
    """Same shape as :func:`build_training_frame` but unlabelled.

    Drops cohort rows missing an embedding and logs the count — a
    deletion between Phase 2 and Phase 4 should be small in practice
    and the writer records the cohort row count so the operator can
    sanity-check.
    """
    pks: list[PrimaryKey] = []
    distances: list[float] = []
    vectors: list[list[float]] = []
    missing: list[PrimaryKey] = []
    for row in cohort:
        vec = embeddings.get(row.pk)
        if vec is None:
            missing.append(row.pk)
            continue
        pks.append(row.pk)
        distances.append(row.nearest_fit_distance)
        vectors.append(vec)
    if missing:
        log.warning(
            "build_cohort_frame: dropped %d cohort row(s) with no Milvus "
            "embedding (first 5: %s)",
            len(missing),
            missing[:5],
        )
    return FeatureFrame(
        pks=pks,
        nearest_fit_distance=distances,
        embeddings=vectors,
        labels=None,
        deciles=None,
    )


def fit_scaler(distances: Sequence[float]) -> ScalerParams:
    """Return a 1-D StandardScaler-equivalent for ``nearest_fit_distance``.

    Hand-rolled instead of sklearn's ``StandardScaler`` because the
    persisted JSON shape is one field — keeping the math here means the
    classifier file can be rehydrated without instantiating a sklearn
    object first.
    """
    n = len(distances)
    if n == 0:
        raise ValueError("fit_scaler: empty distance vector")
    mean = sum(distances) / n
    var = sum((x - mean) ** 2 for x in distances) / n
    scale = max(var**0.5, 1e-12)
    return ScalerParams(mean=[mean], scale=[scale])


def apply_scaler(distances: Iterable[float], *, scaler: ScalerParams) -> list[float]:
    """Standardise a vector of distances using ``scaler``."""
    mean = scaler.mean[0]
    scale = scaler.scale[0]
    return [(d - mean) / scale for d in distances]


def stack_features(frame: FeatureFrame, *, scaler: ScalerParams) -> list[list[float]]:
    """Concatenate ``[embedding..., scaled_distance]`` per row.

    Yields a plain ``list[list[float]]`` so callers can feed it to numpy
    via ``np.asarray`` without importing numpy here. The shape matches
    the LR coefficient layout persisted in
    :class:`src.apply.schema.ClassifierMetadata`.
    """
    scaled = apply_scaler(frame.nearest_fit_distance, scaler=scaler)
    out: list[list[float]] = []
    for vec, dist in zip(frame.embeddings, scaled):
        row = list(vec)
        row.append(dist)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Milvus access (mirrors src.anchor.loader._fetch_chunk_embeddings)
# ---------------------------------------------------------------------------


def _render_pk_literal(pk: PrimaryKey) -> str:
    if isinstance(pk, int):
        return str(pk)
    return json.dumps(pk, ensure_ascii=False)


def fetch_embeddings(
    pks: list[PrimaryKey],
    *,
    collection: str,
    client: MilvusClient | None = None,
    pk_field: str = "id",
    batch_size: int = 2000,
) -> dict[PrimaryKey, list[float]]:
    """Query Milvus for dense embeddings of the given PKs.

    Batched so a 400K-row cohort doesn't try to render a single
    expression Milvus rejects. ``client`` is optional — if absent, a
    one-shot client is opened against ``search.milvus_uri`` and closed
    on exit; tests inject a mocked client to bypass real Milvus.

    Returns ``{pk: embedding}`` for every PK the query returned. Caller
    handles missing PKs.
    """
    if not pks:
        return {}
    owns_client = False
    if client is None:
        cfg = load_default_config()
        try:
            client = MilvusClient(uri=cfg.milvus_uri)
        except MilvusException as exc:
            raise ApplyLoadError(
                f"Failed to connect to Milvus at {cfg.milvus_uri}: {exc}"
            ) from exc
        owns_client = True

    out: dict[PrimaryKey, list[float]] = {}
    try:
        for start in range(0, len(pks), batch_size):
            chunk = pks[start : start + batch_size]
            rendered = ", ".join(_render_pk_literal(pk) for pk in chunk)
            expr = f"{pk_field} in [{rendered}]"
            try:
                rows = client.query(
                    collection_name=collection,
                    filter=expr,
                    output_fields=[pk_field, "dense_embedding"],
                )
            except MilvusException as exc:
                raise ApplyLoadError(
                    f"Milvus query for chunk embeddings failed at offset "
                    f"{start}/{len(pks)}: {exc}"
                ) from exc
            for entry in rows:
                pk = entry.get(pk_field)
                vec = entry.get("dense_embedding")
                if pk is None or vec is None:
                    continue
                out[pk] = [float(x) for x in vec]
    finally:
        if owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("MilvusClient.close() raised: %s", exc)
    return out
