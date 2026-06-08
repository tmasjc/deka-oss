"""Apply configuration — Phase 4 classifier + threshold knobs.

The ``apply:`` YAML section owns the default confidence threshold,
the precision acceptance bar, the train/eval split, the in-training
CV fold count, and the embedding-fetch batch size for the cohort
prediction pass. Loader mirrors :mod:`src.refine.config` — manual
field-by-field validation, all errors funnel through
:class:`src.apply.errors.ApplyConfigError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config_loader import ConfigFileError, load_section

from .errors import ApplyConfigError

_SECTION = "apply"

_REQUIRED_KEYS = frozenset(
    {
        "enabled",
        "confidence_threshold",
        "min_precision",
        "eval_fraction",
        "kfold_splits",
        "seed",
        "embedding_dim",
        "embedding_fetch_batch",
    }
)

# Keys the loader accepts but does not require — back-compat for
# pre-migration configs that haven't been updated yet. Each optional
# key has a default applied at load time.
_OPTIONAL_KEYS = frozenset({"eval_n_repeats"})

_DEFAULT_EVAL_N_REPEATS = 5


@dataclass(frozen=True)
class ApplyConfig:
    """Validated Phase 4 configuration.

    ``eval_fraction`` is deprecated since the eval methodology moved
    to repeated stratified k-fold over the full labelled set — the
    field is still required in YAML for back-compat with existing
    configs but is no longer read by the runner. A one-time warning
    is logged on load.
    """

    enabled: bool
    confidence_threshold: float
    min_precision: float
    eval_fraction: float
    eval_n_repeats: int
    kfold_splits: int
    seed: int
    embedding_dim: int
    embedding_fetch_batch: int


def load_apply_config(
    path: Path | None = None,
    session_overrides: dict[str, Any] | None = None,
) -> ApplyConfig:
    """Load and validate the ``apply:`` section.

    ``session_overrides`` (when present) shadows YAML values for this
    load only; see :mod:`src.web_api.schemas.SessionOverrides`.
    """
    try:
        raw = load_section(_SECTION, explicit=path, overrides=session_overrides)
    except ConfigFileError as exc:
        raise ApplyConfigError(str(exc)) from exc

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise ApplyConfigError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}"
        )
    allowed = _REQUIRED_KEYS | _OPTIONAL_KEYS
    unknown = raw.keys() - allowed
    if unknown:
        raise ApplyConfigError(
            f"config section '{_SECTION}' contains unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )

    enabled = _require_bool(raw, "enabled")
    confidence_threshold = _require_unit_float(raw, "confidence_threshold")
    min_precision = _require_unit_float(raw, "min_precision")
    eval_fraction = _require_unit_float(raw, "eval_fraction")
    if eval_fraction == 0.0 or eval_fraction >= 1.0:
        raise ApplyConfigError(
            f"config section '{_SECTION}': 'eval_fraction' must be in (0, 1); "
            f"got {eval_fraction}."
        )
    kfold_splits = _require_int_ge(raw, "kfold_splits", minimum=2)
    eval_n_repeats = _optional_int_ge(
        raw, "eval_n_repeats", minimum=1, default=_DEFAULT_EVAL_N_REPEATS
    )
    seed = _require_int(raw, "seed")
    embedding_dim = _require_int_ge(raw, "embedding_dim", minimum=1)
    embedding_fetch_batch = _require_int_ge(raw, "embedding_fetch_batch", minimum=1)

    import logging

    log = logging.getLogger(__name__)
    log.debug(
        "apply.eval_fraction=%.2f is deprecated and unused — Phase 4 evaluates "
        "via repeated stratified k-fold (kfold_splits=%d, eval_n_repeats=%d). "
        "The field is still required for back-compat and will be removed in a "
        "future release.",
        eval_fraction,
        kfold_splits,
        eval_n_repeats,
    )
    if "eval_n_repeats" not in raw:
        log.info(
            "apply.eval_n_repeats not set in config; defaulting to %d. "
            "Add `eval_n_repeats: %d` under `apply:` in config.yaml to "
            "silence this message.",
            eval_n_repeats,
            _DEFAULT_EVAL_N_REPEATS,
        )

    return ApplyConfig(
        enabled=enabled,
        confidence_threshold=confidence_threshold,
        min_precision=min_precision,
        eval_fraction=eval_fraction,
        eval_n_repeats=eval_n_repeats,
        kfold_splits=kfold_splits,
        seed=seed,
        embedding_dim=embedding_dim,
        embedding_fetch_batch=embedding_fetch_batch,
    )


def _require_bool(raw: dict, key: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise ApplyConfigError(
            f"config section '{_SECTION}': '{key}' must be a boolean"
        )
    return value


def _require_int(raw: dict, key: str) -> int:
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ApplyConfigError(
            f"config section '{_SECTION}': '{key}' must be an integer"
        )
    return value


def _require_int_ge(raw: dict, key: str, *, minimum: int) -> int:
    value = _require_int(raw, key)
    if value < minimum:
        raise ApplyConfigError(
            f"config section '{_SECTION}': '{key}' must be >= {minimum}"
        )
    return value


def _optional_int_ge(raw: dict, key: str, *, minimum: int, default: int) -> int:
    if key not in raw:
        return default
    return _require_int_ge(raw, key, minimum=minimum)


def _require_unit_float(raw: dict, key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApplyConfigError(
            f"config section '{_SECTION}': '{key}' must be a number in [0, 1]"
        )
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 1.0:
        raise ApplyConfigError(
            f"config section '{_SECTION}': '{key}' must be in [0, 1]; got {fvalue}"
        )
    return fvalue
