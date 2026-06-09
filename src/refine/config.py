"""Refine configuration — Phase 3 derive + judge knobs.

The ``refine:`` YAML section owns the meta-prompt path, the two model
endpoints (derive, judge), the stratified-sample budget, and the
async judge's concurrency / QPS / TPM / timeout / retry knobs. Every
key is required; absence of the section is a hard parse error so
operators don't quietly run a Phase 3 pass with implicit defaults.

The loader mirrors :mod:`src.anchor.config` — manual field-by-field
validation, no extra dependencies, all errors funnel through
:class:`src.refine.errors.RefineConfigError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config_loader import ConfigFileError, load_section

from .errors import RefineConfigError

_SECTION = "refine"

_REQUIRED_KEYS = frozenset(
    {
        "enabled",
        "sample_size",
        "n_bins",
        "seed",
        "meta_prompt_path",
        "max_fit_examples",
        "max_not_fit_examples",
        "derive_model",
        "derive_base_url",
        "derive_temperature",
        "judge_model",
        "judge_base_url",
        "judge_concurrency",
        "judge_qps_limit",
        "judge_tpm_limit",
        "judge_timeout_seconds",
        "judge_max_retries",
        "api_key_env",
        "auto_drop_known_intruders",
    }
)

# Optional keys — present in some deploys but not required for back-compat.
# ``judge_api_key_env`` decouples the judge LLM's bearer from the derive
# LLM's: useful when the two endpoints belong to different providers
# (e.g. derive on DashScope, judge on a self-hosted vLLM that needs no
# real key) so the real DashScope key isn't sent to the vLLM host.
# ``derive_enable_thinking`` forwards ``extra_body={"enable_thinking": True}``
# on derive calls — DashScope thinking-capable models (deepseek-v4-pro,
# qwen3+) double their reasoning budget when this is set, which helps
# the meta-prompt's self-validation walk-through land consistently.
_OPTIONAL_KEYS = frozenset({"judge_api_key_env", "derive_enable_thinking"})


@dataclass(frozen=True)
class RefineConfig:
    """Validated Phase 3 configuration.

    All fields are required; the loader rejects unknown keys to keep
    config drift from masking misconfig as silent default usage.
    """

    enabled: bool
    sample_size: int
    n_bins: int
    seed: int
    meta_prompt_path: str
    # Cap on FIT exemplars included in the rubric-derive prompt. When a
    # converged session has more FITs than this, ``derive_rubric``
    # embeds each FIT's span and picks the most semantically diverse
    # subset via greedy farthest-first selection. Prevents FITs from
    # overflowing the meta-prompt and crowding out NOT_FIT examples.
    max_fit_examples: int
    # Same cap, applied to NOT_FIT exemplars. The Phase 1 NOT_FIT floor
    # (``harvest.min_not_fit``) keeps this list non-empty; this cap
    # bounds it from above when an operator rates many negatives.
    max_not_fit_examples: int
    derive_model: str
    derive_base_url: str
    derive_temperature: float
    judge_model: str
    judge_base_url: str
    judge_concurrency: int
    judge_qps_limit: float
    judge_tpm_limit: float
    judge_timeout_seconds: int
    judge_max_retries: int
    api_key_env: str
    auto_drop_known_intruders: bool
    judge_api_key_env: str | None = None  # falls back to api_key_env when None
    derive_enable_thinking: bool = False  # extra_body={"enable_thinking": True}

    @property
    def per_decile_sample(self) -> int:
        """Convenience: chunks drawn per decile bin."""
        return self.sample_size // self.n_bins


def load_refine_config(
    path: Path | None = None,
    session_overrides: dict[str, Any] | None = None,
) -> RefineConfig:
    """Load and validate the ``refine:`` section.

    Raises :class:`RefineConfigError` on any structural or value
    failure. The exception message is intended to be operator-facing
    — it names the offending key, the expected shape, and (where
    relevant) the reason the constraint exists.

    ``session_overrides`` (when present) shadows YAML values for this
    load only; see :mod:`src.web_api.schemas.SessionOverrides`.
    """
    try:
        raw = load_section(_SECTION, explicit=path, overrides=session_overrides)
    except ConfigFileError as exc:
        raise RefineConfigError(str(exc)) from exc

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise RefineConfigError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}"
        )
    allowed = _REQUIRED_KEYS | _OPTIONAL_KEYS
    unknown = raw.keys() - allowed
    if unknown:
        raise RefineConfigError(
            f"config section '{_SECTION}' contains unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )

    enabled = _require_bool(raw, "enabled")
    auto_drop_known_intruders = _require_bool(raw, "auto_drop_known_intruders")

    sample_size = _require_positive_int(raw, "sample_size")
    n_bins = _require_positive_int(raw, "n_bins")
    max_fit_examples = _require_positive_int(raw, "max_fit_examples")
    max_not_fit_examples = _require_positive_int(raw, "max_not_fit_examples")
    if sample_size % n_bins != 0:
        raise RefineConfigError(
            f"config section '{_SECTION}': 'sample_size' ({sample_size}) must be "
            f"divisible by 'n_bins' ({n_bins}). The stratified draw needs equal "
            "per-bin counts so monotonic keep-rate curves are readable."
        )

    seed = _require_int(raw, "seed")  # any int OK, including 0

    meta_prompt_path = _require_nonempty_string(raw, "meta_prompt_path")
    derive_model = _require_nonempty_string(raw, "derive_model")
    derive_base_url = _require_nonempty_string(raw, "derive_base_url")
    judge_model = _require_nonempty_string(raw, "judge_model")
    judge_base_url = _require_nonempty_string(raw, "judge_base_url")
    api_key_env = _require_nonempty_string(raw, "api_key_env")
    judge_api_key_env: str | None = None
    if "judge_api_key_env" in raw:
        judge_api_key_env = _require_nonempty_string(raw, "judge_api_key_env")
    derive_enable_thinking = False
    if "derive_enable_thinking" in raw:
        derive_enable_thinking = _require_bool(raw, "derive_enable_thinking")

    derive_temperature = _require_nonneg_float(raw, "derive_temperature")
    if derive_temperature > 2.0:
        raise RefineConfigError(
            f"config section '{_SECTION}': 'derive_temperature' "
            f"({derive_temperature}) above 2.0 — almost certainly a misconfig."
        )

    judge_concurrency = _require_positive_int(raw, "judge_concurrency")
    judge_qps_limit = _require_positive_float(raw, "judge_qps_limit")
    judge_tpm_limit = _require_positive_float(raw, "judge_tpm_limit")
    judge_timeout_seconds = _require_positive_int(raw, "judge_timeout_seconds")
    judge_max_retries = raw["judge_max_retries"]
    if (
        not isinstance(judge_max_retries, int)
        or isinstance(judge_max_retries, bool)
        or judge_max_retries < 0
    ):
        raise RefineConfigError(
            f"config section '{_SECTION}': 'judge_max_retries' must be a "
            "non-negative integer (0 disables retries)."
        )

    return RefineConfig(
        enabled=enabled,
        sample_size=sample_size,
        n_bins=n_bins,
        seed=seed,
        meta_prompt_path=meta_prompt_path,
        max_fit_examples=max_fit_examples,
        max_not_fit_examples=max_not_fit_examples,
        derive_model=derive_model,
        derive_base_url=derive_base_url,
        derive_temperature=derive_temperature,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_concurrency=judge_concurrency,
        judge_qps_limit=judge_qps_limit,
        judge_tpm_limit=judge_tpm_limit,
        judge_timeout_seconds=judge_timeout_seconds,
        judge_max_retries=judge_max_retries,
        api_key_env=api_key_env,
        auto_drop_known_intruders=auto_drop_known_intruders,
        judge_api_key_env=judge_api_key_env,
        derive_enable_thinking=derive_enable_thinking,
    )


def _require_bool(raw: dict, key: str) -> bool:
    value = raw[key]
    if not isinstance(value, bool):
        raise RefineConfigError(
            f"config section '{_SECTION}': '{key}' must be a boolean"
        )
    return value


def _require_int(raw: dict, key: str) -> int:
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise RefineConfigError(
            f"config section '{_SECTION}': '{key}' must be an integer"
        )
    return value


def _require_positive_int(raw: dict, key: str) -> int:
    value = _require_int(raw, key)
    if value <= 0:
        raise RefineConfigError(
            f"config section '{_SECTION}': '{key}' must be a positive integer"
        )
    return value


def _require_nonneg_float(raw: dict, key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefineConfigError(
            f"config section '{_SECTION}': '{key}' must be a number"
        )
    if value < 0:
        raise RefineConfigError(f"config section '{_SECTION}': '{key}' must be >= 0")
    return float(value)


def _require_positive_float(raw: dict, key: str) -> float:
    value = _require_nonneg_float(raw, key)
    if value <= 0:
        raise RefineConfigError(f"config section '{_SECTION}': '{key}' must be > 0")
    return value


def _require_nonempty_string(raw: dict, key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise RefineConfigError(
            f"config section '{_SECTION}': '{key}' must be a non-empty string"
        )
    return value
