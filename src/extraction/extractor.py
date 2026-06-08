"""SpanExtractor — LLM-driven span extraction with caching.

Mirrors the client construction pattern in ``src.reflection.agent``:
OpenAI SDK pointed at an OpenAI-compatible endpoint (default
OpenRouter), YAML-backed config at repo root. Temperature is pinned to
0 (not configurable) so the cache key is stable across runs.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from src.config_loader import ConfigFileError, load_section

from .cache import CacheKey, SpanCache
from .errors import ExtractionError
from .prompt import build_messages, load_extraction_prompts
from .schema import SpanExtractionResult

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SECTION = "extraction"
_REQUIRED_KEYS = frozenset(
    {"model", "base_url", "prompt_version", "api_key_env", "cache_root"}
)
# Optional dual-vendor keys (issue #54). When all three are set the
# session factory builds a ``DualSpanExtractor`` that unions the two
# vendors' span_line_indices; when all three are unset the existing
# single-vendor path is preserved. Mixed configurations (one or two
# set) raise ``ExtractionError`` at load time.
_SECONDARY_KEYS = frozenset(
    {"secondary_model", "secondary_base_url", "secondary_api_key_env"}
)

# Total LLM attempts per extract() call. The first failure (API error,
# empty content, or schema rejection) triggers exactly one retry; if
# both attempts fail the original ExtractionError surfaces to the
# caller, which drops the chunk via dropped_by_extractor.
_MAX_LLM_ATTEMPTS = 2


@dataclass(frozen=True)
class ExtractionConfig:
    model: str
    base_url: str
    prompt_version: str
    api_key_env: str
    cache_root: Path
    # Dual-vendor extraction (issue #54). All three are ``None`` for
    # single-vendor mode; set together for dual-vendor mode.
    secondary_model: str | None = None
    secondary_base_url: str | None = None
    secondary_api_key_env: str | None = None

    @property
    def has_secondary(self) -> bool:
        return self.secondary_model is not None


def _load_config(path: Path | None = None) -> ExtractionConfig:
    try:
        raw = load_section(_SECTION, explicit=path)
    except ConfigFileError as exc:
        raise ExtractionError(str(exc)) from exc

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise ExtractionError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}"
        )
    allowed = _REQUIRED_KEYS | _SECONDARY_KEYS
    unknown = raw.keys() - allowed
    if unknown:
        raise ExtractionError(
            f"config section '{_SECTION}' contains unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )

    for key in ("model", "base_url", "prompt_version", "api_key_env"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ExtractionError(
                f"config section '{_SECTION}': '{key}' must be a non-empty string"
            )
    if not isinstance(raw["cache_root"], str) or not raw["cache_root"].strip():
        raise ExtractionError(
            f"config section '{_SECTION}': 'cache_root' must be a non-empty string"
        )

    cache_root = Path(raw["cache_root"])
    if not cache_root.is_absolute():
        cache_root = _REPO_ROOT / cache_root

    # Dual-vendor: all-or-nothing on the three secondary keys.
    present = _SECONDARY_KEYS & raw.keys()
    if present and present != _SECONDARY_KEYS:
        missing_sec = _SECONDARY_KEYS - present
        raise ExtractionError(
            f"config section '{_SECTION}': dual-vendor extraction requires "
            f"all of {sorted(_SECONDARY_KEYS)}; missing {sorted(missing_sec)}"
        )
    if present:
        for key in _SECONDARY_KEYS:
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise ExtractionError(
                    f"config section '{_SECTION}': "
                    f"'{key}' must be a non-empty string"
                )

    return ExtractionConfig(
        model=raw["model"],
        base_url=raw["base_url"],
        prompt_version=raw["prompt_version"],
        api_key_env=raw["api_key_env"],
        cache_root=cache_root,
        secondary_model=raw.get("secondary_model"),
        secondary_base_url=raw.get("secondary_base_url"),
        secondary_api_key_env=raw.get("secondary_api_key_env"),
    )


def _extract_json(raw: str) -> str:
    """Strip ``<think>`` tags and extract the first JSON object from wrapper text."""
    cleaned = _THINK_RE.sub("", raw).strip()
    if cleaned.startswith("{"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class SpanExtractor:
    """Extract a 0-3 line concept span from one chunk at a time."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: OpenAI | None = None,
        config_path: Path | None = None,
        cache: SpanCache | None = None,
        # Optional per-instance overrides (issue #54). Used by the
        # dual-vendor factory to build a secondary extractor whose
        # ``model`` / ``base_url`` / ``api_key_env`` differ from the
        # primary while sharing the same prompt_version and cache.
        model_override: str | None = None,
        base_url_override: str | None = None,
        api_key_env_override: str | None = None,
    ) -> None:
        cfg = _load_config(config_path)

        self._model = model_override or cfg.model
        self._prompt_version = cfg.prompt_version
        base_url = base_url_override or cfg.base_url
        api_key_env = api_key_env_override or cfg.api_key_env

        if client is None:
            resolved_key = api_key or os.environ.get(api_key_env)
            if resolved_key is None:
                raise ExtractionError(
                    f"No API key: set {api_key_env} or pass api_key=/client="
                )
            client = OpenAI(api_key=resolved_key, base_url=base_url)
        self._client = client

        if cache is None:
            cache = SpanCache(cfg.cache_root)
        self._cache = cache

        self._system_block, self._user_template = load_extraction_prompts()

        self._latency_ms_total = 0.0
        self._cache_hits = 0
        self._call_count = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def latency_ms_total(self) -> float:
        return round(self._latency_ms_total, 2)

    def reset_stats(self) -> None:
        self._latency_ms_total = 0.0
        self._cache_hits = 0
        self._call_count = 0

    def extract(
        self,
        *,
        query: str,
        chunk_content: str,
        prior_fit_spans: list[str],
    ) -> SpanExtractionResult:
        self._call_count += 1
        key = CacheKey(
            model_id=self._model,
            prompt_version=self._prompt_version,
            query=query,
            chunk_content=chunk_content,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        messages = build_messages(
            query,
            prior_fit_spans,
            chunk_content,
            system_block=self._system_block,
            user_template=self._user_template,
        )

        for attempt in range(1, _MAX_LLM_ATTEMPTS + 1):
            try:
                result = self._call_llm_once(messages)
                # Bounds-check against the chunk's line count. The
                # Pydantic schema validates length / sign / ordering
                # but ``chunk_content`` isn't in scope at schema time,
                # so the upper bound has to be checked here. Models
                # that occasionally hallucinate an out-of-range index
                # (issue #51) would otherwise reach ``derive_span_text``
                # in the caller and raise an uncaught ``IndexError``.
                n_lines = len(chunk_content.split("\n"))
                out_of_range = [
                    i for i in result.span_line_indices if i >= n_lines
                ]
                if out_of_range:
                    raise ExtractionError(
                        f"LLM returned out-of-range span_line_indices "
                        f"{out_of_range} for chunk with {n_lines} lines "
                        f"(valid 0..{n_lines - 1})"
                    )
            except ExtractionError as exc:
                if attempt == _MAX_LLM_ATTEMPTS:
                    raise
                log.warning(
                    "Span extraction attempt %d/%d failed (%s); retrying once.",
                    attempt,
                    _MAX_LLM_ATTEMPTS,
                    exc,
                )
                continue
            self._cache.put(key, result)
            return result

        # Unreachable: the loop body either returns or re-raises on the
        # final attempt. Kept so type checkers see a return on every path.
        raise ExtractionError("Span extraction loop exited without result")

    def _call_llm_once(self, messages: list[dict[str, str]]) -> SpanExtractionResult:
        """Single LLM round-trip + parse + validate. Raises ExtractionError on any failure."""
        llm_start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
                response_format={"type": "json_object"},
                # Hard-disable Qwen-family chain-of-thought reasoning
                # on DashScope: the extractor returns a tiny JSON
                # object, has no benefit from extended thinking, and
                # the extra tokens dominate Phase 1 latency. Other
                # OpenAI-compatible endpoints ignore unknown
                # ``extra_body`` keys, so this is safe across vendors.
                extra_body={"enable_thinking": False},
            )
        except Exception as exc:
            self._latency_ms_total += (time.perf_counter() - llm_start) * 1000.0
            raise ExtractionError(f"LLM API call failed: {exc}") from exc
        self._latency_ms_total += (time.perf_counter() - llm_start) * 1000.0

        content = response.choices[0].message.content
        if content is None:
            raise ExtractionError("LLM returned empty content")

        extracted = _extract_json(content)
        try:
            return SpanExtractionResult.model_validate_json(extracted)
        except ValidationError as exc:
            raise ExtractionError(
                f"Span result failed validation: {exc}; raw={content[:200]!r}"
            ) from exc

    @staticmethod
    def derive_span_text(chunk_content: str, indices: list[int]) -> str:
        """Join the selected lines verbatim. Empty list -> empty string."""
        if not indices:
            return ""
        lines = chunk_content.split("\n")
        return "\n".join(lines[i] for i in indices)
