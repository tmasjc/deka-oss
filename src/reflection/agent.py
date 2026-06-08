"""ReflectionAgent — narrative reflection using an LLM.

Uses the OpenAI SDK pointed at an OpenAI-compatible endpoint (default:
OpenRouter) to generate structured reflection output.

Reflection is a reasoning-trace, not a tuning controller: it emits
observe / diagnose / hypothesize and, optionally, a CONVERGED status.
The session config is locked once turn 1 starts; the only mid-session
change is a path drop performed via the operator-triggered audit flow
(see :meth:`SessionState.apply_path_drop`). No prescription is parsed
or applied.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from src.config_loader import ConfigFileError, load_section

from .errors import LLMCallError, PromptAssemblyError
from .prompt import build_messages, load_reflection_instructions, load_system_prompt
from .schema import ReflectionOutput

if TYPE_CHECKING:
    from src.session.state import SessionState

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_SECTION = "reflection"
_REQUIRED_KEYS = frozenset({"model", "base_url", "temperature", "api_key_env"})


@dataclass(frozen=True)
class ReflectionConfig:
    model: str
    base_url: str
    temperature: float
    api_key_env: str


def _load_config(path: Path | None = None) -> ReflectionConfig:
    """Load reflection config from the unified YAML; every key is required."""
    try:
        raw = load_section(_SECTION, explicit=path)
    except ConfigFileError as exc:
        raise PromptAssemblyError(str(exc)) from exc

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise PromptAssemblyError(
            f"config section '{_SECTION}' missing required keys: {sorted(missing)}"
        )
    unknown = raw.keys() - _REQUIRED_KEYS
    if unknown:
        raise PromptAssemblyError(
            f"config section '{_SECTION}' contains unknown keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_REQUIRED_KEYS)}"
        )

    for key in ("model", "base_url", "api_key_env"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise PromptAssemblyError(
                f"config section '{_SECTION}': '{key}' must be a non-empty string"
            )
    if not isinstance(raw["temperature"], (int, float)):
        raise PromptAssemblyError(
            f"config section '{_SECTION}': 'temperature' must be a number"
        )

    return ReflectionConfig(
        model=raw["model"],
        base_url=raw["base_url"],
        temperature=float(raw["temperature"]),
        api_key_env=raw["api_key_env"],
    )


def _extract_json(raw: str) -> str:
    """Strip ``<think>`` tags and extract a JSON object from wrapper text.

    Thinking models (e.g. qwen3-max-thinking) wrap output in
    ``<think>...</think>`` tags.  Some models also add prose around
    the JSON object.  This function handles both cases.
    """
    cleaned = _THINK_RE.sub("", raw).strip()
    if cleaned.startswith("{"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class ReflectionAgent:
    """Narrative reflection agent — emits a per-turn reasoning trace."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: OpenAI | None = None,
        config_path: Path | None = None,
    ) -> None:
        cfg = _load_config(config_path)

        resolved_key = api_key or os.environ.get(cfg.api_key_env)
        if resolved_key is None and client is None:
            raise PromptAssemblyError(
                f"No API key: set {cfg.api_key_env} or pass api_key="
            )

        self._client = client or OpenAI(
            api_key=resolved_key,
            base_url=cfg.base_url,
        )
        self._model = cfg.model
        self._temperature = cfg.temperature

        # Load and cache static prompts
        self._system_prompt = load_system_prompt()
        self._reflection_instructions = load_reflection_instructions()

    def reflect(self, state: "SessionState") -> dict[str, Any] | None:
        """Run narrative reflection on the latest completed turn.

        Returns a reflection dict matching ``progress_log.md`` schema,
        or ``None`` if reflection cannot proceed. The returned dict
        carries a ``_diagnostics`` key consumed by the logging hook
        and stripped by the canonical-progress-log writer.
        """
        if not state.turns:
            return None

        messages = build_messages(
            state, self._system_prompt, self._reflection_instructions
        )

        diagnostics: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": messages,
            "raw_response": None,
            "extracted_json": None,
            "latency_ms": None,
            "parse_error": None,
        }

        llm_start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMCallError(f"LLM API call failed: {exc}") from exc
        diagnostics["latency_ms"] = round((time.perf_counter() - llm_start) * 1000.0, 2)

        content = response.choices[0].message.content
        diagnostics["raw_response"] = content
        if content is None:
            log.warning("LLM returned empty content (possible refusal)")
            return {"_diagnostics": diagnostics}

        extracted = _extract_json(content)
        diagnostics["extracted_json"] = extracted

        try:
            output = ReflectionOutput.model_validate_json(extracted)
        except Exception as exc:
            log.warning(
                "Failed to parse LLM response as ReflectionOutput; raw=%s",
                content[:300],
            )
            diagnostics["parse_error"] = str(exc)
            return {"_diagnostics": diagnostics}

        result = output.to_log_dict()
        # Defensive sanitisers on the recommendation field. The prompt
        # rules these cases out but the LLM can still hallucinate;
        # the operator must never see a modal/banner for a
        # recommendation that won't apply or shouldn't repeat. The
        # suppressed value lands on ``_diagnostics`` so analytics
        # tooling can count these as recommendation-precision misses.
        rec = result.get("path_drop_recommendation")
        if rec is not None:
            suppress_reason: str | None = None
            if state.recommended_drop_applied:
                # Session-level safeguard: once an apply has landed,
                # no further recommendations surface for the rest of
                # the session. Each session is entitled to one
                # agent-recommended drop; cascading drops would
                # compound the trust shift the recommendation flow
                # already carries.
                suppress_reason = "recommended_drop_already_applied"
            elif rec.get("path") not in state.current_config.active_paths:
                # The agent occasionally recommends dropping a path
                # that's already been dropped (reasoning from the
                # historical progress log instead of the current
                # config). Strip and stash for analytics.
                suppress_reason = "path_inactive"

            if suppress_reason is not None:
                log.warning(
                    "Suppressing path_drop_recommendation (%s; path=%r, "
                    "active_paths=%s, recommended_drop_applied=%s)",
                    suppress_reason,
                    rec.get("path"),
                    sorted(state.current_config.active_paths),
                    state.recommended_drop_applied,
                )
                diagnostics["suppressed_recommendation"] = {
                    "reason": suppress_reason,
                    "recommendation": rec,
                    "active_paths": sorted(state.current_config.active_paths),
                    "recommended_drop_applied": state.recommended_drop_applied,
                }
                result.pop("path_drop_recommendation", None)
        result["_diagnostics"] = diagnostics
        return result
