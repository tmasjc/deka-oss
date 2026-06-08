"""Derive a query-specific rubric prompt from a converged Phase 1
session.

Three public functions:

- :func:`derive_rubric` — runs the meta-prompt against a strong LLM
  and returns the parsed rubric prompt + metadata. One retry on a
  parser failure with the parser error injected as feedback.
- :func:`parse_rubric_prompt` — re-parse a markdown rubric prompt
  back into structured :class:`RubricMetadata`. Used both on derive
  output and on operator-edited prompts.
- :func:`render_rubric_prompt` — render :class:`RubricMetadata` back
  to canonical markdown. Symmetric with :func:`parse_rubric_prompt`
  on round-trip-clean inputs.

The HTML-comment fence convention (``<!-- check_id: x -->...<!-- /check -->``)
is documented in ``harness/schemas/rubric.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

from openai import OpenAI
from pydantic import ValidationError

from src.paths import prompt_path
from src.prompt_io import (
    load_named_fence_sections,
    prompt_sha256,
)
from src.search.config import SearchConfig, load_default_config
from src.search.embedding import get_embeddings
from src.search.errors import EmbeddingServiceError

from .config import RefineConfig
from .errors import RefineError, RefineParseError
from .load_session import Phase3SessionInputs
from .schema import (
    DeriveLLMOutput,
    RubricCheck,
    RubricFitExample,
    RubricMetadata,
    RubricNotFitExample,
)
from .select import select_diverse

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DERIVE_MAX_ATTEMPTS = 2

_FAILED_CHECK_HEADER_RE = re.compile(
    r"^##\s+失败检查枚举:\s*(?P<ids>[^\n]+?)\s*$",
    re.MULTILINE,
)
_CHECK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHECK_BLOCK_RE = re.compile(
    r"<!--\s*check_id:\s*(?P<id>[^\s-][^\s]*)\s*-->\s*\n"
    r"(?P<body>.*?)"
    r"\n\s*<!--\s*/check\s*-->",
    re.DOTALL,
)
_FIT_BLOCK_RE = re.compile(
    r"<!--\s*fit_example:\s*pk=(?P<pk>[^\s-][^\s]*)\s*-->\s*\n"
    r"(?P<body>.*?)"
    r"\n\s*<!--\s*/fit_example\s*-->",
    re.DOTALL,
)
_NOT_FIT_BLOCK_RE = re.compile(
    r"<!--\s*not_fit_example:\s*pk=(?P<pk>\S+)\s+fails=(?P<fails>[a-z0-9_,\s]+)\s*-->\s*\n"
    r"(?P<body>.*?)"
    r"\n\s*<!--\s*/not_fit_example\s*-->",
    re.DOTALL,
)
_QUERY_LINE_RE = re.compile(
    r"^(?:查询|种子查询|概念|概念（种子查询）):\s*(?P<query>.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class DeriveResult:
    """Output of one derive call."""

    rubric_text: str
    metadata: RubricMetadata
    derive_model_id: str
    latency_ms: float
    attempts: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_rubric(
    *,
    inputs: Phase3SessionInputs,
    cfg: RefineConfig,
    client: OpenAI | None = None,
    api_key: str | None = None,
    repo_root: Path | None = None,
    search_config: SearchConfig | None = None,
) -> DeriveResult:
    """Run the meta-prompt against the derive LLM, then render a
    canonical rubric prompt.

    The LLM emits JSON describing the discriminators (checks +
    per-NOT_FIT failure annotations). The harness combines this with
    the session's FIT/NOT_FIT chunks to build a
    :class:`RubricMetadata` and renders canonical markdown via
    :func:`render_rubric_prompt`. Retries once on JSON parse / schema
    failure, injecting the validation error as feedback.

    Caps the FIT / NOT_FIT example pool to ``cfg.max_fit_examples`` /
    ``cfg.max_not_fit_examples`` via embedding-driven farthest-first
    selection so the rubric prompt stays bounded even on long Phase 1
    sessions. The embed call only fires when truncation is actually
    needed; sessions already within both caps skip it entirely. Pass
    ``search_config`` to use a non-default embed endpoint.
    """
    inputs = _apply_example_caps(inputs, cfg, search_config)
    repo_root = repo_root or _REPO_ROOT
    mpp = Path(cfg.meta_prompt_path)
    if mpp.is_absolute():
        meta_prompt_path = mpp
    elif mpp.parts[:2] == ("harness", "prompts") and len(mpp.parts) == 3:
        # Conventional default form `harness/prompts/<name>` — route via
        # PROMPTS_DIR so DEKA_PROMPTS_DIR overrides apply uniformly with
        # the reflection / extraction loaders.
        meta_prompt_path = prompt_path(mpp.name)
    else:
        meta_prompt_path = repo_root / mpp
    system_instructions, context_template, user_task = load_named_fence_sections(
        meta_prompt_path,
        ("系统", "上下文模板", "用户消息（渲染后）"),
    )
    meta_prompt_full = meta_prompt_path.read_text(encoding="utf-8")
    meta_sha = prompt_sha256(meta_prompt_full)

    client = client or _build_client(cfg, api_key)
    rendered_context = _render_context(context_template, inputs)
    # Static instructions + per-call FIT/NOT_FIT context together form the
    # cacheable prefix. Validation feedback lands after the user task on
    # retry, so the system bytes stay identical across attempts and the
    # bulky context block sits inside the cache region.
    system_message = (
        system_instructions + "\n\n# 输入\n\n" + rendered_context
    )

    last_error: str | None = None
    total_latency_ms = 0.0
    attempt = 0
    derive_output: DeriveLLMOutput | None = None
    response_model_id = cfg.derive_model
    for attempt in range(1, _DERIVE_MAX_ATTEMPTS + 1):
        user_for_attempt = user_task
        if last_error is not None:
            user_for_attempt = (
                user_task
                + "\n\n# Validation feedback on previous attempt\n\n"
                + "Your previous output failed JSON schema validation:\n\n"
                + f"```\n{last_error}\n```\n\n"
                + "Re-emit the JSON object. Match the schema exactly: "
                "no prose around the JSON, every NOT_FIT pk must appear "
                "in `not_fit_annotations` with a non-empty `fails` list "
                "of declared check ids."
            )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_for_attempt},
        ]

        # DashScope thinking-capable models (deepseek-v4-pro, qwen3+)
        # double their reasoning budget when enable_thinking is set,
        # which helps the meta-prompt's self-validation step actually
        # get exercised. Forwarded via extra_body so non-thinking
        # endpoints (OpenRouter, plain qwen-plus) ignore it harmlessly.
        extra_body = (
            {"enable_thinking": True} if cfg.derive_enable_thinking else None
        )
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=cfg.derive_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=cfg.derive_temperature,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )
        except Exception as exc:
            total_latency_ms += (time.perf_counter() - start) * 1000.0
            raise RefineError(f"Derive LLM call failed: {exc}") from exc
        total_latency_ms += (time.perf_counter() - start) * 1000.0
        response_model_id = response.model or cfg.derive_model

        content = (response.choices[0].message.content or "").strip()
        if not content:
            last_error = "LLM returned empty content"
            continue

        cleaned = _extract_json_object(content)
        try:
            derive_output = DeriveLLMOutput.model_validate_json(cleaned)
        except ValidationError as exc:
            last_error = str(exc)
            log.warning(
                "Derive attempt %d/%d failed validation: %s",
                attempt,
                _DERIVE_MAX_ATTEMPTS,
                str(exc)[:200],
            )
            log.warning(
                "Derive attempt %d/%d raw response: %s",
                attempt,
                _DERIVE_MAX_ATTEMPTS,
                content,
            )
            continue
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
            log.warning(
                "Derive attempt %d/%d emitted invalid JSON",
                attempt,
                _DERIVE_MAX_ATTEMPTS,
            )
            log.warning(
                "Derive attempt %d/%d raw response: %s",
                attempt,
                _DERIVE_MAX_ATTEMPTS,
                content,
            )
            continue
        break

    if derive_output is None:
        raise RefineParseError(
            f"Derive LLM produced unparseable output after "
            f"{_DERIVE_MAX_ATTEMPTS} attempts. Last error: {last_error}"
        )

    metadata = _build_metadata_from_derive_output(
        derive_output=derive_output,
        inputs=inputs,
        derive_model_id=response_model_id,
        meta_prompt_path=_relative_path(meta_prompt_path, repo_root),
        meta_prompt_sha256=meta_sha,
    )
    rubric_text = render_rubric_prompt(metadata)
    metadata = metadata.model_copy(update={"prompt_sha256": prompt_sha256(rubric_text)})

    return DeriveResult(
        rubric_text=rubric_text,
        metadata=metadata,
        derive_model_id=response_model_id,
        latency_ms=round(total_latency_ms, 2),
        attempts=attempt,
    )


def _build_metadata_from_derive_output(
    *,
    derive_output: DeriveLLMOutput,
    inputs: Phase3SessionInputs,
    derive_model_id: str,
    meta_prompt_path: str,
    meta_prompt_sha256: str,
) -> RubricMetadata:
    """Combine the LLM's discriminator structure with the session's
    chunks to produce a :class:`RubricMetadata`.

    NOT_FIT pks the LLM didn't annotate fall back to ``[checks[0].id]``
    — the harness will not silently drop a labelled NOT_FIT, it just
    can't promise a per-example discriminator if the LLM omitted one.
    """
    annotations_by_pk: dict[str, list[str]] = {}
    for ann in derive_output.not_fit_annotations:
        annotations_by_pk[str(ann.pk)] = list(ann.fails)

    fallback_check = derive_output.checks[0].id

    fit_examples = [
        RubricFitExample(pk=f.pk, span_text=f.span_text)
        for f in inputs.fits
    ]
    not_fit_examples = [
        RubricNotFitExample(
            pk=n.pk,
            span_text=_first_lines(n.chunk_content, 3),
            fails=annotations_by_pk.get(str(n.pk), [fallback_check]),
        )
        for n in inputs.not_fits
    ]

    return RubricMetadata(
        query=inputs.query,
        source_session_id=inputs.session_id,
        derive_model_id=derive_model_id,
        meta_prompt_path=meta_prompt_path,
        meta_prompt_sha256=meta_prompt_sha256,
        checks=[
            RubricCheck(id=c.id, description=c.description)
            for c in derive_output.checks
        ],
        fit_examples=fit_examples,
        not_fit_examples=not_fit_examples,
        prompt_path="",  # writer fills this in
        prompt_sha256="0" * 64,  # placeholder; render_rubric_prompt updates
        version=1,
    )


def _first_lines(text: str, n: int) -> str:
    """Trim a NOT_FIT chunk to its first n non-empty lines for the
    rubric exemplar. The judge sees the whole chunk at runtime; the
    rubric prompt only needs enough text to anchor the discriminator.
    """
    out: list[str] = []
    for line in text.splitlines():
        if line.strip():
            out.append(line.strip())
            if len(out) >= n:
                break
    return " / ".join(out) if out else "(empty)"


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _extract_json_object(raw: str) -> str:
    """Strip optional outer fences and surrounding prose from the
    derive LLM's JSON output.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl >= 0:
            inner = cleaned[first_nl + 1 :]
            if inner.rstrip().endswith("```"):
                cleaned = inner.rstrip()[:-3].rstrip()
    if cleaned.startswith("{"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_rubric_prompt(
    text: str,
    *,
    source_session_id: str,
    derive_model_id: str,
    meta_prompt_path: str,
    meta_prompt_sha256: str,
    prompt_path: str,
    version: int,
) -> RubricMetadata:
    """Parse a rubric-prompt markdown into :class:`RubricMetadata`.

    The same function runs on derive output and on operator-edited
    rubric prompts (called from the editor's save path). Any
    structural failure raises :class:`RefineParseError` with a
    message naming the missing/malformed marker.
    """
    if not text or not text.strip():
        raise RefineParseError("Rubric prompt is empty")

    enum_match = _FAILED_CHECK_HEADER_RE.search(text)
    if not enum_match:
        raise RefineParseError(
            "Missing '## 失败检查枚举: <id>, <id>, ...' line"
        )
    enum_ids = _split_enum_ids(enum_match.group("ids"))
    if not enum_ids:
        raise RefineParseError(
            "'## 失败检查枚举:' line declares no ids"
        )
    seen_ids: set[str] = set()
    for cid in enum_ids:
        if not _CHECK_ID_RE.match(cid):
            raise RefineParseError(
                f"Check id {cid!r} does not match [a-z][a-z0-9_]*"
            )
        if cid in seen_ids:
            raise RefineParseError(f"Check id {cid!r} declared twice in enum")
        seen_ids.add(cid)
    enum_set = frozenset(enum_ids)

    system_block, context_block, user_block = _split_rubric_sections(text)

    # Check blocks live in the system block.
    checks: list[RubricCheck] = []
    declared_in_blocks: set[str] = set()
    for match in _CHECK_BLOCK_RE.finditer(system_block):
        cid = match.group("id").strip()
        body = match.group("body").strip()
        if cid in declared_in_blocks:
            raise RefineParseError(
                f"Check id {cid!r} has duplicate <!-- check_id: ... --> blocks"
            )
        if cid not in enum_set:
            raise RefineParseError(
                f"<!-- check_id: {cid} --> block found but not declared in "
                f"'## 失败检查枚举:' (declared: {sorted(enum_set)})"
            )
        if not body:
            raise RefineParseError(
                f"<!-- check_id: {cid} --> block has empty body"
            )
        declared_in_blocks.add(cid)
        checks.append(
            RubricCheck(id=cid, description=_strip_leading_dash(body), required=True)
        )
    missing = enum_set - declared_in_blocks
    if missing:
        raise RefineParseError(
            f"'## 失败检查枚举:' declares ids {sorted(missing)} but "
            "no matching <!-- check_id: ... --> blocks found in the system "
            "fenced block"
        )

    # User message must contain the {numbered_chunk} placeholder verbatim.
    if "{numbered_chunk}" not in user_block:
        raise RefineParseError(
            "User-message fenced block missing literal {numbered_chunk} "
            "placeholder"
        )

    # Query line lives in the context block.
    query_match = _QUERY_LINE_RE.search(context_block)
    if not query_match:
        raise RefineParseError(
            "Context fenced block missing '查询: ...' line"
        )
    query = query_match.group("query").strip()
    if not query:
        raise RefineParseError("Query line is empty")

    # FIT examples — context block.
    fit_examples: list[RubricFitExample] = []
    for match in _FIT_BLOCK_RE.finditer(context_block):
        pk_raw = match.group("pk").strip()
        body = match.group("body").strip()
        if not body:
            raise RefineParseError(
                f"<!-- fit_example: pk={pk_raw} --> block has empty body"
            )
        fit_examples.append(
            RubricFitExample(
                pk=_coerce_pk(pk_raw), span_text=_strip_leading_dash(body)
            )
        )
    if not fit_examples:
        raise RefineParseError(
            "Context fenced block declares no <!-- fit_example: ... --> blocks"
        )

    # NOT_FIT examples — context block.
    not_fit_examples: list[RubricNotFitExample] = []
    for match in _NOT_FIT_BLOCK_RE.finditer(context_block):
        pk_raw = match.group("pk").strip()
        fails_raw = match.group("fails").strip()
        body = match.group("body").strip()
        fails = [s.strip() for s in fails_raw.split(",") if s.strip()]
        if not fails:
            raise RefineParseError(
                f"<!-- not_fit_example: pk={pk_raw} --> declares no failed checks"
            )
        unknown = set(fails) - enum_set
        if unknown:
            raise RefineParseError(
                f"<!-- not_fit_example: pk={pk_raw} --> references unknown "
                f"check ids {sorted(unknown)}; declared: {sorted(enum_set)}"
            )
        if not body:
            raise RefineParseError(
                f"<!-- not_fit_example: pk={pk_raw} --> block has empty body"
            )
        not_fit_examples.append(
            RubricNotFitExample(
                pk=_coerce_pk(pk_raw),
                span_text=_strip_leading_dash(body),
                fails=fails,
            )
        )

    # System block must mention the JSON output schema fields. This is a
    # weak check (regex match on field names) but catches a derive LLM
    # that forgot to include the contract entirely.
    for required_field in ("verdict", "evidence_line_indices", "failed_check"):
        if required_field not in system_block:
            raise RefineParseError(
                f"System fenced block missing JSON contract field "
                f"'{required_field}'"
            )

    metadata = RubricMetadata(
        query=query,
        source_session_id=source_session_id,
        derive_model_id=derive_model_id,
        meta_prompt_path=meta_prompt_path,
        meta_prompt_sha256=meta_prompt_sha256,
        checks=checks,
        fit_examples=fit_examples,
        not_fit_examples=not_fit_examples,
        prompt_path=prompt_path,
        prompt_sha256=prompt_sha256(text),
        version=version,
    )
    return metadata


def render_rubric_prompt(metadata: RubricMetadata) -> str:
    """Render :class:`RubricMetadata` back to canonical markdown.

    Symmetric with :func:`parse_rubric_prompt` on round-trip-clean
    inputs: ``parse(render(m)) == m`` modulo ``prompt_sha256``
    (which depends on the rendered text and is therefore re-computed
    after rendering).
    """
    enum_line = "## 失败检查枚举: " + ", ".join(c.id for c in metadata.checks)

    check_blocks: list[str] = []
    for check in metadata.checks:
        check_blocks.append(
            f"<!-- check_id: {check.id} -->\n- {check.description}\n<!-- /check -->"
        )
    checks_rendered = "\n\n".join(check_blocks)

    fit_blocks = []
    for fit in metadata.fit_examples:
        fit_blocks.append(
            f"<!-- fit_example: pk={fit.pk} -->\n- {fit.span_text}\n<!-- /fit_example -->"
        )

    not_fit_blocks = []
    for nf in metadata.not_fit_examples:
        fails_str = ",".join(nf.fails)
        not_fit_blocks.append(
            f"<!-- not_fit_example: pk={nf.pk} fails={fails_str} -->\n"
            f"- {nf.span_text}\n<!-- /not_fit_example -->"
        )

    system_block = f"""你是一名专业的评审员。请将下列命名 check 应用于候选片段，
并输出严格的 JSON 判定结果。

{checks_rendered}

规则:
- ``evidence_line_indices`` 必须逐字引用候选片段的行号。
- 提供 1–3 个升序、唯一的索引，全部落在候选片段的行号范围内。
- 当 KEEP 时: ``failed_check`` 必须为 null。
- 当 DROP 时: ``failed_check`` 必须是上方闭合枚举中第一个失败的 check。
- 边界情况默认 DROP。

请按以下精确 JSON 形式回应:
{{"verdict": "KEEP" | "DROP",
 "evidence_line_indices": [int, int?, int?],
 "failed_check": "<枚举中的一项>" | null,
 "reason": "<一句简短中文>"}}"""

    context_block = f"""查询: {metadata.query}

FIT 示例（包含该概念）:
{chr(10).join(fit_blocks)}

NOT_FIT 示例（与概念相邻但不属于）:
{chr(10).join(not_fit_blocks)}"""

    # 候选片段 lives alone in the user message — it is the only part
    # that varies per chunk. Putting query + FIT + NOT_FIT in 上下文
    # (which the harness concatenates into the system message at judge
    # time) keeps the prefix byte-identical across every chunk in a
    # judge run, so DashScope's automatic prefix cache hits on every
    # call after the first.
    user_block = "候选片段（已编号）:\n{numbered_chunk}"

    text = (
        f"# 评分细则: {metadata.query}\n\n"
        f"{enum_line}\n\n"
        f"## 系统\n\n```\n{system_block}\n```\n\n"
        f"## 上下文\n\n```\n{context_block}\n```\n\n"
        f"## 用户消息（渲染后）\n\n```\n{user_block}\n```\n"
    )
    return text


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _build_client(cfg: RefineConfig, api_key: str | None) -> OpenAI:
    resolved = api_key or os.environ.get(cfg.api_key_env)
    if resolved is None:
        raise RefineError(
            f"No API key for derive LLM: set {cfg.api_key_env} or pass "
            "api_key=/client="
        )
    return OpenAI(api_key=resolved, base_url=cfg.derive_base_url)


def _apply_example_caps(
    inputs: Phase3SessionInputs,
    cfg: RefineConfig,
    search_config: SearchConfig | None,
) -> Phase3SessionInputs:
    """Cap ``inputs.fits`` / ``inputs.not_fits`` to the configured maxima.

    No-op when both pools are already within their caps. When either
    overflows, embeds that pool's texts via the BGE-M3 ``/embed-all``
    endpoint and picks the most semantically diverse subset.

    Embedding failure surfaces as :class:`RefineError`; the operator
    needs to know the cap could not be applied rather than silently
    truncating to the first N.
    """

    need_fit_cap = len(inputs.fits) > cfg.max_fit_examples
    need_not_fit_cap = len(inputs.not_fits) > cfg.max_not_fit_examples
    if not (need_fit_cap or need_not_fit_cap):
        return inputs

    resolved_search = search_config or load_default_config()

    fits = list(inputs.fits)
    if need_fit_cap:
        fit_texts = [f.span_text for f in fits]
        fit_vectors = _embed_for_selection(fit_texts, resolved_search, "FIT")
        original = len(fits)
        fits = select_diverse(fits, fit_vectors, cfg.max_fit_examples)
        log.info(
            "Rubric example cap: kept %d/%d FITs via farthest-first",
            len(fits),
            original,
        )

    not_fits = list(inputs.not_fits)
    if need_not_fit_cap:
        not_fit_texts = [n.chunk_content for n in not_fits]
        not_fit_vectors = _embed_for_selection(
            not_fit_texts, resolved_search, "NOT_FIT"
        )
        original = len(not_fits)
        not_fits = select_diverse(
            not_fits, not_fit_vectors, cfg.max_not_fit_examples
        )
        log.info(
            "Rubric example cap: kept %d/%d NOT_FITs via farthest-first",
            len(not_fits),
            original,
        )

    return replace(inputs, fits=fits, not_fits=not_fits)


def _embed_for_selection(
    texts: list[str], search_config: SearchConfig, label: str
) -> list[list[float]]:
    """Pull dense embeddings for ``texts`` from the configured embed
    service. Raises :class:`RefineError` so the operator sees a clear
    rubric-side message rather than the raw transport error."""

    try:
        response = get_embeddings(
            texts,
            search_config.embed_url,
            timeout=search_config.http_timeout,
        )
    except EmbeddingServiceError as exc:
        raise RefineError(
            f"Embedding {label} exemplars for diversity selection "
            f"failed: {exc}"
        ) from exc

    # BGE-M3's /embed-all returns ``dense_embeddings``; legacy/mocks
    # may use ``dense`` — accept either, matching src/anchor/loader.py.
    dense = response.get("dense_embeddings")
    if dense is None:
        dense = response.get("dense")
    if not isinstance(dense, list) or len(dense) != len(texts):
        raise RefineError(
            f"Embed service returned malformed `dense` for {label} "
            f"exemplars: expected list of length {len(texts)}, got "
            f"{type(dense).__name__}"
            + (f" of length {len(dense)}" if isinstance(dense, list) else "")
        )
    return dense


def _render_context(template: str, inputs: Phase3SessionInputs) -> str:
    fit_block = "\n\n".join(
        f"- pk={f.pk}\n  chunk_id={f.chunk_id}\n  span: {f.span_text}"
        for f in inputs.fits
    )
    not_fit_block = "\n\n".join(
        f"- pk={n.pk}\n  chunk_id={n.chunk_id}\n  chunk_content:\n{_indent(n.chunk_content, 4)}"
        for n in inputs.not_fits
    )
    diagnoses_block = (
        "\n".join(f"- {d}" for d in inputs.reflection_diagnoses)
        if inputs.reflection_diagnoses
        else "(none)"
    )
    # Plain string substitution (not .format()) — the meta-prompt's
    # context template may contain literal braced text the deriver
    # treats as schema, which .format() would mis-interpret as a
    # missing placeholder.
    return (
        template.replace("{query}", inputs.query)
        .replace("{fit_examples}", fit_block)
        .replace("{not_fit_examples}", not_fit_block)
        .replace("{reflection_diagnoses}", diagnoses_block)
    )


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def _split_enum_ids(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _strip_leading_dash(text: str) -> str:
    """Strip a leading ``- `` bullet marker if present."""
    stripped = text.strip()
    if stripped.startswith("- "):
        stripped = stripped[2:]
    elif stripped.startswith("-"):
        stripped = stripped[1:].lstrip()
    return stripped.strip()


def _coerce_pk(raw: str) -> str | int:
    """PKs may be int or string. Coerce numeric-looking strings to int
    (Milvus pks are int64 in the project's collections) but keep strings
    as-is for collections that use VARCHAR keys.
    """
    if raw.lstrip("-").isdigit():
        return int(raw)
    return raw


_HEADING_RE = re.compile(r"^##\s+(?P<header>.+?)\s*$", re.MULTILINE)


def _split_rubric_sections(text: str) -> tuple[str, str, str]:
    """Extract the System + Context + User-message sections from a
    rubric prompt.

    Returns ``(system_body, context_body, user_body)``. Each body has
    its outer triple-backtick fence stripped if present, otherwise
    it's the bare text under the heading.

    Permissive on the fence: derive LLMs sometimes fence one section
    and not the other (we've observed this on real output). The HTML-
    comment markers inside each body are what the harness actually
    keys on, so the outer fence is decorative.

    Raises :class:`RefineParseError` if any of the three headings is
    missing.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        header = match.group("header").strip()
        if header.startswith("失败检查枚举"):
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body.startswith("```"):
            first_nl = body.find("\n")
            if first_nl >= 0:
                inner = body[first_nl + 1 :]
                if inner.rstrip().endswith("```"):
                    body = inner.rstrip()[:-3].rstrip()
        if header not in sections:
            sections[header] = body

    system = sections.get("系统")
    context = sections.get("上下文")
    user = sections.get("用户消息（渲染后）")
    if system is None:
        raise RefineParseError("Missing '## 系统' header")
    if context is None:
        raise RefineParseError("Missing '## 上下文' header")
    if user is None:
        raise RefineParseError("Missing '## 用户消息（渲染后）' header")
    return system, context, user


def _strip_outer_fence(text: str) -> str:
    """If the LLM wrapped the whole rubric in an outer ``markdown`` fence,
    strip it. Some models do this even after being told not to.

    Conservative: only strips when the *entire* text is one fenced block.
    Inner fences (the ## 系统 and ## 用户消息（渲染后） blocks) are
    untouched.
    """
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        # Find the first newline; everything before is the language tag.
        first_nl = stripped.find("\n")
        if first_nl < 0:
            return stripped
        body = stripped[first_nl + 1 : -3]
        return body.rstrip("\n")
    return stripped


# ---------------------------------------------------------------------------
# JSON helpers (used by writer + tests)
# ---------------------------------------------------------------------------


def metadata_to_json(metadata: RubricMetadata) -> str:
    """Serialise :class:`RubricMetadata` to a stable, indented JSON string."""
    return json.dumps(metadata.model_dump(mode="json"), indent=2, ensure_ascii=False)


def metadata_from_json(text: str) -> RubricMetadata:
    """Inverse of :func:`metadata_to_json`."""
    return RubricMetadata.model_validate_json(text)
