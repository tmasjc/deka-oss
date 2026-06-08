# Rubric-derivation meta-prompt

This is the **meta-prompt**: the harness's contract with a strong LLM
that authors a query-specific **rubric prompt** from one converged
Phase 1 session. The LLM emits JSON describing the discriminators;
the harness renders the JSON into a canonical markdown rubric prompt
that any LLM-with-JSON-mode endpoint can apply chunk-by-chunk.

Two layers, two stability requirements:

- The **rubric prompt** the harness renders is locked per turn so
  the judge's verdicts reflect a single predicate.
- This meta-prompt must be stable across queries so the rubric
  prompts the harness produces all share the same shape, JSON
  contract, and quality bar.

The corpus is Chinese sales-conversation transcripts. The deriver
LLM reads Chinese FIT/NOT_FIT spans and emits Chinese check
`description` text; only the JSON keys, check `id` slugs, and
structural fields stay English.

## Three-section shape (caching-aware)

The harness loads three named ``##`` sections from this file and
assembles a Messages-API call as:

- ``system`` = the ``## 系统`` block (static instructions) followed
  by the rendered ``## 上下文模板`` (per-call inputs).
- ``user`` = the ``## 用户消息（渲染后）`` block (short task
  instruction). On a JSON-validation retry, validation feedback is
  appended *after* the task — so the system bytes stay identical
  across attempts and the bulky FIT/NOT_FIT prefix sits inside the
  cacheable region.

Inputs the harness will inject (named placeholders the context
template references):

- `{query}` — the seed query of the Phase 1 session.
- `{fit_examples}` — formatted block of `(pk, span_text)` for every
  FIT-rated chunk from Phase 1.
- `{not_fit_examples}` — formatted block of `(pk, chunk_content)` for
  every NOT_FIT-rated chunk from Phase 1.
- `{reflection_diagnoses}` — concatenated `diagnose` strings from the
  Phase 1 reflections that named the discriminator pattern.

The meta-prompt's task: identify the smallest set of named **checks**
that reproduces the FIT/NOT_FIT split on the given examples, and
annotate each NOT_FIT with the check id(s) it violates.

---

## 系统

```
你为一条特定查询撰写鉴别器结构（即评分细则的判别项）。
请阅读 FIT 与 NOT_FIT 示例，找出二者之间的语言学边界，
并输出严格的 JSON 对象，由系统渲染为 markdown 形式的评分细则。

# 鉴别器识别任务

FIT 与 NOT_FIT 通常共享话题词。两者之间的边界往往体现在
以下语言学维度上：

- 言语行为（陈述 / 提问 / 指令 / 转述）
- 说话者（对话中是哪一方说出）
- 对象（话语指向的人或事物）
- 立场（肯定 / 否定 / 含糊）
- 关系（谁对谁施加动作）

请找出最小的一组命名 check，使其作为整体能够正确地
KEEP 全部 FIT 示例并 DROP 全部 NOT_FIT 示例。每个 check
是一条单一的判别命题，由判定 LLM 对候选片段进行评估。

质量规则：

- 每个 check 必须以「候选必须积极满足才能 KEEP」的方式表达。
- 边界情况默认 DROP。
- 仅依赖候选片段、查询，以及给定的示例。
- check 的 `id` 必须匹配 `[a-z][a-z0-9_]*`，最长 64 字符，
  按其测试的维度语义化命名（例如 `speaker`、`speech_act`、
  `target`）。
- **description 必须描述「语义关系」，不能编码「字面词汇」。**
  评分细则的目的是识别 FIT 示例共享的语义特征，而不是
  少数 FIT 示例最显眼的表层用词。
    - ✅「表达对持续来电的不满、抗议或希望停止的诉求」
    - ❌「明确说出『投诉』一词或威胁投诉」
    - ✅「说话者承担受影响方的角色，陈述自身遭遇」
    - ❌「使用『我』作为主语并出现『骚扰』一词」
  好的 description 即使在 FIT 示例之间表述差异很大时，
  仍能通过语义共性识别它们。
- 通常 3 条 check 比较合适。2 条可用于概念非常紧凑的情况；
  4 条或更多通常意味着概念尚未被干净地拆解。

# 自检步骤（强制，输出前完成）

在生成最终 JSON 之前，请对每一条候选 check 逐个对照
**每一个** FIT 示例的 span，做下面的内部检查（不写入输出）：

1. 默念该 FIT 示例的 span 文字。
2. 问自己：「把这段 span 交给判定 LLM，按这条 check 的
   description 字面判定，它会 KEEP 吗？」严格按字面理解
   description，不要替候选脑补任何未明说的语义。
3. 如果有 **任何** FIT 示例在某条 check 上不能 KEEP，
   说明该 check 抓的是少数 FIT 的表层特征，不是全组
   FIT 共有的语义特征。请：
    - 把该 check 的 description 改写得更**语义化**、
      更宽泛，直到全部 FIT 示例都能 KEEP；或者
    - 如果改写后仍覆盖不了，删除该 check，另外构造一条
      真正抓住全组 FIT 共性的 check。

只有当**全部** FIT 示例都能通过你列出的**每一条** check，
你才可以输出 JSON。如果做不到这一点，宁可让 check 集合
更宽——后续 NOT_FIT 例会被列入 `not_fit_annotations`，
说明它们如何在某条 check 上失败；NOT_FIT 不必每条都被
**所有** check 拒绝，只需至少一条拒绝。

# 输出契约

输出一个 JSON 对象，且严格符合以下形态（不要 markdown，
不要任何解说，不要代码围栏，不要多余文字）：

{
  "checks": [
    {"id": "<snake_case_id>", "description": "<一到两句中文>"},
    ...
  ],
  "not_fit_annotations": [
    {"pk": <pk>, "fails": ["<check_id>", ...]},
    ...
  ]
}

强制约束：

- `description` 字段必须使用中文撰写，与语料一致。
- `id` 字段必须是英文 snake_case（保持为符号 ID，跨语言稳定）。
- `checks` 必须包含 2-4 条记录。每个 `id` 唯一，并匹配
  `[a-z][a-z0-9_]*`。每个 `description` 是一到两句中文。
- `not_fit_annotations` 必须列出输入中**每一个** NOT_FIT 的
  pk。对每一个，`fails` 是一个**非空** JSON 字符串数组，
  其中每个字符串都是 `checks` 中已声明的 check id。
  绝不能是整数，绝不能是计数：即便只有一个 check 失败，
  也必须输出 `["check_id"]`。如果有多个 check 失败，则按
  最关键失败在前的顺序列出全部 id。
- 使用输入中提供的 pk 原值——整数即整数，字符串即字符串。
- 不要任何额外的顶层字段，JSON 周围不要任何文字。

举例（仅作形态参考，你的取值会不同）：

{
  "checks": [
    {"id": "speech_act", "description": "该话语是显式的投诉或威胁投诉，而非对第三方的转述。"},
    {"id": "speaker", "description": "说话者是受影响的本人，而非销售方或无关第三方。"},
    {"id": "target", "description": "投诉对象是公司本身，而非无关实体。"}
  ],
  "not_fit_annotations": [
    {"pk": "abc-123-9-pre-p", "fails": ["speaker"]},
    {"pk": "def-456-0-pre-p", "fails": ["speech_act", "target"]}
  ]
}

注意 `fails` 始终是字符串 JSON 数组，单一失败的情况也是
`["speaker"]`。
```

## 上下文模板

```
查询: {query}

# FIT 示例（包含该概念）

下列 Phase 1 评定为 FIT 的片段表达了该概念。人工评估者
高亮的 span 是使每个片段成为 FIT 的逐字短语。请阅读它们；
你给出的鉴别器应当让下面**每一个** span 都积极满足。

{fit_examples}

# NOT_FIT 示例（与概念相邻但不属于）

下列 Phase 1 评定为 NOT_FIT 的片段是与 FIT 一起被检索出来的
（因此共享话题词），但被人工评估者剔除。对每一条，请识别
它在哪个语言学维度上区别于 FIT。每一条 NOT_FIT 必须至少
被你给出的 check 集合中的一条所拒绝。

{not_fit_examples}

# 反思诊断（Phase 1 智能体的自我刻画，可能为空）

{reflection_diagnoses}
```

## 用户消息（渲染后）

```
请基于上文给出的查询、FIT 与 NOT_FIT 示例和反思诊断，
输出一个 JSON 对象，符合系统提示中声明的形态：

- `checks`（2-4 条）声明鉴别器。
- `not_fit_annotations` 将每一个 NOT_FIT pk 映射到它失败的
  check id 列表。

只输出 JSON 对象。不要代码围栏，不要解说，不要多余文字。
```
