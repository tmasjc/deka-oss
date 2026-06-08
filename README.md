# Deka

**Definition and Embedding Knowledge Alignment** — a human-in-the-loop
workbench that escalates a domain expert's intuitive notion of a query into a
precise, reproducible, and scalable set of labelled results over a vector-search
corpus. The operator only ever renders judgements about content; the harness
owns all retrieval mechanics.

It proceeds in four phases:

1. **Probe** — interactively tune hybrid (dense + learned-sparse) retrieval until the operator's relevance judgements converge.
2. **Harvest** — treat the validated FIT examples as a query and sweep the corpus for their embedding neighbourhood.
3. **Refine** — distil that geometric cohort into an explicit, auditable language rubric, then judge a stratified sample with an LLM.
4. **Apply** — train a low-cost classifier on the rubric-judged sample to label the full cohort at near-zero marginal cost.

See [`whitepaper/whitepaper.md`](whitepaper/whitepaper.md) for the full design.

## File structure

```
.
├── src/                     # Python backend
│   ├── search/              # Phase 1 — hybrid search + RRF fusion
│   ├── reflection/          # Phase 1 — LLM reflection agent
│   ├── extraction/          # span extraction (cached)
│   ├── anchor/              # Phase 2 — FIT-anchored harvest
│   ├── refine/              # Phase 3 — rubric derivation + LLM judging
│   ├── apply/               # Phase 4 — logistic-regression classifier
│   ├── session/             # session state machine
│   ├── scopes/              # corpus scope registry
│   ├── postgres/            # original-content fetcher
│   ├── replay/              # convergence metrics
│   ├── logging/             # progress-log writer
│   ├── auth/                # cookie-session auth
│   └── web_api/             # FastAPI backend (the `deka-web` entry point)
├── web/                     # React + Vite frontend (screens, components, state)
├── harness/prompts/         # runtime LLM prompts (system, reflection, extraction, rubric)
├── whitepaper/              # design paper + figures
├── tests/                   # pytest suite
├── config.yaml.example      # service endpoints + per-phase tuning
├── scopes.yaml.example      # scope → {Milvus collection, Postgres table}
├── users.yaml.example       # web auth — users + token SHA-256s
├── .env.example             # API keys + endpoint overrides
└── pyproject.toml           # Python project (managed with uv)
```

The `*.example` files are templates; copy each to its real name (gitignored) to
override it. Loaders fall back to the example when the real file is absent.

## Setup

### Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/)
- **Node.js 18+** (for the web UI)
- For live retrieval: a **Milvus** instance, an **embeddings service** returning
  dense + learned-sparse vectors, an **OpenRouter-compatible LLM** endpoint, and
  (optional) **PostgreSQL** for original-content lookup.

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure

```bash
cp .env.example .env                 # set OPENROUTER_API_KEY
cp config.yaml.example config.yaml   # point search.embed_url / milvus_uri and postgres.dsn at your services
cp scopes.yaml.example scopes.yaml   # map each scope to your Milvus collection + Postgres table
cp users.yaml.example users.yaml     # add a web user (next step)
```

### 3. Create a login token

The web UI gates every request on a signed-cookie session keyed to `users.yaml`.
Generate a token and store **only its SHA-256**:

```bash
python -c "import secrets,hashlib; t=secrets.token_hex(32); print('token: ',t); print('sha256:',hashlib.sha256(t.encode()).hexdigest())"
```

Put the `sha256` under a user `id` in `users.yaml`; keep the `token` to log in.
Optionally `export DEKA_SESSION_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(32))")` so sessions survive a backend restart.

## Run the web app

Start the backend and the frontend in two terminals:

```bash
# Terminal 1 — FastAPI backend (http://127.0.0.1:8787)
uv run deka-web

# Terminal 2 — Vite dev server (http://localhost:5173)
cd web && npm install && npm run dev
```

Open <http://localhost:5173> and sign in with the token from step 3.

> **Single-server alternative:** `cd web && npm install && npm run build`, then
> `uv run deka-web` serves the built UI directly at <http://127.0.0.1:8787>.

## Development

```bash
uv run pytest        # run the test suite
uv run ruff check .  # lint
```
