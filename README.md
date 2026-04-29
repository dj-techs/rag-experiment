# GenAI Demo — Industry RAG Showcase

A working, runnable RAG showcase over **SEC 10-K filings** (Apple, Microsoft, NVIDIA — most recent fiscal year), built as the minimum viable subset of a production GenAI agent architecture (see `ARCHITECTURE.md`).

The spine: a deterministic **LangGraph** state machine (Router → Planner → Executor → Verifier) over **hybrid retrieval** (dense + BM25 + RRF) with strict-grounding prompts, citation validation, and a token-budget circuit breaker. Eval harness wired as a CI gate.

---

## Quick start

```bash
# 0. Clone + enter the repo, then:
uv sync                                              # install pinned deps
cp .env.example .env                                 # then put your key in
# echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# 1. One-time: download + index the 3 most recent 10-Ks (~3 min, $0.00 — local embeddings)
uv run python -m src.ingest

# 2. Start the API gateway (terminal 1)
uv run uvicorn src.api:app --reload

# 3. Start the Streamlit UI (terminal 2)
uv run streamlit run ui/app.py
```

Open `http://localhost:8501` and click any sample query.

To re-run the eval harness any time:

```bash
uv run python -m src.eval                            # demo threshold (0.7)
uv run python -m src.eval --threshold 0.8            # production gate
```

---

## What's in the box

| File | Diagram node | Notes |
|------|--------------|-------|
| `ui/app.py`             | Client + Response Layer | Streamlit, single page, sample queries |
| `src/api.py`            | API Gateway             | FastAPI `/query` + `/health` |
| `src/guardrails.py`     | Input + Output Guardrail| Regex PII + injection; citation cross-check |
| `src/orchestrator.py`   | Orchestrator (LangGraph)| Router/Planner/Executor/Verifier; `MAX_ITERATIONS=3` |
| `prompts/*.md`          | Prompt Registry         | Versioned via git; loaded at runtime |
| `src/rag.py`            | Retrieval + Reranker    | Dense + BM25 + RRF; LLM-based rerank (Haiku) |
| `data/chroma/`          | Vector Store            | Local file-backed Chroma |
| `src/ingest.py`         | Ingestion Pipeline      | SEC EDGAR HTML → chunk → embed → upsert |
| `src/llm.py`            | LLM Provider Router     | Anthropic primary, OpenAI fallback; token budget; logging |
| `src/eval.py`           | Eval Harness            | LLM-as-judge per `prompts/verifier.md`; CI gate |
| `tests/test_smoke.py`   | —                       | Wiring + guardrail unit tests |

Diagram: see `architecture_diagram.png`. Full architecture (including v2 components) in `ARCHITECTURE.md`.

---

## Three principles you can defend at the whiteboard

1. **Every irreversible action is gated.** v1 has no destructive tools; the architecture accommodates them via the Orchestrator's conditional edges and the (UX-level) HITL surface.
2. **Every quality decision is measurable.** `src/eval.py` is the CI gate. Prompts in `prompts/` are versioned alongside it. Per-dimension scoring (faithfulness / citation correctness / relevance) lets you see exactly what regressed.
3. **Every unit of cost has a circuit breaker.** `TOKEN_BUDGET` in `src/llm.py` (16K input / 4K output per call); `MAX_ITERATIONS=3` in `src/orchestrator.py`; cost-aware tier routing (Haiku for triage/rerank, Sonnet for synthesis, no Opus in v1).

---

## Eval harness

```bash
uv run python -m src.eval
```

Runs every case in `data/eval/eval_set.json` through the real orchestrator. Out-of-scope, ambiguous, and adversarial cases are scored by hard rules; in-corpus factual cases by **LLM-as-judge** (`prompts/verifier.md`, Sonnet) on three dimensions:

- **Faithfulness** (0-3): claims supported by sources
- **Citation correctness** (0-3): cited chunks exist and back the claim
- **Relevance** (0-3): the answer addresses the question

Pass requires total ≥ 7/9 with no zero on any individual dimension. Reports land in `.eval-results/{ts}.json`; `last.json` enables regression detection across runs.

---

## Things to know

- **Corpus** is downloaded fresh from SEC EDGAR (`data.sec.gov/submissions/...`) at ingest time. No checked-in PDFs.
- **Embeddings** use `sentence-transformers/all-MiniLM-L6-v2` locally (no API cost, ~80MB on first run).
- **`USER_AGENT`** in `.env` is required by SEC (format: `Name email@example.com`).
- **`OPENAI_API_KEY`** is optional — set it to enable the Anthropic→OpenAI fallback in `src/llm.py`.
- **All structured outputs** are validated against Pydantic schemas in `src/schemas.py`. No raw dicts cross module boundaries.
- **All prompts** live in `prompts/` as `.md` files, loaded at runtime. Edit, then `uv run python -m src.eval` before shipping.

---

## What's deliberately not here (v2)

Marked in the diagram and in code with `# v2:` comments. Pull the full list from `ARCHITECTURE.md`. Highlights: semantic cache, subagent pool, multi-tenancy enforcement, real MCP tool integration, drift monitor, real HITL queue, production observability, auth/rate limiting.
