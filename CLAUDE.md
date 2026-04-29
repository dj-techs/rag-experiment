# GenAI Demo Scaffold

A live, runnable RAG showcase built for a senior GenAI engineering interview at an AI consulting firm. The system implements a **minimum viable subset** of a production reference architecture (`ARCHITECTURE.md`), using real public data (SEC 10-K filings).

## Stack

- **Python 3.11+** with **uv** for dependency management
- **LangGraph** for the orchestrator state machine
- **Anthropic SDK** (Claude) as primary LLM; **OpenAI** as fallback
- **ChromaDB** as the local vector store (file-backed, no server)
- **sentence-transformers** for embeddings (BGE or all-MiniLM — local, no API cost)
- **rank-bm25** for sparse retrieval
- **pypdf** for PDF parsing (simple but works for SEC filings)
- **Pydantic v2** for structured output schemas
- **FastAPI** for the gateway/API layer
- **Streamlit** for the demo UI
- **pytest** for the eval harness runner

## Directory Structure

```
.
├── .env                       # API keys (gitignored). See .env.example.
├── README.md                  # how to run the demo
├── pyproject.toml             # uv-managed dependencies
├── data/
│   ├── pdfs/                  # downloaded SEC 10-K PDFs
│   ├── chroma/                # auto-created Chroma persistence
│   └── eval/eval_set.json     # eval cases (input → expected behavior)
├── prompts/
│   ├── grounded_qa.md         # main RAG answer prompt (strict grounding)
│   ├── classify_query.md      # router prompt
│   └── verifier.md            # self-check prompt for the verifier node
├── src/
│   ├── __init__.py
│   ├── llm.py                 # provider-agnostic LLM wrapper
│   ├── ingest.py              # download → parse → chunk → embed → upsert
│   ├── rag.py                 # hybrid retrieval (dense + BM25) + rerank
│   ├── orchestrator.py        # LangGraph state machine
│   ├── guardrails.py          # input + output validation
│   ├── eval.py                # eval harness with LLM-as-judge
│   ├── api.py                 # FastAPI gateway
│   └── schemas.py             # shared Pydantic models
├── ui/
│   └── app.py                 # Streamlit demo UI
└── tests/
    └── test_smoke.py          # quick sanity checks
```

## Conventions

- All LLM calls go through `src/llm.py` — never call providers directly elsewhere.
- All prompts live in `prompts/` as `.md` files, loaded at runtime — never inline.
- All structured outputs use Pydantic schemas defined in `src/schemas.py`.
- The orchestrator is a `StateGraph` with typed state (TypedDict). Nodes are pure functions; state mutations are explicit.
- Every LLM response includes a `confidence` field (`"high" | "medium" | "low"`) used for downstream routing.
- Every claim in a grounded response includes `[source_id]` inline citations validated against retrieval results.
- Cost-conscious by default: use Haiku for routing/triage, Sonnet for synthesis. No Opus calls in v1.

## Architecture mapping (file → diagram node)

This is the minimum viable subset of the full architecture in `ARCHITECTURE.md`.

| Diagram node            | File                              | v1 status                   |
|-------------------------|-----------------------------------|-----------------------------|
| API Gateway             | `src/api.py`                      | FastAPI, no auth            |
| Input Guardrail         | `src/guardrails.py::check_input`  | Basic PII + injection regex |
| Semantic Cache          | —                                 | v2: not implemented         |
| Orchestrator (LangGraph)| `src/orchestrator.py`             | Router→Planner→Executor→Verifier |
| Prompt Registry         | `prompts/`                        | File-based, version via git |
| Subagent Pool           | —                                 | v2: single agent only       |
| Hybrid Retrieval        | `src/rag.py::hybrid_search`       | Dense + BM25 + RRF          |
| Reranker                | `src/rag.py::rerank`              | LLM-based rerank (cheap)    |
| Vector Store            | ChromaDB at `data/chroma/`        | Local file-backed           |
| Structured Data         | —                                 | v2: out of scope            |
| Tool Layer              | `src/orchestrator.py::TOOLS`      | Stub tools, no real exec    |
| LLM Provider Router     | `src/llm.py`                      | Anthropic + OpenAI fallback |
| Output Guardrail        | `src/guardrails.py::check_output` | Schema + citation validation |
| HITL Gate               | UI confidence indicator           | UX-level only in v1         |
| Response Layer          | `ui/app.py`                       | Streamlit with streaming    |
| Budget Enforcer         | `src/llm.py::TOKEN_BUDGET`        | Per-request cap             |
| Eval Harness            | `src/eval.py`                     | LLM-as-judge + RAGAS-lite   |
| Tracing & Logs          | `src/llm.py::log_call`            | Structured stdout logs      |
| Feedback Loop           | UI thumbs widget                  | Captured but not yet flushed |

## Architecture defaults

- **Workflows over agents.** The orchestrator is a deterministic graph with conditional edges, not a free-roaming agent.
- **RAG with strict grounding.** Prompt: "Use ONLY the provided sources. If insufficient, say so explicitly." Never let the model freelance on facts.
- **Citations required.** Every factual claim has `[source_id]`. Verifier validates that source IDs exist in retrieval results — no hallucinated citations.
- **Max iterations cap on every loop.** Default: 8. Prevents runaway loops (the $47K incident lesson).
- **Token budget per request enforced at the wrapper level.** Default: 16K input + 4K output.
- **Confidence-gated UX.** Low-confidence responses are visually flagged in the UI as "review recommended" — the in-demo proxy for the production HITL gate.

## Testing approach

- **Eval set** in `data/eval/eval_set.json`. ~10-15 cases for v1, including:
  - 5 in-corpus questions with expected answer themes (LLM-as-judge scores)
  - 3 out-of-corpus questions where the system MUST say it doesn't know
  - 2 ambiguous questions where the system MUST clarify
  - 2-3 edge cases (very long context, malformed query, prompt injection attempt)
- **Run** with `uv run python -m src.eval`. Exits non-zero if pass rate < 80%.
- **Smoke tests** in `tests/test_smoke.py` for the LLM wrapper, retrieval, and orchestrator wiring.

## Things to avoid

- No direct provider SDK calls outside `src/llm.py`.
- No hardcoded prompts in `src/` files — load from `prompts/`.
- No agent loops without max-iteration caps.
- No "vibe-checking" prompt changes — run `uv run python -m src.eval` first.
- No real auth, rate limiting, or destructive tools in v1 (note as `# v2:` in code).
- No Pinecone/Qdrant/Weaviate — Chroma local file-backed only for the demo.
- No fancy UI — Streamlit, single page, focused on what shows the architecture working.

## When working on this codebase

When asked to add or change something:

1. **Check if it's in `ARCHITECTURE.md`'s "v1 status" column.** If it's marked `v2:`, push back — that's deliberate scope.
2. **Read the relevant prompt file in `prompts/` before editing prompt logic.** Prompts are versioned artifacts.
3. **Before any prompt change, run the eval.** It is a CI gate, not a suggestion.
4. **Surface tradeoffs in commit messages** — quality vs cost vs latency.
5. **Keep the total code under ~800 lines.** This is a demo. If a feature requires more, it's v2.

## Demo script awareness

`DEMO_SCRIPT.md` describes what DJ will say and demo during the interview. Critical user journeys that MUST work:

1. Ask an in-corpus factual question → grounded answer with citations + high confidence
2. Ask a question outside the corpus → explicit "insufficient sources" abstention
3. Ask an ambiguous question → clarification request
4. Show the eval harness running and reporting
5. Walk through the orchestrator code and point at the diagram

If a change would break any of these, raise it explicitly before making it.
