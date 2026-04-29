# Build Plan

This is the ordered task list Claude Code should execute. Work top-to-bottom. After each phase, run a smoke test before moving on.

## Phase 0 — Project setup (5 min)

1. Initialize the project with `uv init --python 3.11`
2. Create the directory structure per `CLAUDE.md`
3. Write `pyproject.toml` with these dependencies:
   - `anthropic>=0.40` — Claude SDK
   - `openai>=1.50` — fallback provider
   - `langgraph>=0.2` — orchestration
   - `langchain-core>=0.3` — typed messages
   - `chromadb>=0.5` — vector store
   - `sentence-transformers>=3.0` — embeddings (all-MiniLM-L6-v2 default)
   - `rank-bm25>=0.2` — sparse retrieval
   - `pypdf>=4.0` — PDF parsing
   - `pydantic>=2.0` — schemas
   - `fastapi>=0.115` — gateway
   - `uvicorn>=0.30` — ASGI server
   - `streamlit>=1.40` — UI
   - `requests>=2.32` — for SEC EDGAR downloads
   - `python-dotenv>=1.0`
   - `pytest>=8.0` (dev)
4. Create `.env.example` with `ANTHROPIC_API_KEY=`, `OPENAI_API_KEY=`, `USER_AGENT=` (required by SEC EDGAR)
5. Create `.gitignore` covering `.env`, `data/chroma/`, `data/pdfs/`, `__pycache__/`, `.venv/`
6. Run `uv sync` and confirm clean install

**Smoke test:** `uv run python -c "import anthropic, langgraph, chromadb, streamlit; print('ok')"`

## Phase 1 — LLM wrapper (`src/llm.py`) (10 min)

This is the foundation everything depends on. Get it right first.

Required exports:
- `class LLMResponse(pydantic.BaseModel)` — `text: str`, `input_tokens: int`, `output_tokens: int`, `cost_usd: float`, `model: str`, `latency_ms: int`
- `def call_llm(messages, *, model_tier="reasoning", max_tokens=4096, system=None, response_format=None) -> LLMResponse`
- `def stream_llm(messages, *, model_tier="reasoning", max_tokens=4096, system=None) -> Iterator[str]`
- `TOKEN_BUDGET = {"input": 16_000, "output": 4_000}` — enforce these per call, raise on overage
- `MODEL_TIERS = {"triage": "claude-haiku-4-5-...", "reasoning": "claude-sonnet-4-6-...", "hard": "claude-opus-4-7"}` — confirm exact model strings; if uncertain, ask the user or use the latest known-stable identifiers

Internal:
- `_log_call(...)` — structured stdout log with tokens, cost, latency, model. JSON one-liner format.
- Provider fallback: try Anthropic; on failure (auth, network), try OpenAI with equivalent tier. Log the fallback.
- Cost calculation using a small dict of per-million-token prices. Update once at top of file with current numbers; don't try to fetch live.

**Smoke test:** Add a `__main__` block that does a 1-token call and prints the LLMResponse. Run it.

## Phase 2 — Schemas (`src/schemas.py`) (3 min)

Centralize all Pydantic models so they're imported everywhere from one place:

- `class Citation(BaseModel)` — `source_id: str`, `chunk_id: str`, `excerpt: str`, `page: int | None`
- `class GroundedAnswer(BaseModel)` — `answer: str`, `citations: list[Citation]`, `confidence: Literal["high", "medium", "low"]`, `reasoning: str`
- `class QueryClassification(BaseModel)` — `intent: Literal["factual", "ambiguous", "out_of_scope"]`, `requires_clarification: bool`, `clarifying_question: str | None`
- `class RetrievalResult(BaseModel)` — `chunk_id: str`, `source_id: str`, `text: str`, `score: float`, `metadata: dict`
- `class OrchestratorState(TypedDict)` — `query`, `tenant_id`, `classification`, `retrieved`, `answer`, `confidence`, `iteration`, `error`

## Phase 3 — Ingestion (`src/ingest.py`) (15 min)

A CLI script that:

1. **Downloads SEC 10-K filings** for 3 companies (default: Apple AAPL, Microsoft MSFT, NVIDIA NVDA, most recent fiscal year). Use SEC EDGAR full-text search API or direct CIK lookups. Respect SEC's request requirements: include a User-Agent header from `USER_AGENT` env var (format: "Name email@example.com").
   - Fallback: if EDGAR is being awkward, ship a small script that downloads from known-stable URLs (the company's IR page often hosts PDF copies). Keep it simple.
2. **Saves PDFs** to `data/pdfs/` with stable filenames like `aapl_10k_2024.pdf`
3. **Parses each PDF** with pypdf, page-by-page, preserving page numbers as metadata
4. **Chunks** with simple recursive-character splitting (1000 tokens, 200 overlap). Note `# v2: structure-aware chunking` in a comment.
5. **Embeds** using sentence-transformers `all-MiniLM-L6-v2` (small, fast, local — no API cost during the demo)
6. **Upserts** into Chroma collection `sec_filings` with metadata: `source_id` (filename), `page`, `chunk_index`, `company`, `filing_year`, `tenant_id` (default: "demo"). Stable IDs: `f"{source_id}-{chunk_index}"`.
7. **Builds a BM25 index** alongside, persisted as a pickle to `data/bm25.pkl` (chroma doesn't ship sparse retrieval out of the box; we maintain it ourselves)
8. **Prints a summary**: docs, chunks, total tokens, est. cost (zero, since we're using local embeddings)

CLI: `uv run python -m src.ingest [--companies AAPL,MSFT,NVDA] [--year 2024]`

**Smoke test:** Run the ingest script and confirm `data/chroma/` and `data/bm25.pkl` exist. Open Chroma collection and run a sample query.

## Phase 4 — Retrieval (`src/rag.py`) (10 min)

Required functions:

- `def hybrid_search(query: str, *, top_k: int = 20, tenant_id: str = "demo") -> list[RetrievalResult]`:
  - Embed query, run dense search via Chroma → top_k results
  - Run BM25 search → top_k results
  - **Reciprocal Rank Fusion** to merge: `score = Σ 1/(60 + rank_i)` per chunk
  - Return top_k merged results, with metadata preserved

- `def rerank(query: str, candidates: list[RetrievalResult], *, top_n: int = 5) -> list[RetrievalResult]`:
  - LLM-based rerank using triage model (Haiku — cheap)
  - Prompt the model with the query + numbered candidates, ask for top N indices in JSON
  - Defensive: parse LLM output, fall back to original order if parse fails
  - Note `# v2: cross-encoder rerank (Cohere/BGE) for production` in comment

- `def rewrite_query(query: str) -> str`:
  - HyDE-lite: ask the LLM to generate a hypothetical answer to the query
  - Return the concatenation `"{query}\n\n{hypothetical_answer}"` for retrieval
  - Use triage tier for cost; on failure return original query
  - Note `# v2: full HyDE + sub-question split + acronym resolution` in comment

**Smoke test:** Call `hybrid_search("What were Apple's revenue drivers?")` after ingest; print top 5 with scores.

## Phase 5 — Guardrails (`src/guardrails.py`) (5 min)

Required functions:

- `def check_input(text: str) -> tuple[bool, str | None]`:
  - Regex for SSN, credit card patterns, common API key prefixes (`sk-`, `ghp_`, `AKIA`)
  - Regex for known prompt injection patterns ("ignore previous", "disregard your instructions", "reveal your system prompt", base64 blobs over a length threshold)
  - Returns `(allow, reason)`. `allow=False` blocks the query at the gateway with the reason returned to user.
  - Note `# v2: ML-based PII detection + injection classifier` in comment.

- `def check_output(answer: GroundedAnswer, retrieved: list[RetrievalResult]) -> tuple[bool, list[str]]`:
  - Validate `answer.citations`: every `chunk_id` must exist in `retrieved`. Hallucinated citations → fail.
  - Validate that the answer text doesn't contain any of the input-guardrail patterns either (LLM shouldn't echo PII back).
  - Validate confidence is set.
  - Returns `(passed, list_of_failures)`. Failures get logged; in v1, surface "low confidence" badge in UI rather than blocking.

## Phase 6 — Orchestrator (`src/orchestrator.py`) (15 min)

The LangGraph state machine. This is the heart of the demo — keep it readable.

```python
from langgraph.graph import StateGraph, END
from src.schemas import OrchestratorState

def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_classification, {
        "out_of_scope": END,
        "ambiguous": END,
        "factual": "planner",
    })
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges("verifier", route_after_verify, {
        "retry": "executor",  # max 2 retries (track in state.iteration)
        "done": END,
    })

    return graph.compile()
```

Each node:

- `router_node` — uses triage tier + `prompts/classify_query.md`, populates `state["classification"]` (a QueryClassification). If out_of_scope or ambiguous, populate `state["answer"]` with the appropriate user-facing message and exit.
- `planner_node` — chooses `top_k` based on classification confidence (default 20). Stores plan parameters in state.
- `executor_node` — runs `rewrite_query → hybrid_search → rerank` → calls LLM with `prompts/grounded_qa.md` and the reranked context. Parses response into `GroundedAnswer`. Increments iteration counter.
- `verifier_node` — runs `check_output`. If failures and iteration < 2, route back to executor with feedback. Otherwise route to done.

**Max iterations cap: 2 retries (3 total attempts).** Hard limit. Surface in code as `MAX_ITERATIONS = 3`.

**Smoke test:** Build the graph, run it on "What was Apple's revenue in fiscal 2024?" and print the final state.

## Phase 7 — API gateway (`src/api.py`) (5 min)

A minimal FastAPI app:

- `POST /query` body `{ "query": str, "tenant_id": str = "demo" }`
- Calls `check_input` → on fail, return 400 with reason
- Otherwise invokes the orchestrator graph and returns the final state's answer + citations + confidence
- Add `/health` endpoint returning `{"status": "ok"}`
- Streaming: optional. For demo simplicity, return the full response. Streaming is an `# v2:` comment in code.

**Smoke test:** Run `uv run uvicorn src.api:app --reload`, hit `localhost:8000/health` and `POST /query`.

## Phase 8 — Eval harness (`src/eval.py`) (10 min)

Required behavior:

1. Load `data/eval/eval_set.json`
2. For each case, invoke the orchestrator (don't mock — run the real pipeline)
3. Score each case:
   - **Out-of-scope cases**: assert classification was `out_of_scope` and answer mentions insufficient sources. Pass/fail boolean.
   - **In-corpus factual cases**: use LLM-as-judge with `prompts/verifier.md` rubric (faithfulness, relevance, citation correctness 0-3 each). Pass if total ≥ 7/9.
   - **Ambiguous cases**: assert clarifying question was returned.
4. Report:
   - Per-case pass/fail with diff
   - Overall pass rate
   - Total cost (from LLM wrapper logs)
   - Average latency per case
5. Save report to `.eval-results/{timestamp}.json`
6. Compare to `.eval-results/last.json` if exists; flag regressions
7. Save current as `last.json`
8. Exit code: 0 if pass rate ≥ 80% AND no regression on previously-passing cases. 1 otherwise.

CLI: `uv run python -m src.eval [--cases path/to/cases.json]`

## Phase 9 — Streamlit UI (`ui/app.py`) (10 min)

Single-page Streamlit app:

- Title: "GenAI Demo — Industry RAG Showcase"
- Sidebar: brief explainer of what's loaded (companies, doc count) and a link to `architecture_diagram.png`
- Main:
  - Text input for the query
  - "Ask" button
  - Response area showing:
    - The streaming answer text
    - Confidence badge (green/yellow/red for high/med/low)
    - Citations as numbered references with hover-to-expand excerpt
    - Latency + token cost displayed unobtrusively at the bottom
  - Thumbs up/down feedback buttons → write to `data/feedback.jsonl`
- Sample queries pre-populated as buttons:
  - "What were Apple's main revenue drivers in fiscal 2024?"
  - "Compare R&D spending across the three companies"
  - "What was the weather in Tokyo last Tuesday?" (out-of-scope demo)
  - "Tell me about their products" (ambiguous demo — should ask clarification)

The UI must talk to the orchestrator either directly (import) or via the FastAPI endpoint — whichever is simpler. For the demo, direct import is fine and avoids needing two terminals.

**Smoke test:** `uv run streamlit run ui/app.py`, click each sample query, confirm all four behaviors work.

## Phase 10 — Eval cases (`data/eval/eval_set.json`) (5 min)

Use `eval_set.example.json` from the handoff package as the starting point — adjust expected behaviors if needed once the system is built. Aim for 12-15 cases covering all categories described in `CLAUDE.md`.

## Phase 11 — README + final smoke test (5 min)

Write `README.md` with:
- One-paragraph description
- "Quick start" — clone, set env, ingest, run
- Architecture summary with link to `architecture_diagram.png`
- Eval and CI gate explanation

Final end-to-end smoke test:
1. Fresh `uv sync`
2. `uv run python -m src.ingest`
3. `uv run streamlit run ui/app.py`
4. Run all 4 sample queries
5. `uv run python -m src.eval`

If all pass, the demo is ready.

## Time check

Estimated total: ~90 minutes of work for Claude Code if uninterrupted. Phases 0, 2, 5, 7, 11 are quick (~5 min each). Phase 3 (ingestion with download) is the longest single phase (~15 min including waiting for SEC downloads).

## When to ask the user

- If `ANTHROPIC_API_KEY` isn't in `.env` after Phase 0, stop and ask
- If SEC downloads fail, propose alternatives (arXiv? a smaller curated PDF set?) and let user pick
- If exact model identifiers (e.g., `claude-haiku-4-5-20251001`) need confirmation, ask before hardcoding
- Anything else, push forward and surface tradeoffs in commit messages
