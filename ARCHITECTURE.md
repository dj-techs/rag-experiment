# Architecture

## The diagram

See `architecture_diagram.png` in this folder. The full version is the production reference architecture; the v1 demo is a strategic subset.

## Six lanes (top-to-bottom request flow)

1. **CLIENT** — Streamlit UI in v1 (web client only)
2. **EDGE / GATEWAY** — FastAPI with input guardrail (PII + injection regex)
3. **ORCHESTRATION** — LangGraph state machine: Router → Planner → Executor → Verifier
4. **RETRIEVAL & KNOWLEDGE** — Hybrid retrieval (dense + BM25) + LLM-based reranker over ChromaDB
5. **GENERATION & TOOLS** — LLM provider wrapper, output guardrail, response streaming, budget enforcer
6. **OBSERVABILITY & CONTROL** — Eval harness as CI gate, structured stdout logs, in-UI feedback widget

## Three governing principles

1. **Every irreversible action is gated.** v1 has no destructive tools; the architecture accommodates them when added.
2. **Every quality decision is measurable.** The eval harness is a CI gate, not optional. Prompt changes don't ship without passing.
3. **Every unit of cost has a circuit breaker.** Token budget per request, max iterations per loop. Demo limits are conservative.

## Request flow (the path a query takes)

```
User → Streamlit UI
  → POST /query (FastAPI)
    → Input Guardrail (regex-based PII + injection check)
      → Orchestrator (LangGraph)
        ├─ Router: classify query (factual / ambiguous / out-of-scope)
        ├─ Planner: choose retrieval depth (top_k) and strategy
        ├─ Executor:
        │   ├─ Query Rewriter (HyDE-lite)
        │   ├─ Hybrid Retrieval (dense + BM25 + RRF)
        │   ├─ Reranker (LLM-based, top 5)
        │   └─ LLM Generation (with retrieved context + grounding prompt)
        └─ Verifier: citation grounding check + confidence assignment
      ← Output Guardrail (schema validation + citation cross-check)
    ← Streaming response with citations + confidence
  ← User sees response with inline [1][2] citations and confidence badge
  → Optional: thumbs feedback → persisted to feedback log
```

## File-to-node mapping

| File                              | Maps to diagram node              | Purpose                                                |
|-----------------------------------|------------------------------------|--------------------------------------------------------|
| `ui/app.py`                       | Client + Response Layer            | Streamlit UI, streams responses, captures feedback     |
| `src/api.py`                      | API Gateway                        | FastAPI app, single `/query` endpoint                  |
| `src/guardrails.py::check_input`  | Input Guardrail                    | PII regex + injection pattern checks                   |
| `src/orchestrator.py`             | Orchestrator (LangGraph)           | StateGraph with Router/Planner/Executor/Verifier nodes |
| `prompts/`                        | Prompt Registry                    | Versioned markdown prompts loaded at runtime           |
| `src/rag.py`                      | Query Rewriter, Hybrid Retrieval, Reranker | Full retrieval pipeline                       |
| `data/chroma/`                    | Vector Store                       | ChromaDB persistence directory                         |
| `src/ingest.py`                   | Ingestion Pipeline                 | Download → parse → chunk → embed → upsert             |
| `src/llm.py`                      | LLM Provider Router + Tracing      | Wraps Anthropic + OpenAI; logs tokens, cost, latency  |
| `src/llm.py::TOKEN_BUDGET`        | Budget Enforcer                    | Per-request token cap                                  |
| `src/guardrails.py::check_output` | Output Guardrail                   | Schema + citation cross-validation                     |
| `src/eval.py`                     | Eval Harness                       | LLM-as-judge runner with pass/fail gate                |

## What's deliberately omitted (v2)

These nodes are in the production architecture but excluded from v1 to keep the demo under 800 lines:

- **Semantic Cache** — Redis layer; ~30% cost savings in production but adds operational complexity
- **Subagent Pool** — single agent loop is sufficient for the demo's question shapes
- **Multi-tenancy** — `tenant_id` field plumbed through schemas but not enforced
- **Real MCP tool integration** — tool layer has stubs only; would need MCP server setup
- **Drift Monitor** — production concern; demo corpus is too small to need it
- **Graph KB** — not domain-relevant for SEC filings
- **Fresh Data / web search** — corpus is static for demo predictability
- **Per-tenant budget limits** — single-tenant demo
- **Production observability** — structured stdout logs in v1; LangSmith/Helicone is v2
- **Real HITL gate** — confidence is shown in UI as a visual badge, not routed to a human queue
- **Security & Audit** — single-user demo; no audit log
- **Admin Console** — out of scope; the eval harness is the operator interface for now

When the interviewer asks why something is missing, the answer is **always**: "It's in the production architecture (point to diagram), but for the demo I scoped to the minimum that demonstrates the request lifecycle, the eval gate, and the safety/cost principles. With more time the v2 work would be [X]."

## How this defends against the canonical failure cases

| Failure case                              | What in this architecture prevents it          |
|-------------------------------------------|------------------------------------------------|
| Air Canada — chatbot invented policy      | Strict grounding prompt + citation validation  |
| Chevrolet — $1 Tahoe prompt injection     | Input guardrail + bounded LLM authority (no commitment tools) |
| Replit — agent deleted prod DB            | No destructive tools in v1; HITL pattern documented |
| $47K runaway loop (Towards AI, Nov 2025)  | Max iterations cap + per-request token budget |
| Hallucinated citations                    | Verifier cross-checks citation IDs against retrieval |
| Prompt regression on model version change | Eval harness as CI gate; pinned prompt+model version |
| Klarna — silent quality degradation       | Eval harness runs continuously; feedback loop wires user signals back |

These are the talking points to be ready for. When the interviewer asks "what about hallucinations?" or "what about runaway costs?" — point at the relevant diagram node and the relevant code file.
