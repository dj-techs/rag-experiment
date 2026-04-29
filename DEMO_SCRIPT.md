# Demo Script

What DJ will say and demonstrate during the live interview. This is the user journey the build must support.

## Setup (before the call)

- `architecture_diagram.png` open on a second monitor or in a browser tab
- Streamlit demo running at `localhost:8501`
- VS Code or Claude Code open on the project folder in another window
- Terminal ready with `uv run python -m src.eval` queued

## Opening (60 sec)

> "I built a working showcase of a production GenAI agent architecture. Let me walk you through it. Two things on screen — the architecture diagram, and the running demo. The diagram is the production reference; the demo is a deliberately scoped subset that exercises the request lifecycle, the eval gate, and the safety/cost principles. I'll demo it first, then we can dive into any layer you want."

Pull up the architecture diagram briefly. Don't dwell — this is the appetizer.

## Demo flow (5-7 minutes)

### 1. The grounded answer (the happy path)

Click the "What were Apple's main revenue drivers in fiscal 2024?" button.

> "I'm asking a question about a public document we've indexed — Apple's most recent 10-K filing. Watch the response stream in. Notice the inline citations [1] [2] — every factual claim is sourced. The confidence badge is green: high confidence. If I hover over a citation, I see the excerpt from the actual filing."

Hover over citations to show.

> "Behind the scenes, the orchestrator [point to diagram] classified this as a factual query, the planner picked retrieval depth, the executor ran hybrid retrieval — dense plus BM25 — reranked the top 5 with an LLM, generated the answer with a strict-grounding prompt, and the verifier confirmed every cited chunk was actually retrieved. That last step is the citation-grounding check — it's how I prevent hallucinated citations like the lawyer in Mata v. Avianca."

### 2. The abstention (the out-of-scope path)

Click "What was the weather in Tokyo last Tuesday?"

> "Now I'll ask something the system can't possibly know — there's no weather data in the corpus. Notice it doesn't try to answer. It says 'this is outside the scope of the loaded documents' and stops. That's the abstention path — explicit refusal is a feature, not a bug. Compare this to the Air Canada chatbot that invented a refund policy and lost a court case in early 2024."

### 3. The ambiguity (clarification path)

Click "Tell me about their products"

> "Ambiguous question — whose products? The system doesn't guess. It asks me to clarify which company. That's the router catching ambiguity at the front of the pipeline."

### 4. The eval harness (the senior signal)

Switch to terminal. Run:

```bash
uv run python -m src.eval
```

> "Now the part most demos skip — the eval harness. Same pipeline, run against a curated set of cases including in-corpus questions, out-of-scope questions, ambiguous questions, and adversarial inputs. Each one is scored with an LLM-as-judge against a rubric. The runner exits non-zero on regression. This is what I'd wire into CI — prompt changes don't deploy unless they pass."

Show the output. Point at the per-case scores.

> "If I edited any prompt right now and dropped quality, I'd see it here. That's the eval-as-CI-gate principle — same as you'd never ship code that fails tests, you don't ship a prompt that regresses on quality."

### 5. The architecture walk (the deep dive)

Switch back to the architecture diagram. Now talk through where each component sits in the codebase.

> "Quick file mapping. The orchestrator is `src/orchestrator.py` — it's a LangGraph state machine with four nodes: Router, Planner, Executor, Verifier. The retrieval is `src/rag.py` — hybrid dense plus BM25 with reciprocal rank fusion, then a rerank pass. The guardrails are `src/guardrails.py` — input PII and prompt injection checks at the gateway, output citation validation at the response. The LLM provider wrapper is `src/llm.py` — it tracks tokens, cost, latency on every call, enforces a per-request token budget, and falls back from Anthropic to OpenAI. Every prompt is in `prompts/` as a separate markdown file — versioned in git, gated by the eval harness on change."

Open `src/orchestrator.py` in your editor. Show the graph definition.

> "This is what makes it debuggable — explicit state machine, conditional edges, max-iteration cap of 3. If something goes wrong I can see exactly which node failed and replay from there."

## Common interviewer questions and where to point

| Question                              | Point to                              | Talking point |
|---------------------------------------|---------------------------------------|---------------|
| "How does this prevent hallucinations?" | Output Guardrail (red) on diagram + `src/guardrails.py::check_output` | Strict grounding prompt + citation cross-validation. Verifier rejects answers with citation IDs not in retrieval. |
| "How does it scale?"                  | The whole diagram | "At 100 users, this runs as-is. At 10K, the cache becomes critical and we move from pgvector to Pinecone. At 100K, we shard the vector store, route queries to read replicas, and the HITL queue becomes its own service. The architecture survives because responsibilities are factored cleanly." |
| "Why LangGraph?"                      | Orchestration lane | "Cycles, checkpointing, HITL primitives. For a deterministic linear workflow I'd skip it; for a state machine with retries and conditional routing, it earns its weight." |
| "Why these models?"                   | LLM Provider node (3-tier) | "Cost-aware routing. Triage on Haiku, synthesis on Sonnet, hard reasoning on Opus. 60-80% cost reduction in production vs flagship-only." |
| "What about cost?"                    | Budget Enforcer (yellow) | "Per-request token cap, per-tenant daily limits in production. The $47K runaway-loop incident from late 2025 is why this isn't optional." |
| "What's missing for production?"      | The diagram nodes I marked v2 | List them: semantic cache, drift monitor, real HITL queue, multi-tenancy enforcement, real auth, observability beyond stdout. |
| "Show me the code for [X]"            | The file mapping in `ARCHITECTURE.md` | Open it, walk through. |
| "How would this change for healthcare?"| The whole diagram | Tighter HITL by default, clinical reviewer in the eval set, on-prem deployment, output guardrail with explicit policy rules. Same shape, different parameters. |

## Recovery moves if something breaks

- **The demo throws an error**: Don't panic. Say "let me see what went wrong" — open the terminal logs, narrate the debugging. The interviewers explicitly said reflection-on-failure is what they're scoring. A graceful recovery is worth more than a flawless demo.
- **A query returns garbage**: Treat it as a teaching moment. "Interesting — let's see why. The retrieval returned [these chunks], the model said [this]. Looks like the chunking lost context here. In production I'd add a structure-aware chunker — that's exactly the kind of thing the eval harness would surface."
- **The Streamlit UI hangs**: Switch to running queries via `curl localhost:8000/query` directly. The architecture is the message, not the UI.
- **Anthropic API is down**: The wrapper falls back to OpenAI. Mention it: "the provider fallback just kicked in — that's the multi-provider design point in action."

## Closing (30 sec)

> "That's the showcase. Two things to call out: this is intentionally minimal — the production reference architecture has more layers I scoped out for the demo, marked clearly in the codebase as v2 work. And the spine of it — the orchestrator graph, the eval harness, the citation-grounded prompt, the budget caps — that's what I'd build first for any client engagement. Everything else stages in based on the client's risk profile and traffic shape."

> "Happy to dive into any specific layer."

## What this demo proves

1. **Engineering speed** — built and runnable end-to-end with public data
2. **Architecture sense** — every file maps to a deliberate design decision
3. **Production instincts** — eval harness, citation validation, budget caps, structured logging
4. **Client framing** — every choice has a business reason on hand
5. **Self-awareness** — explicit about what's v1 vs v2, can defend tradeoffs
