# Conversation Summary — Reference Context

This is a condensed summary of the planning conversation that produced this handoff package. Claude Code can reference it for additional context but should NOT need to read it to execute the build — the actionable instructions are in `KICKOFF_PROMPT.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `BUILD_PLAN.md`, and `DEMO_SCRIPT.md`.

## Who is DJ and what's the situation

DJ (James Travis McHorse) is a Senior Full-Stack Engineer at Left Coast Stack LLC, currently on a Toptal contract for a voter-engagement CRM (Threshold). Background: 12+ years engineering, recent specialization in AI/ML, LLM integrations, RAG systems, agentic workflows.

**The interview**: 2-hour live coding session for an AI consulting firm hiring GenAI Engineers for greenfield product development. Role split: 60% engineering, 20% research, 20% architecture & strategy. Contract-to-hire.

**Interviewer guidance DJ received**:
- Use industry-accepted tools (Claude Code, Codex, etc.) — no fully custom builds
- Balance technical thinking with business impact
- Be prepared to explain reasoning; not everything has to work perfectly
- Depth matters more than name-dropping
- Reflect as you go — self-awareness is being scored
- Communication counts — collaborative, client-facing environment

## What we built together (across multiple turns)

1. **GenAI_Interview_Prep_Guide.docx** — comprehensive prep guide: reading the role, prompt engineering examples (good/bad with real cases), agentic patterns, 5 showcase scenarios, reasoning frameworks, behavioral story bank, live failure cases (Air Canada, Klarna, Replit, $47K incident, etc.)

2. **Companion_Guide_Frameworks_CLAUDEmd_MCP.docx** — frameworks comparison (LangGraph, LlamaIndex, Anthropic SDK, OpenAI Agents SDK), CLAUDE.md templates from real GitHub repos, 8 production slash commands, 8 essential MCP servers with config

3. **Architecture_Walkthrough_Mock_Interview.docx + architecture_diagram.png** — production GenAI architecture diagram (6 swim lanes), node-by-node walkthrough, tradeoff analysis, simulated 8-question mock interview with model answers, failure scenario STAR stories, scaling discussion (100 → 100K users), industry-specific variants

4. **This handoff package** — instructs Claude Code to build a working showcase that demonstrates the architecture with live SEC filing data

## The three architectural principles (must come through in code)

1. **Every irreversible action is gated** — v1 has no destructive tools, but the architecture accommodates them
2. **Every quality decision is measurable** — eval harness as CI gate
3. **Every unit of cost has a circuit breaker** — token budgets, max iterations, model routing

## Real-world failure cases the architecture defends against

These come up in the interview. The code should make the connection clear (via comments or naming).

- **Air Canada (Feb 2024)** — chatbot invented refund policy → defended by strict-grounding prompt + citation validation
- **Chevrolet $1 Tahoe (Dec 2023)** — prompt injection, LLM "agreed" to absurd terms → defended by input guardrail + bounded LLM authority (no commitment tools)
- **Replit DB deletion (July 2025)** — agent destroyed prod DB → defended by HITL gate on destructive actions (architectural pattern, not implemented in v1)
- **$47K runaway loop (Towards AI, Nov 2025)** — multi-agent system ran unbounded → defended by max iteration cap + token budget
- **Mata v. Avianca (2023)** — lawyer's brief had fabricated case citations → defended by citation cross-validation in verifier
- **Klarna AI reversal (May 2025)** — silent quality degradation → defended by eval harness + feedback loop
- **Stanford GPT-4 regression (2023)** — same prompt, same task, model drift → defended by pinned model+prompt versions in eval harness

## Communication style (matches DJ's preferences)

- Async-first, structured, section-broken with clear ownership
- Direct and professional without filler language
- Tone calibrated by context (formal for client-facing docs, casual for internal Slack)
- Always surfaces tradeoffs — quality vs cost vs latency
- Self-corrects in the open: "my assumption was X, that turned out wrong because Y"
- Concrete examples over abstractions
- Specific numbers over qualitative claims when available

## Stack preferences and reasoning

- **Python over Node.js for the demo** — RAG ecosystem maturity, LangGraph quality
- **uv over pip/poetry/pipenv** — speed, modern, DJ already uses it
- **Streamlit over a custom React UI** — minimal time investment for maximum demo signal
- **Chroma over Pinecone/Qdrant** — file-backed, no server, perfect for a 2hr demo
- **sentence-transformers over OpenAI embeddings** — local, no API cost during the demo, no API key dependency
- **LangGraph over raw SDK loop** — the orchestrator graph is the visual artifact that matches the diagram; if we just had a function, the architecture mapping would be weaker
- **Pydantic v2 everywhere** — schema enforcement is a senior signal; never pass dicts across boundaries
- **SEC 10-K filings over arXiv/government docs** — financial Q&A reads as professional and is universally recognizable

## What NOT to do

- No frameworks or tools we can't justify with a specific reason
- No "agentify everything" — the orchestrator is a deterministic graph with conditional edges, not a free-roaming agent
- No mock data — public SEC filings or nothing
- No fancy UI engineering — Streamlit, focused, single page
- No security theater — guardrails are real (regex-based PII, schema validation, citation validation), not aspirational
- No silent prompt edits — eval harness is a gate, not a suggestion
- No code over ~800 lines total — this is a demo, not a product
