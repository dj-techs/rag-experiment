# Kickoff Prompt

> **Instructions for DJ:** Open Claude Code in a fresh project folder that contains the `handoff_package/` directory. Paste everything below the line as your first message. Claude Code will take it from there.

---

I'm preparing for an AI consulting firm interview where I'll demo a GenAI/RAG system live. I need you to build a working, runnable showcase that maps to a specific reference architecture I've designed.

**Read these files first, in this order, before doing anything else:**

1. `handoff_package/CLAUDE.md` — project conventions, stack, principles
2. `handoff_package/ARCHITECTURE.md` — the system architecture and how each component maps to source files
3. `handoff_package/BUILD_PLAN.md` — the ordered task list you'll execute
4. `handoff_package/DEMO_SCRIPT.md` — what I'll show in the interview, so you know what must work end-to-end

After reading those four files, propose your build plan back to me in 5-8 bullets, ask any clarifying questions, then start building.

**Constraints (non-negotiable):**

- Use **Python 3.11+** with **uv** for dependency management
- Use **LangGraph** for the orchestrator (Router→Planner→Executor→Verifier nodes)
- Use **Chroma** as the local vector store (no managed vendors for the demo)
- Use **Anthropic** as the primary LLM provider, **OpenAI** as fallback in the wrapper
- Use **Streamlit** for the demo UI — minimal, single page, but show citations and confidence
- Real public data only — **SEC 10-K filings** as the default corpus (Apple, Microsoft, NVIDIA, three filings is enough for a demo)
- Every prompt lives in `prompts/` as `.md` files, loaded at runtime — never inline in code
- The eval harness is the CI gate — must run with `uv run python -m src.eval` and exit non-zero on regression
- Every irreversible action in the design (none in v1) requires HITL — even though we're not implementing destructive tools, the architecture must accommodate them

**Out of scope for v1 (note these in code comments as `# v2:`):**

- Subagent pool (single agent loop is fine)
- Multi-tenancy (single tenant, but tenant_id field plumbed through)
- Real MCP integration (tool layer exists but holds stub tools)
- Drift monitor (just an empty module with a docstring)
- Production observability (just structured stdout logs + per-request token tracking)
- Auth/rate limiting at the gateway (FastAPI without auth is fine for demo)
- Fresh data / web search (stubbed)
- Graph KB (omitted)

**What "done" looks like:**

1. I clone the repo and run `uv sync` cleanly
2. I set `ANTHROPIC_API_KEY` in `.env`
3. I run `uv run python -m src.ingest` and watch it download 3 SEC 10-Ks and index them (5-10 minutes)
4. I run `uv run streamlit run ui/app.py` and the demo loads at localhost:8501
5. I type "What were Apple's main revenue drivers in fiscal 2024?" and get a grounded answer with citations and a confidence score, streamed
6. I type a question outside the corpus and the system says it doesn't know
7. I run `uv run python -m src.eval` and see a pass/fail report with per-case breakdown

**Style of work:**

- Walk me through your plan before writing code
- After each major file, run a quick smoke test in the terminal to confirm it works
- Surface tradeoffs as you go ("I'm picking X over Y because Z; if we cared about W we'd revisit")
- If something doesn't work, narrate the debugging — don't silently move on
- Keep total code under ~800 lines across all `src/` files. This is a demo, not a product

Ready when you are. Read the four files and then propose the plan.
