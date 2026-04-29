"""LangGraph orchestrator: Router -> Planner -> Executor -> Verifier.

A deterministic state machine with conditional edges and a hard iteration
cap. Workflow over agent: every transition is explicit; no free-roaming
loops. The verifier can ask the executor to retry up to MAX_ITERATIONS
total attempts — anything beyond that returns whatever we have.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from src.guardrails import check_output
from src.llm import call_llm
from src.rag import hybrid_search, rerank, rewrite_query
from src.schemas import (
    GroundedAnswer,
    OrchestratorState,
    QueryClassification,
)


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
MAX_ITERATIONS = 3       # the $47K runaway-loop incident lesson
DEFAULT_TOP_K = 20       # candidate pool for the reranker
RERANK_TOP_N = 5         # context size for the executor (~7.5K chars)
EXECUTOR_MAX_TOKENS = 2500  # long compound queries need headroom to finish JSON

# v2: real MCP tool integration. Stubs only in v1.
TOOLS: dict[str, Any] = {
    "web_search": lambda q: {"error": "web_search is a v2 stub", "query": q},
    "calc": lambda expr: {"error": "calc is a v2 stub", "expr": expr},
}


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


# Loaded once at import. Prompts are versioned via git per CLAUDE.md.
PROMPT_CLASSIFY = _load_prompt("classify_query")
PROMPT_GROUNDED_QA = _load_prompt("grounded_qa")
PROMPT_VERIFIER = _load_prompt("verifier")


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerant of markdown
    code fences, surrounding prose, and extra braces in instructions echoed
    back (e.g. the schema example from the prompt). Uses raw_decode so we
    don't have to balance braces ourselves."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"no parseable JSON object in: {text[:300]!r}")


# -- Nodes -------------------------------------------------------------------

def router_node(state: OrchestratorState) -> dict[str, Any]:
    """Classify the query. Short-circuit on out-of-scope / ambiguous."""
    query = state["query"]
    prompt = PROMPT_CLASSIFY.replace("{query}", query)
    resp = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model_tier="triage",
        max_tokens=400,
    )
    try:
        classification = QueryClassification(**_extract_json(resp.text))
    except Exception:
        classification = QueryClassification(intent="factual")

    update: dict[str, Any] = {
        "classification": classification,
        "iteration": 0,
        "cost_usd": (state.get("cost_usd") or 0.0) + resp.cost_usd,
    }
    if classification.intent == "out_of_scope":
        update["answer"] = GroundedAnswer(
            answer=(
                "This question is outside the scope of the loaded SEC 10-K "
                "filings (Apple, Microsoft, NVIDIA most recent fiscal year). "
                "I can't provide an answer from the available sources."
            ),
            citations=[],
            confidence="low",
            reasoning="Out-of-scope per router classification.",
        )
    elif classification.intent == "ambiguous":
        update["answer"] = GroundedAnswer(
            answer=(
                classification.clarifying_question
                or "Your question could refer to multiple companies. Could you "
                   "clarify which one (Apple, Microsoft, or NVIDIA)?"
            ),
            citations=[],
            confidence="low",
            reasoning="Ambiguous per router classification.",
        )
    return update


def planner_node(state: OrchestratorState) -> dict[str, Any]:
    """Pass-through in v1. The retrieval depth knob is DEFAULT_TOP_K above.

    # v2: dynamic top_k by classification confidence, query length-based
    # tier escalation, sub-question decomposition planning, tool selection.
    """
    return {}


def executor_node(state: OrchestratorState) -> dict[str, Any]:
    query = state["query"]
    iteration = (state.get("iteration") or 0) + 1
    tenant_id = state.get("tenant_id", "demo")

    expanded = rewrite_query(query)
    candidates = hybrid_search(expanded, top_k=DEFAULT_TOP_K, tenant_id=tenant_id)
    top = rerank(query, candidates, top_n=RERANK_TOP_N)

    src_lines = []
    for r in top:
        page = r.metadata.get("page", "")
        page_str = f", page {page}" if page else ""
        src_lines.append(
            f"[{r.chunk_id}] (source: {r.source_id}{page_str})\n"
            f"{r.text[:1500]}"
        )
    sources_block = "\n\n---\n\n".join(src_lines)

    prompt = (
        PROMPT_GROUNDED_QA
        .replace("{sources}", sources_block)
        .replace("{query}", query)
    )
    feedback = state.get("verifier_feedback")
    if feedback:
        prompt += (
            f"\n\n## Verifier feedback (previous attempt failed)\n{feedback}\n"
        )

    resp = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model_tier="reasoning",
        max_tokens=EXECUTOR_MAX_TOKENS,
    )
    try:
        answer = GroundedAnswer(**_extract_json(resp.text))
    except Exception as e:
        answer = GroundedAnswer(
            answer="The system encountered a parse error. Please retry.",
            citations=[],
            confidence="low",
            reasoning=f"executor parse failure: {e}",
        )

    return {
        "retrieved": top,
        "answer": answer,
        "iteration": iteration,
        "cost_usd": (state.get("cost_usd") or 0.0) + resp.cost_usd,
        "verifier_feedback": None,
    }


def verifier_node(state: OrchestratorState) -> dict[str, Any]:
    """Run the output guardrail. Demote confidence on failure; queue a retry
    if iteration budget allows."""
    answer = state["answer"]
    retrieved = state.get("retrieved") or []
    passed, failures = check_output(answer, retrieved)
    update: dict[str, Any] = {"verifier_passed": passed}
    if not passed:
        update["verifier_feedback"] = (
            "Issues detected: " + "; ".join(failures)
            + ". Regenerate using only chunk_ids that appear in SOURCES."
        )
        if answer.confidence == "high":
            update["answer"] = answer.model_copy(update={"confidence": "medium"})
    return update


# -- Routing -----------------------------------------------------------------

def route_after_classification(state: OrchestratorState) -> str:
    intent = state["classification"].intent
    if intent in ("out_of_scope", "ambiguous"):
        return intent
    return "factual"


def route_after_verify(state: OrchestratorState) -> str:
    if state.get("verifier_passed"):
        return "done"
    if (state.get("iteration") or 0) >= MAX_ITERATIONS:
        return "done"
    return "retry"


# -- Graph -------------------------------------------------------------------

def build_graph():
    g = StateGraph(OrchestratorState)
    g.add_node("router", router_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("verifier", verifier_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        route_after_classification,
        {"out_of_scope": END, "ambiguous": END, "factual": "planner"},
    )
    g.add_edge("planner", "executor")
    g.add_edge("executor", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verify,
        {"retry": "executor", "done": END},
    )
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(query: str, tenant_id: str = "demo") -> OrchestratorState:
    """Convenience entry point. Returns the final state."""
    initial: OrchestratorState = {
        "query": query,
        "tenant_id": tenant_id,
        "iteration": 0,
        "cost_usd": 0.0,
        "verifier_passed": False,
    }
    return get_graph().invoke(initial)


if __name__ == "__main__":
    import sys
    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What were Apple's main revenue drivers in fiscal 2025?"
    )
    state = run(q)
    cls = state.get("classification")
    print("\n=== FINAL STATE ===")
    print(f"intent: {cls.intent if cls else 'n/a'}")
    print(f"iterations: {state.get('iteration')}")
    print(f"verifier_passed: {state.get('verifier_passed')}")
    print(f"cost: ${state.get('cost_usd', 0):.4f}")
    a = state.get("answer")
    if a:
        print(f"\nconfidence: {a.confidence}")
        print(f"answer: {a.answer[:600]}")
        print(f"citations: {len(a.citations)}")
        for c in a.citations[:3]:
            print(f"  - {c.chunk_id}: {c.excerpt[:80]}")
