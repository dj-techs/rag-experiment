"""FastAPI gateway.

Single /query endpoint that runs check_input then invokes the orchestrator.
No auth, no rate limiting, no streaming in v1 — those slot into this layer
unchanged. The diagram's API Gateway node maps here.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.guardrails import check_input
from src.orchestrator import run as run_graph
from src.schemas import Citation

app = FastAPI(title="GenAI Demo Gateway", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    tenant_id: str = "demo"


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str
    intent: str
    iterations: int
    cost_usd: float
    verifier_passed: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    allow, reason = check_input(req.query)
    if not allow:
        raise HTTPException(status_code=400, detail=reason)

    state = run_graph(req.query, tenant_id=req.tenant_id)
    answer = state.get("answer")
    if not answer:
        raise HTTPException(status_code=500, detail="orchestrator returned no answer")
    cls = state.get("classification")
    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        intent=cls.intent if cls else "unknown",
        iterations=state.get("iteration", 0) or 0,
        cost_usd=round(state.get("cost_usd", 0.0) or 0.0, 6),
        verifier_passed=state.get("verifier_passed", False),
    )


# v2: POST /query/stream with text/event-stream for token-by-token UX
# v2: /admin/eval to trigger eval runs from the gateway
