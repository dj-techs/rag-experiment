"""Centralized Pydantic v2 schemas + LangGraph state.

All structured outputs cross module boundaries through these models — never
pass raw dicts. Single source of truth for the contract between nodes.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    excerpt: str
    page: int | None = None


class GroundedAnswer(BaseModel):
    """Executor node output. Citations validated by the verifier against retrieval."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    reasoning: str = ""


class QueryClassification(BaseModel):
    """Router node output. Drives the conditional edge after classification."""

    intent: Literal["factual", "ambiguous", "out_of_scope"]
    requires_clarification: bool = False
    clarifying_question: str | None = None


class RetrievalResult(BaseModel):
    chunk_id: str
    source_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Envelope returned by every src/llm.py call. Used for tracing + cost rollup."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    latency_ms: int
    fallback_used: bool = False


class OrchestratorState(TypedDict, total=False):
    """LangGraph state. total=False so nodes can populate fields incrementally."""

    query: str
    tenant_id: str
    classification: QueryClassification | None
    retrieved: list[RetrievalResult]
    answer: GroundedAnswer | None
    iteration: int
    error: str | None
    verifier_feedback: str | None
    verifier_passed: bool
    cost_usd: float
