"""Provider-agnostic LLM wrapper.

The single chokepoint for every model call. Enforces:
  - Per-request token budget (input + output) — circuit breaker
  - Anthropic primary, OpenAI fallback on auth/network/5xx
  - Structured one-line JSON logs to stderr (model, tokens, cost, latency)
  - Three-tier model routing: triage / reasoning / hard

All other modules MUST go through call_llm() / stream_llm() — never import
the provider SDKs directly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv

from src.schemas import LLMResponse

load_dotenv(override=True)  # uv pre-injects empty values; we win.


# -- Configuration -----------------------------------------------------------

# Conservative caps that catch runaway prompts before they hit the provider.
# The $47K runaway-loop incident is the lesson — every call has a ceiling.
TOKEN_BUDGET = {"input": 16_000, "output": 4_000}

# Cost-aware tiering: triage on Haiku, synthesis on Sonnet. Opus reserved
# (CLAUDE.md: "no Opus calls in v1").
MODEL_TIERS = {
    "triage":    "claude-haiku-4-5",
    "reasoning": "claude-sonnet-4-6",
    "hard":      "claude-opus-4-7",  # v2: not used in v1
}

OPENAI_FALLBACK = {
    "triage":    "gpt-4o-mini",
    "reasoning": "gpt-4o",
    "hard":      "gpt-4o",
}

# Per-million-token pricing (USD). Pinned snapshot — refresh manually.
# Source: claude-api skill, cached 2026-04-15. OpenAI: docs as of 2026-04.
COSTS = {
    "claude-haiku-4-5":  {"input": 1.00, "output":  5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00},
    "gpt-4o-mini":       {"input": 0.15, "output":  0.60},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
}


class BudgetExceededError(RuntimeError):
    """Raised when a request would breach TOKEN_BUDGET. Catch at the gateway."""


# -- Lazy clients ------------------------------------------------------------

_anthropic_client: anthropic.Anthropic | None = None
_openai_client: openai.OpenAI | None = None


def _anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _openai() -> openai.OpenAI | None:
    global _openai_client
    if not os.getenv("OPENAI_API_KEY"):
        return None
    if _openai_client is None:
        _openai_client = openai.OpenAI()
    return _openai_client


# -- Helpers -----------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """4-chars-per-token heuristic. Fine for budget gating; not for billing."""
    return max(1, len(text) // 4)


def _enforce_budget(
    messages: list[dict], system: str | None, max_tokens: int
) -> None:
    parts = [system or ""]
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
    est_input = _estimate_tokens("\n".join(parts))
    if est_input > TOKEN_BUDGET["input"]:
        raise BudgetExceededError(
            f"input ~{est_input} tokens exceeds budget {TOKEN_BUDGET['input']}"
        )
    if max_tokens > TOKEN_BUDGET["output"]:
        raise BudgetExceededError(
            f"max_tokens {max_tokens} exceeds budget {TOKEN_BUDGET['output']}"
        )


def _cost_usd(model: str, in_toks: int, out_toks: int) -> float:
    rate = COSTS.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (in_toks / 1_000_000) * rate["input"]
        + (out_toks / 1_000_000) * rate["output"],
        6,
    )


def _log_call(payload: dict[str, Any]) -> None:
    """Structured one-liner. Goes to stderr so demo stdout stays clean."""
    print(json.dumps({"event": "llm_call", **payload}), file=sys.stderr, flush=True)


# -- Public API --------------------------------------------------------------

def call_llm(
    messages: list[dict],
    *,
    model_tier: str = "reasoning",
    max_tokens: int = 4096,
    system: str | None = None,
    response_format: type | None = None,  # informational; callers parse JSON themselves
) -> LLMResponse:
    """Call the LLM with provider fallback. Returns LLMResponse."""
    _enforce_budget(messages, system, max_tokens)
    model = MODEL_TIERS[model_tier]
    start = time.perf_counter()

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            # Auto-cache the last cacheable block. No-op if prefix < model min.
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            kwargs["system"] = system
        resp = _anthropic().messages.create(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        in_toks = resp.usage.input_tokens
        out_toks = resp.usage.output_tokens
        cost = _cost_usd(model, in_toks, out_toks)
        result = LLMResponse(
            text=text, input_tokens=in_toks, output_tokens=out_toks,
            cost_usd=cost, model=model, latency_ms=latency_ms,
        )
        _log_call({
            "provider": "anthropic", "model": model, "tier": model_tier,
            "in": in_toks, "out": out_toks, "cost_usd": cost,
            "latency_ms": latency_ms,
        })
        return result
    except (
        anthropic.AuthenticationError,
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
    ) as e:
        _log_call({"provider": "anthropic", "model": model,
                   "fallback_reason": type(e).__name__})
        return _openai_call(
            messages, model_tier=model_tier, max_tokens=max_tokens,
            system=system, started_at=start,
        )


def _openai_call(
    messages: list[dict],
    *,
    model_tier: str,
    max_tokens: int,
    system: str | None,
    started_at: float,
) -> LLMResponse:
    client = _openai()
    if client is None:
        raise RuntimeError(
            "Anthropic call failed and OPENAI_API_KEY is not set; "
            "no fallback available"
        )
    model = OPENAI_FALLBACK[model_tier]
    oai_messages: list[dict] = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens, messages=oai_messages,
    )
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    text = resp.choices[0].message.content or ""
    in_toks = resp.usage.prompt_tokens
    out_toks = resp.usage.completion_tokens
    cost = _cost_usd(model, in_toks, out_toks)
    result = LLMResponse(
        text=text, input_tokens=in_toks, output_tokens=out_toks,
        cost_usd=cost, model=model, latency_ms=latency_ms,
        fallback_used=True,
    )
    _log_call({
        "provider": "openai_fallback", "model": model, "tier": model_tier,
        "in": in_toks, "out": out_toks, "cost_usd": cost,
        "latency_ms": latency_ms,
    })
    return result


def stream_llm(
    messages: list[dict],
    *,
    model_tier: str = "reasoning",
    max_tokens: int = 4096,
    system: str | None = None,
) -> Iterator[str]:
    """Yield text deltas from Anthropic. Falls back to one-shot OpenAI on error."""
    _enforce_budget(messages, system, max_tokens)
    model = MODEL_TIERS[model_tier]
    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens, "messages": messages,
    }
    if system:
        kwargs["system"] = system
    try:
        with _anthropic().messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                yield delta
            final = stream.get_final_message()
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log_call({
            "provider": "anthropic_stream", "model": model, "tier": model_tier,
            "in": final.usage.input_tokens, "out": final.usage.output_tokens,
            "cost_usd": _cost_usd(model, final.usage.input_tokens,
                                  final.usage.output_tokens),
            "latency_ms": latency_ms,
        })
    except (
        anthropic.AuthenticationError,
        anthropic.APIConnectionError,
        anthropic.APIStatusError,
    ) as e:
        _log_call({"provider": "anthropic_stream",
                   "fallback_reason": type(e).__name__})
        result = _openai_call(
            messages, model_tier=model_tier, max_tokens=max_tokens,
            system=system, started_at=start,
        )
        yield result.text


if __name__ == "__main__":
    # Phase 1 smoke: 1 cheap Haiku call, print the envelope.
    r = call_llm(
        [{"role": "user", "content": "Reply with exactly the two letters: OK"}],
        model_tier="triage",
        max_tokens=10,
    )
    print(f"text={r.text!r}  in={r.input_tokens} out={r.output_tokens}  "
          f"cost=${r.cost_usd}  latency={r.latency_ms}ms  model={r.model}")
