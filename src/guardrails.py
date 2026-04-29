"""Input + output guardrails.

Input: regex-based PII + injection patterns at the gateway.
Output: schema validation + citation cross-check vs retrieval.

# v2: ML-based PII detection (Presidio etc.) + injection classifier
# v2: per-tenant policy rules (compliance, content moderation)
"""
from __future__ import annotations

import re

from src.schemas import GroundedAnswer, RetrievalResult


# Patterns that should never reach the model.
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                # US SSN
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),              # credit card-ish
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b"),   # API keys
    re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),             # GitHub PATs
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                 # AWS access key id
]

# Common prompt-injection carriers. Cheap and effective for v1.
_INJECTION_PATTERNS = [
    re.compile(
        r"\b(ignore|disregard|forget)\b.*\b(previous|prior|earlier|all)\b"
        r".*\b(instructions|prompt|rules)\b", re.I,
    ),
    re.compile(r"\breveal\b.*\b(system\s+prompt|instructions)\b", re.I),
    re.compile(
        r"\b(act|pretend|roleplay)\s+as\b.*\b(developer|admin|system)\b", re.I,
    ),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
]

# Long base64 blobs are a common smuggling vector.
_LONG_BASE64 = re.compile(r"\b[A-Za-z0-9+/]{200,}={0,2}\b")


def check_input(text: str) -> tuple[bool, str | None]:
    """Gateway-level input check. Returns (allow, reason)."""
    for pat in _PII_PATTERNS:
        if pat.search(text):
            return False, (
                "Input appears to contain sensitive data (PII or credential). "
                "Please remove and retry."
            )
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return False, (
                "Input matches a known prompt-injection pattern. "
                "Please rephrase your question."
            )
    if _LONG_BASE64.search(text):
        return False, (
            "Input contains a suspiciously long encoded blob. "
            "Please send plain text."
        )
    return True, None


def check_output(
    answer: GroundedAnswer, retrieved: list[RetrievalResult],
) -> tuple[bool, list[str]]:
    """Validate citations + confidence. Returns (passed, failure_reasons).

    In v1 the verifier surfaces failures as a visual flag in the UI rather
    than blocking — the architecture accommodates a hard block when the
    HITL queue is wired up.
    """
    failures: list[str] = []
    valid_ids = {r.chunk_id for r in retrieved}

    for cite in answer.citations:
        if cite.chunk_id not in valid_ids:
            failures.append(
                f"hallucinated citation: chunk_id {cite.chunk_id!r} "
                "not in retrieved set"
            )

    if answer.confidence not in {"high", "medium", "low"}:
        failures.append(f"invalid confidence value: {answer.confidence!r}")

    for pat in _PII_PATTERNS:
        if pat.search(answer.answer):
            failures.append("output contains potential PII; redact before returning")
            break

    return len(failures) == 0, failures
