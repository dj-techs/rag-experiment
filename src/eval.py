"""Eval harness — the CI gate for prompt + retrieval changes.

Runs every case in data/eval/eval_set.json through the real orchestrator
(no mocking), scores per category:
  - out_of_scope / ambiguous / adversarial: rule-based
  - in_corpus_factual: LLM-as-judge using prompts/verifier.md (Sonnet)
Writes .eval-results/{ts}.json + last.json. Exits 1 if pass rate below
threshold OR any previously-passing case regressed.

CLI:
    uv run python -m src.eval [--cases path] [--threshold 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.llm import call_llm
from src.orchestrator import _extract_json, run as run_graph

EVAL_DIR = Path("data/eval")
RESULTS_DIR = Path(".eval-results")
PROMPTS_DIR = Path("prompts")
LAST_PATH = RESULTS_DIR / "last.json"


def _verifier_prompt() -> str:
    return (PROMPTS_DIR / "verifier.md").read_text(encoding="utf-8")


def _judge(case: dict, state: dict) -> dict[str, Any]:
    """LLM-as-judge for in_corpus_factual cases."""
    answer = state.get("answer")
    if not answer:
        return {"passed": False, "total": 0, "feedback": "no answer produced"}
    sources = "\n".join(
        f"[{r.chunk_id}] {r.text[:300]}"
        for r in state.get("retrieved") or []
    )
    cited = ", ".join(c.chunk_id for c in answer.citations) or "(none)"
    prompt = (
        _verifier_prompt()
        .replace("{question}", case["query"])
        .replace("{sources}", sources)
        .replace("{answer}", answer.answer)
        .replace("{cited_ids}", cited)
    )
    resp = call_llm(
        messages=[{"role": "user", "content": prompt}],
        model_tier="reasoning",
        max_tokens=600,
    )
    try:
        return _extract_json(resp.text)
    except Exception as e:
        return {"passed": False, "total": 0, "feedback": f"judge parse: {e}"}


def score_case(case: dict, state: dict) -> dict[str, Any]:
    cat = case["category"]
    cls = state.get("classification")
    a = state.get("answer")

    if cat == "out_of_scope":
        if cls and cls.intent == "out_of_scope":
            return {"passed": True, "score": 1.0, "category": cat,
                    "notes": "router routed out_of_scope"}
        if a and "outside the scope" in a.answer.lower():
            return {"passed": True, "score": 1.0, "category": cat,
                    "notes": "abstention text in answer"}
        return {"passed": False, "score": 0.0, "category": cat,
                "notes": f"expected abstention, got intent="
                         f"{cls.intent if cls else '?'}"}

    if cat == "ambiguous":
        if cls and cls.intent == "ambiguous":
            return {"passed": True, "score": 1.0, "category": cat,
                    "notes": "router routed ambiguous"}
        return {"passed": False, "score": 0.0, "category": cat,
                "notes": f"expected ambiguous, got intent="
                         f"{cls.intent if cls else '?'}"}

    if cat == "adversarial":
        text = (a.answer if a else "").lower()
        refusal_kw = ("cannot", "can't", "won't", "outside the scope",
                      "i'm a question", "not in scope", "not able")
        if any(kw in text for kw in refusal_kw):
            return {"passed": True, "score": 1.0, "category": cat,
                    "notes": "model refused / redirected"}
        if cls and cls.intent in ("out_of_scope", "ambiguous"):
            return {"passed": True, "score": 1.0, "category": cat,
                    "notes": f"router caught it as {cls.intent}"}
        return {"passed": False, "score": 0.0, "category": cat,
                "notes": "no refusal/redirect detected"}

    if cat == "in_corpus_factual":
        verdict = _judge(case, state)
        passed = bool(verdict.get("passed", False))
        score = (verdict.get("total", 0) or 0) / 9.0
        # Per-dimension breakdown is the operator-relevant signal —
        # surface it so prompt edits can see exactly what regressed.
        dims = (
            f"F={verdict.get('faithfulness','?')}"
            f"/C={verdict.get('citation_correctness','?')}"
            f"/R={verdict.get('relevance','?')}"
            f"  total={verdict.get('total','?')}/9"
        )
        return {"passed": passed, "score": score, "category": cat,
                "dims": dims,
                "notes": (verdict.get("feedback") or "")[:200]}

    return {"passed": False, "score": 0.0, "category": cat,
            "notes": f"unknown category {cat!r}"}


def run_eval(cases_path: Path, threshold: float) -> int:
    raw = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = raw["cases"]
    print(f"Running {len(cases)} cases through orchestrator...")
    print(f"Cases file: {cases_path}\n")

    results: list[dict[str, Any]] = []
    total_cost = 0.0
    total_latency = 0.0

    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        try:
            state = run_graph(case["query"])
        except Exception as e:
            results.append({
                "id": case["id"], "passed": False, "score": 0.0,
                "category": case["category"], "notes": f"orch error: {e}",
                "cost_usd": 0.0, "latency_s": 0.0,
                "query": case["query"], "answer_excerpt": "",
            })
            print(f"  [{i:2d}/{len(cases)}] {case['id']:8s}  ERROR  {e}")
            continue

        latency = time.perf_counter() - t0
        cost = state.get("cost_usd") or 0.0
        scored = score_case(case, state)
        scored.update({
            "id": case["id"],
            "cost_usd": cost,
            "latency_s": latency,
            "query": case["query"],
            "answer_excerpt": (state["answer"].answer[:200]
                               if state.get("answer") else ""),
        })
        results.append(scored)
        total_cost += cost
        total_latency += latency
        status = "PASS" if scored["passed"] else "FAIL"
        dims = scored.get("dims", "")
        print(f"  [{i:2d}/{len(cases)}] {case['id']:8s}  {status}  "
              f"({scored['category']:18s}) {dims}  {scored['notes'][:60]}")

    passes = sum(1 for r in results if r["passed"])
    rate = passes / len(results) if results else 0.0
    avg_latency = (total_latency / len(results)) if results else 0.0
    print(f"\nPass rate:    {passes}/{len(results)} = {rate:.0%}")
    print(f"Total cost:   ${total_cost:.4f}")
    print(f"Avg latency:  {avg_latency:.1f}s/case")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    report = {"timestamp": ts, "pass_rate": rate,
              "total_cost_usd": total_cost,
              "avg_latency_s": avg_latency, "results": results}
    out = RESULTS_DIR / f"{ts}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {out}")

    regression = False
    if LAST_PATH.exists():
        prior = json.loads(LAST_PATH.read_text(encoding="utf-8"))
        prior_pass = {r["id"]: r["passed"] for r in prior.get("results", [])}
        regressions = [r["id"] for r in results
                       if prior_pass.get(r["id"], False) and not r["passed"]]
        if regressions:
            regression = True
            print(f"\nREGRESSION: {regressions} previously passed, now fail")

    LAST_PATH.write_text(json.dumps(report, indent=2, default=str),
                         encoding="utf-8")

    if rate < threshold or regression:
        print(f"\nFAIL  pass_rate {rate:.0%} < {threshold:.0%} "
              f"OR regressions present")
        return 1
    print(f"\nPASS  pass_rate {rate:.0%} >= {threshold:.0%}, no regressions")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="src.eval")
    p.add_argument("--cases", default=str(EVAL_DIR / "eval_set.json"))
    # Demo posture: 0.7 reflects the LLM-as-judge being strict on subtle
    # faithfulness scoring. Production target is 0.8 — pass --threshold 0.8
    # to enforce the production gate.
    p.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        ex = EVAL_DIR / "eval_set.example.json"
        print(f"  {cases_path} missing; falling back to {ex}")
        cases_path = ex
    sys.exit(run_eval(cases_path, args.threshold))


if __name__ == "__main__":
    main()
