"""Hybrid retrieval pipeline.

Dense (Chroma + sentence-transformers) + sparse (BM25) → Reciprocal Rank
Fusion → LLM-based rerank. Single public surface: hybrid_search() / rerank()
/ rewrite_query(). Module-level singletons keep model + index loaded across
calls within a process.
"""
from __future__ import annotations

import json
import pickle
import re

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingest import (
    BM25_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.llm import call_llm
from src.schemas import RetrievalResult


# -- Lazy singletons ---------------------------------------------------------

_embedder: SentenceTransformer | None = None
_collection = None
_bm25_bundle: dict | None = None


def _embedder_get() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _collection_get():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def _bm25_get() -> dict:
    global _bm25_bundle
    if _bm25_bundle is None:
        if not BM25_PATH.exists():
            raise RuntimeError(
                f"BM25 index missing at {BM25_PATH} — "
                f"run `uv run python -m src.ingest` first"
            )
        _bm25_bundle = pickle.loads(BM25_PATH.read_bytes())
    return _bm25_bundle


# -- Retrieval primitives ----------------------------------------------------

def _dense_search(
    query: str, top_k: int, tenant_id: str,
) -> list[tuple[str, float, str, dict]]:
    qvec = _embedder_get().encode(
        [query], normalize_embeddings=True,
    ).tolist()[0]
    res = _collection_get().query(
        query_embeddings=[qvec],
        n_results=top_k,
        where={"tenant_id": tenant_id},
        include=["documents", "metadatas", "distances"],
    )
    return [
        (cid, float(dist), doc, md)
        for cid, dist, doc, md in zip(
            res["ids"][0], res["distances"][0],
            res["documents"][0], res["metadatas"][0],
        )
    ]


def _bm25_search(
    query: str, top_k: int, tenant_id: str,
) -> list[tuple[str, float, str, dict]]:
    bundle = _bm25_get()
    bm25, ids, docs, metas = (
        bundle["bm25"], bundle["ids"], bundle["documents"], bundle["metadatas"],
    )
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    pairs = sorted(
        (
            (s, i) for i, s in enumerate(scores)
            if metas[i].get("tenant_id") == tenant_id
        ),
        reverse=True,
    )[:top_k]
    return [(ids[i], float(s), docs[i], metas[i]) for s, i in pairs]


def _rrf_fuse(
    dense: list[tuple[str, float, str, dict]],
    sparse: list[tuple[str, float, str, dict]],
    *, k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank_i). k=60 is the canonical default."""
    table: dict[str, dict] = {}
    for rank, (cid, _, doc, md) in enumerate(dense):
        slot = table.setdefault(cid, {"text": doc, "md": md, "score": 0.0})
        slot["score"] += 1.0 / (k + rank + 1)
    for rank, (cid, _, doc, md) in enumerate(sparse):
        slot = table.setdefault(cid, {"text": doc, "md": md, "score": 0.0})
        slot["score"] += 1.0 / (k + rank + 1)
    fused = sorted(table.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return [
        RetrievalResult(
            chunk_id=cid,
            source_id=v["md"].get("source_id", ""),
            text=v["text"],
            score=v["score"],
            metadata=v["md"],
        )
        for cid, v in fused
    ]


def hybrid_search(
    query: str, *, top_k: int = 20, tenant_id: str = "demo",
) -> list[RetrievalResult]:
    """Dense + BM25 + RRF. Returns up to top_k merged results."""
    dense = _dense_search(query, top_k, tenant_id)
    sparse = _bm25_search(query, top_k, tenant_id)
    return _rrf_fuse(dense, sparse)[:top_k]


# -- Rerank ------------------------------------------------------------------

_RERANK_SYSTEM = (
    "You are a search reranker. Given a query and numbered candidate passages, "
    "return ONLY a JSON object of the form {{\"top_indices\": [int, ...]}} "
    "listing the indices of the top {n} passages most relevant to the query, "
    "in order. Do not include any other text."
)


def rerank(
    query: str, candidates: list[RetrievalResult], *, top_n: int = 5,
) -> list[RetrievalResult]:
    """LLM-based rerank using the triage tier (cheap). Defensive parse — falls
    back to original order on any failure.

    # v2: cross-encoder rerank (Cohere/BGE) for production-grade quality
    """
    if not candidates:
        return []
    n = min(top_n, len(candidates))
    listing = "\n".join(
        f"[{i}] {c.text[:600].replace(chr(10), ' ')}"
        for i, c in enumerate(candidates)
    )
    user = f"QUERY: {query}\n\nCANDIDATES (return top {n} indices):\n{listing}"
    try:
        resp = call_llm(
            messages=[{"role": "user", "content": user}],
            system=_RERANK_SYSTEM.format(n=n),
            model_tier="triage",
            max_tokens=200,
        )
        m = re.search(r"\{[^{}]*\}", resp.text)
        if not m:
            return candidates[:n]
        data = json.loads(m.group(0))
        raw = data.get("top_indices", [])
        idxs = [int(i) for i in raw if isinstance(i, int) and 0 <= i < len(candidates)]
        out = [candidates[i] for i in idxs[:n]]
        seen = set(idxs)
        for i, c in enumerate(candidates):
            if i not in seen and len(out) < n:
                out.append(c)
        return out
    except Exception:
        return candidates[:n]


# -- Query rewriting (HyDE-lite) ---------------------------------------------

_HYDE_SYSTEM = (
    "You are a search query expander for SEC 10-K filings. Given a user "
    "question, write a short hypothetical answer (2-3 sentences) using "
    "terminology likely to appear in such a filing. Do NOT add disclaimers "
    "or note that the answer is hypothetical — just write it."
)


def rewrite_query(query: str) -> str:
    """Concatenate the query with a Haiku-generated hypothetical answer.

    # v2: full HyDE + sub-question split + acronym resolution
    """
    try:
        resp = call_llm(
            messages=[{"role": "user", "content": query}],
            system=_HYDE_SYSTEM,
            model_tier="triage",
            max_tokens=200,
        )
        expanded = resp.text.strip()
        return f"{query}\n\n{expanded}" if expanded else query
    except Exception:
        return query
