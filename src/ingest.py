"""SEC 10-K ingestion pipeline.

Download the most recent 10-K filing for each requested company from SEC
EDGAR, parse the HTML primary doc into text, chunk, embed locally, and
upsert into a persistent Chroma collection. Persists a BM25 index alongside
for the hybrid retrieval pipeline (Chroma doesn't ship sparse retrieval).

CLI:
    uv run python -m src.ingest [--companies AAPL,MSFT,NVDA]
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import time
from collections.abc import Iterable
from pathlib import Path

import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)


# -- Config ------------------------------------------------------------------

# CIKs are stable, public. Padded to 10 digits for the SEC submissions API.
CIKS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"  # holds HTML in v1; named pdfs for compat
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25.pkl"

COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE_CHARS = 4000     # ~1000 tokens at 4 chars/token
CHUNK_OVERLAP_CHARS = 800   # ~200 tokens

SEC_RATE_LIMIT_SLEEP = 0.15  # SEC asks for <10 req/s


# -- SEC EDGAR helpers --------------------------------------------------------

def _ua_headers() -> dict[str, str]:
    ua = os.getenv("USER_AGENT") or ""
    if not ua or "@" not in ua:
        raise RuntimeError(
            "USER_AGENT env var required by SEC EDGAR "
            "(format 'Name email@example.com')."
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def find_latest_10k(cik_padded: str) -> dict:
    """Walk the recent-filings index for the latest 10-K. Returns metadata."""
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=_ua_headers(), timeout=30)
    r.raise_for_status()
    time.sleep(SEC_RATE_LIMIT_SLEEP)
    recent = r.json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
    raise RuntimeError(f"no 10-K found for CIK {cik_padded}")


def download_10k_html(cik_padded: str, ticker: str, dest: Path) -> dict:
    meta = find_latest_10k(cik_padded)
    accession_clean = meta["accession"].replace("-", "")
    cik_int = str(int(cik_padded))
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_clean}/{meta['primary_doc']}"
    )
    print(f"  downloading {ticker} 10-K: {meta['accession']} "
          f"({meta['filing_date']})")
    r = requests.get(url, headers=_ua_headers(), timeout=60)
    r.raise_for_status()
    time.sleep(SEC_RATE_LIMIT_SLEEP)
    dest.write_bytes(r.content)
    return {**meta, "url": url, "size_bytes": len(r.content)}


# -- Parse & chunk ------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(path: Path) -> str:
    """Aggressive HTML strip. SEC filings are full of inline XBRL noise."""
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, *, size: int = CHUNK_SIZE_CHARS,
               overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Sliding-window chunker on character count.

    # v2: structure-aware chunking on 10-K Items / sections preserves more
    # semantic boundaries than fixed-size windows.
    """
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


# -- Pipeline -----------------------------------------------------------------

def ingest(companies: Iterable[str]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model (first run downloads ~80MB)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for ticker in companies:
        cik = CIKS.get(ticker.upper())
        if cik is None:
            print(f"  skipping unknown ticker {ticker}")
            continue
        print(f"\n[{ticker}]")
        html_path = PDF_DIR / f"{ticker.lower()}_10k.html"
        if html_path.exists():
            print(f"  cached: {html_path.name}")
            meta = {"accession": "cached", "filing_date": "cached"}
        else:
            meta = download_10k_html(cik, ticker, html_path)

        text = html_to_text(html_path)
        chunks = chunk_text(text)
        print(f"  parsed {len(text):,} chars -> {len(chunks)} chunks")

        source_id = html_path.name
        company = ticker.upper()
        ids = [f"{source_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source_id": source_id,
                "company": company,
                "chunk_index": i,
                "tenant_id": "demo",
                "filing_accession": meta.get("accession", ""),
                "filing_date": meta.get("filing_date", ""),
            }
            for i in range(len(chunks))
        ]

        print(f"  embedding {len(chunks)} chunks...")
        embeddings = embedder.encode(
            chunks, show_progress_bar=False, normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=chunks, metadatas=metadatas,
        )

    # Rebuild BM25 from the full collection so it stays in sync across runs.
    print("\nRebuilding BM25 index from full collection...")
    full = collection.get(include=["documents", "metadatas"])
    docs: list[str] = full["documents"]
    bm25 = BM25Okapi([d.split() for d in docs])
    BM25_PATH.write_bytes(pickle.dumps({
        "bm25": bm25,
        "ids": full["ids"],
        "documents": docs,
        "metadatas": full["metadatas"],
    }))
    print(f"  BM25 index -> {BM25_PATH} "
          f"({BM25_PATH.stat().st_size // 1024} KB, {len(docs)} docs)")

    total = collection.count()
    print(f"\nDone. collection={COLLECTION_NAME}  chunks={total}  "
          f"chroma={CHROMA_DIR}  bm25={BM25_PATH}")
    print("Estimated cost: $0.00 (local embeddings + free SEC data).")


def main() -> None:
    p = argparse.ArgumentParser(prog="src.ingest")
    p.add_argument(
        "--companies", default="AAPL,MSFT,NVDA",
        help="comma-separated tickers (default: AAPL,MSFT,NVDA)",
    )
    args = p.parse_args()
    tickers = [t.strip().upper() for t in args.companies.split(",") if t.strip()]
    ingest(tickers)


if __name__ == "__main__":
    main()
