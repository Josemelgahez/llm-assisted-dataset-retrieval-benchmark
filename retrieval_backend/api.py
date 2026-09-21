"""FastAPI entry point for the minimal retrieval backend."""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI, Query

from .common import DEFAULT_QDRANT_URL
from .search import RetrievalSearcher, ensure_clean_schema

app = FastAPI(title="datos.gob.es benchmark retrieval backend")
searcher: RetrievalSearcher | None = None


def create_app(qdrant_url: str = DEFAULT_QDRANT_URL) -> FastAPI:
    """Create the minimal API using a local unauthenticated Qdrant URL."""
    global searcher
    searcher = RetrievalSearcher(qdrant_url=qdrant_url)
    return app


def _get_searcher() -> RetrievalSearcher:
    if searcher is None:
        create_app()
    assert searcher is not None
    return searcher


def _search_bm25(q: str):
    hits = _get_searcher().search_lexical(q)
    return {"hits": ensure_clean_schema(hits)}


@app.get("/search/bm25")
def search_bm25(q: str = Query(...)):
    return _search_bm25(q)


@app.get("/search/lexical")
def search_lexical(q: str = Query(...)):
    return _search_bm25(q)


@app.get("/search/harrier/semantic")
def search_harrier(q: str = Query(...)):
    hits = _get_searcher().search_semantic("harrier", q)
    return {"hits": ensure_clean_schema(hits)}


@app.get("/search/qwen/semantic")
def search_qwen(q: str = Query(...)):
    hits = _get_searcher().search_semantic("qwen", q)
    return {"hits": ensure_clean_schema(hits)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean retrieval API.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    create_app(qdrant_url=args.qdrant_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

