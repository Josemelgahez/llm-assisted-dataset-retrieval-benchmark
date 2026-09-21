#!/usr/bin/env python3
"""Run the clean benchmark retrieval API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retrieval_backend.common import DEFAULT_QDRANT_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean retrieval API.")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    import uvicorn
    from retrieval_backend.api import app, create_app

    create_app(qdrant_url=args.qdrant_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
