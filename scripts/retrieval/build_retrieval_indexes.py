#!/usr/bin/env python3
"""Build the clean benchmark retrieval indexes in local Qdrant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from retrieval_backend.common import DEFAULT_QDRANT_URL
from retrieval_backend.indexing import build_indexes


def parse_profiles(value: str) -> set[str]:
    profiles = {item.strip().lower() for item in value.split(",") if item.strip()}
    valid = {"bm25", "harrier", "qwen"}
    unknown = profiles - valid
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown profile(s): {', '.join(sorted(unknown))}")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the historical lexical, Harrier, and Qwen retrieval indexes."
    )
    parser.add_argument("--input", required=True, help="Prepared collection directory.")
    parser.add_argument(
        "--profiles",
        type=parse_profiles,
        default=parse_profiles("bm25,harrier,qwen"),
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dense-encoder-method",
        choices=("auto", "encode_document", "encode"),
        default="auto",
        help=(
            "Dense document encoder method. Default 'auto' preserves the "
            "historical behavior by using encode_document() when available. "
            "Use 'encode' only for controlled compatibility tests."
        ),
    )
    args = parser.parse_args()

    build_indexes(
        input_dir=Path(args.input).expanduser().resolve(),
        qdrant_url=args.qdrant_url,
        profiles=args.profiles,
        batch_size=args.batch_size,
        dense_encoder_method=args.dense_encoder_method,
    )


if __name__ == "__main__":
    main()
