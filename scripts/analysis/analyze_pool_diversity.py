#!/usr/bin/env python3
"""Analyze candidate-pool diversity and retrieval-profile contribution.

This script reads the Stage 2 candidate pool and computes retrieved-pair
coverage, exclusive contribution, and pairwise Jaccard similarity for the
retrieval profiles. It does not modify candidate-pool records.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPOSITORY_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (
    extract_query_number,
    list_query_json_files,
    load_json_object,
    make_pair_key,
    parse_dataset_id,
    require_positive_int,
    write_csv_atomic,
)


PROFILE_LABELS = {
    "bm25": "BM25",
    "harrier": "Harrier",
    "qwen": "Qwen",
}

PROFILE_ORDER = (
    "bm25",
    "harrier",
    "qwen",
)

PROFILE_ALIASES = {
    "lexical": "bm25",
}


def canonical_profile_name(profile: str) -> str:
    return PROFILE_ALIASES.get(profile, profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute candidate-pool diversity metrics from Stage 2 "
            "candidate-pool JSON files."
        )
    )

    parser.add_argument("--candidate-pool", type=Path, default=Path("data/candidate_pool"), help="Directory containing query_*.json candidate-pool files.")
    parser.add_argument("--output", type=Path, default=Path("results/pool_diversity.csv"), help="Output CSV path.")

    return parser.parse_args()


def validate_rank(
    pair_key: str,
    profile: str,
    profile_data: Any,
) -> int:
    if not isinstance(profile_data, dict):
        raise ValueError(
            f"Invalid retrieval entry for {pair_key} / {profile}"
        )

    rank = profile_data.get("rank")

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(
            f"Invalid rank for {pair_key} / {profile}: {rank!r}"
        )

    return rank


def load_profile_sets(
    candidate_pool: Path,
) -> tuple[dict[str, set[str]], int, int]:
    if not candidate_pool.is_dir():
        raise FileNotFoundError(
            f"Candidate-pool directory does not exist: {candidate_pool}"
        )

    query_files = list_query_json_files(candidate_pool)
    profile_sets = {
        profile: set()
        for profile in PROFILE_ORDER
    }
    all_pairs: set[str] = set()
    profile_set = set(PROFILE_ORDER)

    for path in query_files:
        payload = load_json_object(path)

        for required_key in ("id_query", "query", "hits"):
            if required_key not in payload:
                raise ValueError(
                    f"{path}: missing {required_key!r}."
                )

        filename_query_id = extract_query_number(path.name)
        query_id = require_positive_int(
            payload["id_query"],
            f"{path}: id_query",
        )

        if query_id != filename_query_id:
            raise ValueError(
                f"{path}: filename ID {filename_query_id} "
                f"!= id_query {query_id}."
            )

        query = payload["query"]

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"{path}: invalid query text {query!r}."
            )

        hits = payload["hits"]

        if not isinstance(hits, list):
            raise ValueError(
                f"'hits' must be a list in {path}"
            )

        pairs_in_query: set[str] = set()
        ranks_by_profile: dict[str, list[int]] = defaultdict(list)

        for hit_index, hit in enumerate(hits, start=1):
            context = f"{path}, hit {hit_index}"

            if not isinstance(hit, dict):
                raise ValueError(
                    f"{context}: hit must be an object."
                )

            if "id" not in hit:
                raise ValueError(
                    f"{context}: missing dataset ID."
                )

            dataset_id = parse_dataset_id(
                hit["id"],
                context,
            )

            pair_key = make_pair_key(
                query_id,
                dataset_id,
            )

            if pair_key in pairs_in_query:
                raise ValueError(
                    f"Duplicate pair inside {path}: {pair_key}"
                )

            if pair_key in all_pairs:
                raise ValueError(
                    f"Duplicate query-dataset pair: {pair_key}"
                )

            pairs_in_query.add(pair_key)
            all_pairs.add(pair_key)

            retrieval = hit.get("retrieval")

            if not isinstance(retrieval, dict) or not retrieval:
                raise ValueError(
                    f"Invalid or empty retrieval provenance for {pair_key}"
                )

            canonical_retrieval = {
                canonical_profile_name(profile): profile_data
                for profile, profile_data in retrieval.items()
            }

            if len(canonical_retrieval) != len(retrieval):
                raise ValueError(
                    f"Duplicate retrieval profiles after aliasing for {pair_key}"
                )

            unknown_profiles = set(canonical_retrieval) - profile_set

            if unknown_profiles:
                raise ValueError(
                    f"Unknown retrieval profiles for {pair_key}: "
                    f"{sorted(unknown_profiles)}"
                )

            for profile, profile_data in canonical_retrieval.items():
                rank = validate_rank(
                    pair_key,
                    profile,
                    profile_data,
                )

                ranks_by_profile[profile].append(rank)
                profile_sets[profile].add(pair_key)

        for profile, ranks in ranks_by_profile.items():
            duplicate_ranks = [
                rank
                for rank, count in Counter(ranks).items()
                if count > 1
            ]

            if duplicate_ranks:
                raise ValueError(
                    f"Duplicate ranks in {path} for {profile}: "
                    f"{duplicate_ranks[:10]}"
                )

    return (
        profile_sets,
        len(query_files),
        len(all_pairs),
    )


def jaccard(
    left: set[str],
    right: set[str],
) -> float | None:
    union = left | right

    if not union:
        return None

    return len(left & right) / float(len(union))


def compute_rows(
    profile_sets: dict[str, set[str]],
) -> list[dict[str, Any]]:
    jaccards: dict[tuple[str, str], float | None] = {}

    for left, right in combinations(PROFILE_ORDER, 2):
        similarity = jaccard(
            profile_sets[left],
            profile_sets[right],
        )

        jaccards[(left, right)] = similarity
        jaccards[(right, left)] = similarity

    for profile in PROFILE_ORDER:
        jaccards[(profile, profile)] = (
            1.0
            if profile_sets[profile]
            else None
        )

    rows = []

    for profile in PROFILE_ORDER:
        others = set().union(
            *[
                profile_sets[other]
                for other in PROFILE_ORDER
                if other != profile
            ]
        )

        retrieved_pairs = len(profile_sets[profile])
        exclusive_pairs = len(
            profile_sets[profile] - others
        )

        exclusive_rate = (
            exclusive_pairs / float(retrieved_pairs)
            if retrieved_pairs
            else None
        )

        rows.append(
            {
                "profile": PROFILE_LABELS[profile],
                "retrieved_pairs": retrieved_pairs,
                "exclusive_pairs": exclusive_pairs,
                "exclusive_rate": exclusive_rate,
                "jaccard_with_bm25": jaccards[
                    (profile, "bm25")
                ],
                "jaccard_with_harrier": jaccards[
                    (profile, "harrier")
                ],
                "jaccard_with_qwen": jaccards[
                    (profile, "qwen")
                ],
            }
        )

    return rows


def print_summary(
    rows: list[dict[str, Any]],
    query_count: int,
    unique_pair_count: int,
    output_path: Path,
) -> None:
    by_label = {
        row["profile"]: row
        for row in rows
    }

    print("Candidate-pool diversity analysis completed.")
    print(f" - Queries: {query_count}")
    print(
        f" - Unique query-dataset pairs: "
        f"{unique_pair_count}"
    )

    print("Retrieved pairs, exclusive pairs, and exclusive rates:")

    for row in rows:
        exclusive_rate = row["exclusive_rate"]

        rate_text = (
            f"{exclusive_rate:.3f}"
            if exclusive_rate is not None
            else "N/A"
        )

        print(
            f" - {row['profile']}: "
            f"retrieved={row['retrieved_pairs']}, "
            f"exclusive={row['exclusive_pairs']}, "
            f"exclusive_rate={rate_text}"
        )

    print("Pairwise Jaccard similarities:")

    comparisons = (
        ("BM25", "Harrier", "jaccard_with_harrier"),
        ("BM25", "Qwen", "jaccard_with_qwen"),
        ("Harrier", "Qwen", "jaccard_with_qwen"),
    )

    for left, right, column in comparisons:
        similarity = by_label[left][column]

        similarity_text = (
            f"{similarity:.3f}"
            if similarity is not None
            else "N/A"
        )

        print(
            f" - {left}-{right}: "
            f"{similarity_text}"
        )

    print(f" - Output: {output_path}")


def main() -> None:
    args = parse_args()

    (
        profile_sets,
        query_count,
        unique_pair_count,
    ) = load_profile_sets(
        args.candidate_pool
    )

    rows = compute_rows(
        profile_sets
    )

    fieldnames = [
        "profile",
        "retrieved_pairs",
        "exclusive_pairs",
        "exclusive_rate",
        "jaccard_with_bm25",
        "jaccard_with_harrier",
        "jaccard_with_qwen",
    ]

    write_csv_atomic(
        rows,
        args.output,
        fieldnames,
    )

    print_summary(
        rows,
        query_count,
        unique_pair_count,
        args.output,
    )


if __name__ == "__main__":
    main()
