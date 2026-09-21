#!/usr/bin/env python3
"""Evaluate retrieval profiles against the canonical benchmark.

The evaluation computes retrieval effectiveness from the Stage 6 benchmark
JSONL using its binary `default_relevance` labels.

Definitions:
  * The default relevance source is defined by the benchmark configuration.
  * Recall is measured against all relevant query-dataset pairs in the judged
    candidate pool for each query.
  * Rankings are reconstructed from the stored retrieval ranks.
  * Metrics are macro-averaged across queries.

Binary nDCG uses the standard formulation:
  DCG@k = sum((2^rel_i - 1) / log2(i + 1))
where `i` is the 1-based rank and `rel_i` is 0 or 1. With binary relevance,
the gain term is therefore 1 for relevant items and 0 otherwise. IDCG@k is
computed from the distribution of default relevance labels in the full judged
candidate pool for that query, limited to the same cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPOSITORY_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (
    load_benchmark_config,
    make_pair_key,
    parse_dataset_id,
    parse_pair_key,
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


def canonical_retrieval(retrieval: dict[str, Any], pair_key: str) -> dict[str, Any]:
    canonical = {
        canonical_profile_name(profile): profile_data
        for profile, profile_data in retrieval.items()
    }

    if len(canonical) != len(retrieval):
        raise ValueError(
            f"Duplicate retrieval profiles after aliasing for {pair_key}"
        )

    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25, Harrier, and Qwen retrieval effectiveness "
            "from the canonical benchmark JSONL."
        )
    )

    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark/benchmark.jsonl"), help="Canonical Stage 6 benchmark JSONL.")
    parser.add_argument("--benchmark-config", type=Path, default=Path("config/benchmark_config.json"), help="Benchmark methodology configuration JSON.")
    parser.add_argument("--output", type=Path, default=Path("results/retrieval_effectiveness.csv"), help="Output CSV for retrieval effectiveness metrics.")

    return parser.parse_args()


def retrieval_evaluation_settings_from_config(config):
    """Read retrieval-evaluation settings from the benchmark config."""
    return {
        "default_relevance_source": (
            config["benchmark"]["default_relevance_source"].strip()
        ),
    }


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark file does not exist: {path}"
        )

    records = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid benchmark record at {path}:{line_number}: "
                    "expected an object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"Benchmark contains no records: {path}"
        )

    return records


def validate_benchmark(
    records: list[dict[str, Any]],
    default_relevance_source: str,
) -> dict[str, Any]:
    pair_keys: set[str] = set()
    query_ids: set[int] = set()
    memberships = Counter()
    relevant_pairs = 0
    non_relevant_pairs = 0

    for record_index, record in enumerate(records, start=1):
        context = f"benchmark record #{record_index}"

        for required_key in (
            "pair_key",
            "id_query",
            "id_dataset",
            "default_relevance_source",
            "default_relevance",
            "retrieval",
        ):
            if required_key not in record:
                raise ValueError(
                    f"{context}: missing {required_key!r}."
                )

        query_id = require_positive_int(
            record["id_query"],
            f"{context}: id_query",
        )

        dataset_id = parse_dataset_id(
            record["id_dataset"],
            f"{context}: id_dataset",
        )

        expected_pair_key = make_pair_key(
            query_id,
            dataset_id,
        )

        stored_pair_key = parse_pair_key(
            record["pair_key"],
            context,
        )

        if stored_pair_key != expected_pair_key:
            raise ValueError(
                f"{context}: inconsistent pair_key "
                f"{stored_pair_key!r} != {expected_pair_key!r}"
            )

        if stored_pair_key in pair_keys:
            raise ValueError(
                f"Duplicate benchmark pair: {stored_pair_key}"
            )

        pair_keys.add(stored_pair_key)
        query_ids.add(query_id)

        source = record["default_relevance_source"]

        if source != default_relevance_source:
            raise ValueError(
                f"{stored_pair_key}: unexpected default relevance source "
                f"{source!r} != {default_relevance_source!r}"
            )

        relevance = record["default_relevance"]

        if not isinstance(relevance, bool):
            raise ValueError(
                f"{stored_pair_key}: default_relevance must be Boolean."
            )

        if relevance:
            relevant_pairs += 1
        else:
            non_relevant_pairs += 1

        retrieval = record["retrieval"]

        if not isinstance(retrieval, dict):
            raise ValueError(
                f"{stored_pair_key}: retrieval must be an object."
            )

        record["retrieval"] = canonical_retrieval(
            retrieval,
            stored_pair_key,
        )

        for profile in PROFILE_ORDER:
            if profile in record["retrieval"]:
                memberships[profile] += 1

    return {
        "records": len(records),
        "queries": len(query_ids),
        "relevant_pairs": relevant_pairs,
        "non_relevant_pairs": non_relevant_pairs,
        "profile_memberships": dict(memberships),
    }


def group_by_query(
    records: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        grouped[record["id_query"]].append(record)

    return dict(grouped)


def validate_profile_ranks(
    grouped_records: dict[int, list[dict[str, Any]]],
) -> None:
    for query_id, records in grouped_records.items():
        for profile in PROFILE_ORDER:
            ranks = []

            for record in records:
                retrieval = record["retrieval"]

                if profile not in retrieval:
                    continue

                profile_data = retrieval[profile]

                if not isinstance(profile_data, dict):
                    raise ValueError(
                        "Invalid retrieval profile object for "
                        f"{record['pair_key']} / {profile}"
                    )

                rank = profile_data.get("rank")

                if (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank <= 0
                ):
                    raise ValueError(
                        f"Invalid rank for "
                        f"{record['pair_key']} / {profile}: {rank!r}"
                    )

                ranks.append(rank)

            duplicate_ranks = [
                rank
                for rank, count in Counter(ranks).items()
                if count > 1
            ]

            if duplicate_ranks:
                raise ValueError(
                    f"Duplicate ranks for query {query_id} / {profile}: "
                    f"{duplicate_ranks[:10]}"
                )

            if ranks:
                sorted_ranks = sorted(ranks)
                expected_ranks = list(
                    range(1, len(sorted_ranks) + 1)
                )

                if sorted_ranks != expected_ranks:
                    raise ValueError(
                        f"Non-contiguous ranks for query "
                        f"{query_id} / {profile}: "
                        f"expected 1..{len(sorted_ranks)}"
                    )


def dcg_at_k(
    gains: list[int],
    k: int,
) -> float:
    return sum(
        ((2 ** relevance) - 1)
        / math.log2(index + 1)
        for index, relevance in enumerate(
            gains[:k],
            start=1,
        )
    )


def ndcg_at_k(
    ranked_gains: list[int],
    relevant_count: int,
    k: int,
) -> float:
    dcg = dcg_at_k(
        ranked_gains,
        k,
    )

    ideal_gains = [
        1
        for _ in range(
            min(relevant_count, k)
        )
    ]

    ideal_dcg = dcg_at_k(
        ideal_gains,
        k,
    )

    if ideal_dcg == 0.0:
        raise ValueError(
            "Cannot compute nDCG for a query with zero relevant pairs."
        )

    return dcg / ideal_dcg


def recall_at_k(
    ranked_gains: list[int],
    relevant_count: int,
    k: int,
) -> float:
    if relevant_count == 0:
        raise ValueError(
            "Cannot compute recall for a query with zero relevant pairs."
        )

    return (
        sum(ranked_gains[:k])
        / float(relevant_count)
    )


def evaluate_profile(
    grouped_records: dict[int, list[dict[str, Any]]],
    profile: str,
) -> dict[str, Any]:
    ndcg_10_values = []
    ndcg_50_values = []
    recall_10_values = []
    recall_50_values = []

    zero_relevant_queries = [
        query_id
        for query_id, records in grouped_records.items()
        if not any(
            record["default_relevance"] is True
            for record in records
        )
    ]

    if zero_relevant_queries:
        raise ValueError(
            "Queries with zero relevant pairs in the judged "
            "candidate pool: "
            f"{zero_relevant_queries}"
        )

    for records in grouped_records.values():
        relevant_count = sum(
            record["default_relevance"] is True
            for record in records
        )

        ranked_records = sorted(
            (
                record
                for record in records
                if profile in record["retrieval"]
            ),
            key=lambda record: (
                record["retrieval"][profile]["rank"]
            ),
        )

        ranked_gains = [
            1
            if record["default_relevance"] is True
            else 0
            for record in ranked_records
        ]

        ndcg_10_values.append(
            ndcg_at_k(
                ranked_gains,
                relevant_count,
                10,
            )
        )

        ndcg_50_values.append(
            ndcg_at_k(
                ranked_gains,
                relevant_count,
                50,
            )
        )

        recall_10_values.append(
            recall_at_k(
                ranked_gains,
                relevant_count,
                10,
            )
        )

        recall_50_values.append(
            recall_at_k(
                ranked_gains,
                relevant_count,
                50,
            )
        )

    query_count = len(grouped_records)

    return {
        "profile": PROFILE_LABELS[profile],
        "ndcg_at_10": (
            sum(ndcg_10_values) / query_count
        ),
        "ndcg_at_50": (
            sum(ndcg_50_values) / query_count
        ),
        "recall_at_10": (
            sum(recall_10_values) / query_count
        ),
        "recall_at_50": (
            sum(recall_50_values) / query_count
        ),
        "queries_evaluated": query_count,
    }


def print_summary(
    benchmark_summary: dict[str, Any],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    print("Retrieval evaluation completed.")
    print(
        f" - Benchmark records: "
        f"{benchmark_summary['records']}"
    )
    print(
        f" - Queries: "
        f"{benchmark_summary['queries']}"
    )
    print(
        f" - Relevant pairs: "
        f"{benchmark_summary['relevant_pairs']}"
    )
    print(
        f" - Non-relevant pairs: "
        f"{benchmark_summary['non_relevant_pairs']}"
    )

    print("Profile memberships:")

    for profile in PROFILE_ORDER:
        print(
            f" - {PROFILE_LABELS[profile]}: "
            f"{benchmark_summary['profile_memberships'].get(profile, 0)}"
        )

    print("Retrieval metrics:")

    for row in rows:
        print(
            f" - {row['profile']}: "
            f"nDCG@10={row['ndcg_at_10']:.3f}, "
            f"nDCG@50={row['ndcg_at_50']:.3f}, "
            f"Recall@10={row['recall_at_10']:.3f}, "
            f"Recall@50={row['recall_at_50']:.3f}"
        )

    print(f" - Output: {output_path}")


def main() -> None:
    args = parse_args()

    benchmark_config = load_benchmark_config(
        args.benchmark_config
    )
    evaluation_config = (
        retrieval_evaluation_settings_from_config(
            benchmark_config
        )
    )

    records = load_benchmark(
        args.benchmark
    )

    benchmark_summary = validate_benchmark(
        records,
        evaluation_config["default_relevance_source"],
    )

    grouped_records = group_by_query(
        records
    )

    validate_profile_ranks(
        grouped_records
    )

    rows = [
        evaluate_profile(
            grouped_records,
            profile,
        )
        for profile in PROFILE_ORDER
    ]

    fieldnames = [
        "profile",
        "ndcg_at_10",
        "ndcg_at_50",
        "recall_at_10",
        "recall_at_50",
        "queries_evaluated",
    ]

    write_csv_atomic(
        rows,
        args.output,
        fieldnames,
    )

    print_summary(
        benchmark_summary,
        rows,
        args.output,
    )


if __name__ == "__main__":
    main()
