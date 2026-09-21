#!/usr/bin/env python3
"""Generate the frozen human-annotation subset from the candidate pool.

The human audit is a simple random sample without replacement over all
query-dataset pairs in the Stage 2 candidate pool. The population is sorted
deterministically before sampling so that the configured seed fully determines
the selected pairs and their order.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_utils import (
    extract_query_number,
    list_query_json_files,
    load_benchmark_config,
    load_json_object,
    make_pair_key,
    parse_dataset_id,
    require_positive_int,
    write_csv_atomic,
)


DEFAULT_CANDIDATE_POOL = Path("data/candidate_pool")
DEFAULT_BENCHMARK_CONFIG = Path("config/benchmark_config.json")
DEFAULT_MAPPING_OUTPUT = Path(
    "data/human_annotations/human_annotation_subset_mapping.csv"
)
DEFAULT_JSONL_OUTPUT = Path(
    "data/human_annotations/human_annotation_subset.jsonl"
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the human-annotation subset by simple random sampling "
            "without replacement from the Stage 2 candidate pool."
        )
    )

    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL, help="Directory containing Stage 2 query_*.json candidate-pool files.")
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_BENCHMARK_CONFIG, help="Benchmark configuration JSON containing human_audit.sample_size and human_audit.sample_seed.")
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT, help="Output CSV mapping for the sampled query-dataset pairs.")
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT, help="Output JSONL payload for human annotators.")

    return parser.parse_args()


def human_annotation_settings_from_config(config) -> dict[str, int]:
    """Read human-annotation sampling settings from the benchmark config."""
    human_config = config["human_audit"]

    return {
        "sample_size": human_config["sample_size"],
        "sample_seed": human_config["sample_seed"],
    }


def load_candidate_pairs(
    candidate_pool_dir: Path,
) -> dict[str, dict[str, Any]]:
    if not candidate_pool_dir.is_dir():
        fail(
            f"Candidate-pool directory does not exist: "
            f"{candidate_pool_dir}"
        )

    input_files = list_query_json_files(candidate_pool_dir)

    pairs: dict[str, dict[str, Any]] = {}
    seen_query_ids: set[int] = set()

    for path in input_files:
        filename_query_id = extract_query_number(path.name)
        payload = load_json_object(path)

        id_query = require_positive_int(
            payload.get("id_query"),
            f"{path}: id_query",
        )

        if id_query != filename_query_id:
            fail(
                f"Filename/query ID mismatch in {path}: "
                f"{filename_query_id} != {id_query}"
            )

        if id_query in seen_query_ids:
            fail(
                f"Duplicate query ID in candidate pool: {id_query}"
            )

        seen_query_ids.add(id_query)

        query = payload.get("query")

        if not isinstance(query, str) or not query.strip():
            fail(
                f"Invalid query text in {path}: {query!r}"
            )

        query = query.strip()
        hits = payload.get("hits")

        if not isinstance(hits, list):
            fail(f"'hits' must be a list in {path}")

        seen_dataset_ids: set[str] = set()

        for result_index, hit in enumerate(hits):
            if not isinstance(hit, dict):
                fail(
                    f"Invalid hit at index {result_index} in {path}"
                )

            id_dataset = parse_dataset_id(
                hit.get("id"),
                f"{path}, hit {result_index}",
            )

            if id_dataset in seen_dataset_ids:
                fail(
                    f"Duplicate dataset ID {id_dataset!r} in {path}"
                )

            seen_dataset_ids.add(id_dataset)

            pair_key = make_pair_key(
                id_query,
                id_dataset,
            )

            if pair_key in pairs:
                fail(
                    "Duplicate query-dataset pair in candidate pool: "
                    f"{pair_key}"
                )

            pairs[pair_key] = {
                "pair_key": pair_key,
                "id_query": id_query,
                "query": query,
                "result_index": result_index,
                "id_dataset": id_dataset,
                "title": hit.get("title", ""),
                "description": hit.get("description", ""),
                "header": hit.get("header", ""),
                "content": hit.get("content", ""),
            }

    return pairs


def draw_sample(
    pairs: dict[str, dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[str]:
    if sample_size > len(pairs):
        fail(
            f"Sample size {sample_size} is larger than candidate pool "
            f"({len(pairs)} pairs)."
        )

    population = sorted(pairs)
    rng = random.Random(seed)

    return rng.sample(
        population,
        sample_size,
    )


def write_mapping(
    path: Path,
    sample_keys: list[str],
    pairs: dict[str, dict[str, Any]],
) -> None:
    """Write the sampled query-dataset mapping."""
    fieldnames = [
        "order",
        "pair_key",
        "id_query",
        "query",
        "result_index",
        "id_dataset",
        "title",
    ]

    rows = []

    for order, pair_key in enumerate(sample_keys):
        item = pairs[pair_key]

        rows.append(
            {
                "order": order,
                "pair_key": pair_key,
                "id_query": item["id_query"],
                "query": item["query"],
                "result_index": item["result_index"],
                "id_dataset": item["id_dataset"],
                "title": item["title"],
            }
        )

    write_csv_atomic(
        rows,
        path,
        fieldnames,
    )


def write_jsonl(
    path: Path,
    sample_keys: list[str],
    pairs: dict[str, dict[str, Any]],
) -> None:
    """Write the human-annotation payload atomically as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for order, pair_key in enumerate(sample_keys):
                item = dict(pairs[pair_key])
                item["order"] = order

                file.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        temporary_path.replace(path)

    except Exception:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass

        raise


def summarize_sample(
    sample_keys: list[str],
    pairs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    query_counts = Counter(
        pairs[pair_key]["id_query"]
        for pair_key in sample_keys
    )

    return {
        "candidate_pool_pairs": len(pairs),
        "sampled_pairs": len(sample_keys),
        "distinct_queries": len(query_counts),
        "min_pairs_per_query": (
            min(query_counts.values())
            if query_counts
            else 0
        ),
        "max_pairs_per_query": (
            max(query_counts.values())
            if query_counts
            else 0
        ),
    }


def print_sample_summary(
    summary: dict[str, Any],
    seed: int,
    mapping_output: Path,
    jsonl_output: Path,
) -> None:
    print("Human-annotation subset generated.")
    print(
        f" - Candidate-pool pairs: "
        f"{summary['candidate_pool_pairs']}"
    )
    print(
        f" - Sample size: "
        f"{summary['sampled_pairs']}"
    )
    print(f" - Random seed: {seed}")
    print(
        f" - Distinct represented queries: "
        f"{summary['distinct_queries']}"
    )
    print(
        " - Sampled pairs per represented query: "
        f"min={summary['min_pairs_per_query']}, "
        f"max={summary['max_pairs_per_query']}"
    )
    print(f" - Mapping CSV: {mapping_output}")
    print(f" - Annotation JSONL: {jsonl_output}")


def main() -> None:
    args = parse_args()

    benchmark_config = load_benchmark_config(
        args.benchmark_config
    )
    sampling_config = human_annotation_settings_from_config(
        benchmark_config
    )

    sample_size = sampling_config["sample_size"]
    seed = sampling_config["sample_seed"]

    pairs = load_candidate_pairs(
        args.candidate_pool
    )

    sample_keys = draw_sample(
        pairs,
        sample_size,
        seed,
    )

    write_mapping(
        args.mapping_output,
        sample_keys,
        pairs,
    )

    write_jsonl(
        args.jsonl_output,
        sample_keys,
        pairs,
    )

    summary = summarize_sample(
        sample_keys,
        pairs,
    )

    print_sample_summary(
        summary=summary,
        seed=seed,
        mapping_output=args.mapping_output,
        jsonl_output=args.jsonl_output,
    )


if __name__ == "__main__":
    main()