#!/usr/bin/env python3
"""Aggregate Stage 3 LLM judgments using configured three-judge majority voting."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from pipeline_utils import (
    annotation_reason_and_relevance,
    extract_query_number,
    list_query_json_files,
    load_benchmark_config,
    load_json_object,
    load_model_ids,
    make_pair_key,
    parse_dataset_id,
    parse_pair_key,
    parse_query_id,
    require_positive_int,
    write_csv_atomic,
)


def parse_filter_pair(row, context):
    """Parse one pair-filter row into its canonical query-dataset key."""    
    pair_key_value = row.get("pair_key")
    query_value = row.get("id_query")
    dataset_value = row.get("id_dataset")

    has_pair_key = (
        pair_key_value is not None
        and str(pair_key_value).strip() != ""
    )
    has_query_dataset = (
        query_value is not None
        and str(query_value).strip() != ""
        and dataset_value is not None
        and str(dataset_value).strip() != ""
    )

    if not has_pair_key and not has_query_dataset:
        raise ValueError(
            f"{context}: expected pair_key or id_query + id_dataset."
        )

    pair_from_key = (
        parse_pair_key(pair_key_value, context)
        if has_pair_key
        else None
    )

    pair_from_columns = None

    if has_query_dataset:
        query_id = parse_query_id(query_value, context)
        dataset_id = str(dataset_value).strip()

        if not dataset_id:
            raise ValueError(
                f"{context}: id_dataset must be non-empty."
            )

        pair_from_columns = make_pair_key(query_id, dataset_id)

    if (
        pair_from_key is not None
        and pair_from_columns is not None
        and pair_from_key != pair_from_columns
    ):
        raise ValueError(
            f"{context}: pair_key is inconsistent with id_query/id_dataset."
        )

    if pair_from_key is not None:
        return pair_from_key

    return pair_from_columns


def add_pair_filter_key(pair_keys, pair_key, context):
    """Add a pair to the filter while rejecting duplicate entries."""
    if pair_key in pair_keys:
        raise ValueError(
            f"Duplicate pair in pair filter at {context}: {pair_key}"
        )

    pair_keys.add(pair_key)


def majority_vote_2_of_3(values):
    """
    Apply strict 2-out-of-3 majority voting.

    All three judgments must be available. If any judgment is missing or
    invalid, return None.
    """
    if len(values) != 3:
        raise ValueError(
            "2-out-of-3 majority voting requires exactly three values."
        )

    if any(value is None for value in values):
        return None

    positive_votes = sum(value is True for value in values)

    return positive_votes >= 2


def load_pair_filter(path):
    """
    Load an optional fixed set of query-dataset pairs.

    Supported formats:
      - CSV
      - JSONL

    Each row must provide either:
      - id_query and id_dataset
      - pair_key
    """
    if path is None:
        return None

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Pair-filter file does not exist: {path}")

    pair_keys = set()

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(f"Pair-filter CSV has no header: {path}")

            fieldnames = set(reader.fieldnames)

            if (
                "pair_key" not in fieldnames
                and not {"id_query", "id_dataset"}.issubset(fieldnames)
            ):
                raise ValueError(
                    "Pair-filter CSV header must contain pair_key or "
                    "both id_query and id_dataset."
                )

            for row_number, row in enumerate(reader, start=2):
                if all(
                    value is None or str(value).strip() == ""
                    for value in row.values()
                ):
                    raise ValueError(
                        f"Blank row in pair-filter CSV at line {row_number}."
                    )

                context = f"{path}, line {row_number}"
                pair_key = parse_filter_pair(row, context)
                add_pair_filter_key(pair_keys, pair_key, context)

    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in pair-filter at line {line_number}: "
                        f"{exc.msg}"
                    ) from exc

                if not isinstance(row, dict):
                    raise ValueError(
                        f"Invalid pair-filter row at line {line_number}: "
                        "expected a JSON object."
                    )

                context = f"{path}, line {line_number}"
                pair_key = parse_filter_pair(row, context)
                add_pair_filter_key(pair_keys, pair_key, context)

    else:
        raise ValueError(
            "Unsupported pair-filter format. Use CSV or JSONL."
        )

    if not pair_keys:
        raise ValueError(f"Pair-filter file contains no pairs: {path}")

    return pair_keys


def load_annotated_pairs(input_dir):
    """
    Load Stage 3 LLM annotation files.

    Returns one record per query-dataset pair while preserving the complete
    model annotation mapping.
    """
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {input_dir}"
        )

    input_files = list_query_json_files(input_dir)
    pairs = {}

    for input_path in input_files:
        payload = load_json_object(input_path)

        filename_query_id = extract_query_number(input_path.name)

        for required_key in ("id_query", "query", "hits"):
            if required_key not in payload:
                raise ValueError(
                    f"Annotation file is missing {required_key!r}: {input_path}"
                )

        query_id = require_positive_int(
            payload["id_query"],
            f"{input_path}: id_query",
        )

        if query_id != filename_query_id:
            raise ValueError(
                f"Annotation filename/query ID mismatch in {input_path}: "
                f"filename ID {filename_query_id} != id_query {query_id}"
            )

        query = payload["query"]

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Invalid query text in {input_path}: {query!r}"
            )

        query = query.strip()
        hits = payload["hits"]

        if not isinstance(hits, list):
            raise ValueError(f"'hits' must be a list in {input_path}")

        seen_dataset_ids = set()

        for hit_index, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                raise ValueError(
                    f"Hit #{hit_index} must be an object in {input_path}"
                )

            if "id" not in hit:
                raise ValueError(
                    f"Hit #{hit_index} is missing dataset ID in {input_path}"
                )

            dataset_id = parse_dataset_id(
                hit["id"],
                f"{input_path}, hit #{hit_index}",
            )

            if dataset_id in seen_dataset_ids:
                raise ValueError(
                    f"Duplicate dataset ID {dataset_id!r} in {input_path}"
                )

            seen_dataset_ids.add(dataset_id)

            pair_key = make_pair_key(query_id, dataset_id)

            if pair_key in pairs:
                raise ValueError(
                    "Duplicate query-dataset pair found in annotation files: "
                    f"{pair_key}"
                )

            if "annotations" not in hit:
                raise ValueError(
                    f"Hit {dataset_id!r} is missing annotations in {input_path}"
                )

            annotations = hit["annotations"]

            if not isinstance(annotations, dict):
                raise ValueError(
                    f"Invalid annotations object for pair {pair_key}"
                )

            pairs[pair_key] = {
                "pair_key": pair_key,
                "id_query": str(query_id),
                "query": query,
                "id_dataset": dataset_id,
                "annotations": annotations,
            }

    return pairs


def validate_models(pairs, models):
    """
    Validate that every requested model appears somewhere in the annotation
    collection.
    """
    observed_models = set()

    for pair in pairs.values():
        observed_models.update(pair["annotations"].keys())

    missing_models = [
        model
        for model in models
        if model not in observed_models
    ]

    if missing_models:
        raise ValueError(f"Requested models were not found in the annotation files: {missing_models}")


def build_aggregations(
    pairs,
    models,
    relevance_mapping,
    pair_filter=None,
):
    """
    Generate every three-judge combination and apply strict 2-out-of-3
    majority voting for every selected query-dataset pair.
    """
    # Generate combinations in configured model order so judge_1, judge_2, and
    # judge_3 remain stable and reproducible across runs.
    combinations = list(itertools.combinations(models, 3))

    rows = []
    found_pair_keys = set()

    for pair_key in sorted(
        pairs,
        key=lambda key: (
            int(pairs[key]["id_query"])
            if pairs[key]["id_query"].isdigit()
            else pairs[key]["id_query"],
            pairs[key]["id_dataset"],
        ),
    ):
        if pair_filter is not None and pair_key not in pair_filter:
            continue

        pair = pairs[pair_key]
        found_pair_keys.add(pair_key)

        for combination in combinations:
            judge_1, judge_2, judge_3 = combination
            # Missing, malformed, or explicitly invalid model annotations contribute a
            # missing vote.
            _, relevance_1 = annotation_reason_and_relevance(
                pair["annotations"].get(judge_1),
                relevance_mapping,
            )
            _, relevance_2 = annotation_reason_and_relevance(
                pair["annotations"].get(judge_2),
                relevance_mapping,
            )
            _, relevance_3 = annotation_reason_and_relevance(
                pair["annotations"].get(judge_3),
                relevance_mapping,
            )

            votes = [
                relevance_1,
                relevance_2,
                relevance_3,
            ]

            aggregated_relevance = majority_vote_2_of_3(votes)
            valid_votes = sum(value is not None for value in votes)

            rows.append(
                {
                    "pair_key": pair_key,
                    "id_query": pair["id_query"],
                    "id_dataset": pair["id_dataset"],
                    "combination": " + ".join(combination),
                    "judge_1": judge_1,
                    "judge_2": judge_2,
                    "judge_3": judge_3,
                    "judge_1_relevance": relevance_1,
                    "judge_2_relevance": relevance_2,
                    "judge_3_relevance": relevance_3,
                    "valid_votes": valid_votes,
                    "aggregated_relevance": aggregated_relevance,
                }
            )
    # A requested pair must never disappear silently from a filtered aggregation.
    if pair_filter is not None:
        missing_pairs = pair_filter - found_pair_keys

        if missing_pairs:
            raise ValueError(
                "Some pairs from the requested filter were not found "
                "in the annotation files: "
                f"{sorted(missing_pairs)[:10]} "
                f"(total missing: {len(missing_pairs)})"
            )

    return rows, combinations


def write_aggregations_csv(rows, output_path):
    """Write the pair-level three-judge aggregation table."""
    fieldnames = [
        "pair_key",
        "id_query",
        "id_dataset",
        "combination",
        "judge_1",
        "judge_2",
        "judge_3",
        "judge_1_relevance",
        "judge_2_relevance",
        "judge_3_relevance",
        "valid_votes",
        "aggregated_relevance",
    ]

    write_csv_atomic(rows, output_path, fieldnames)


def build_summary(rows):
    """
    Summarize valid, missing, positive, and negative aggregate judgments
    for each three-judge combination.
    """
    grouped = defaultdict(
        lambda: {
            "pairs": 0,
            "valid": 0,
            "missing": 0,
            "positive": 0,
            "negative": 0,
        }
    )

    for row in rows:
        stats = grouped[row["combination"]]
        stats["pairs"] += 1

        relevance = row["aggregated_relevance"]

        if relevance is None:
            stats["missing"] += 1
        else:
            stats["valid"] += 1

            if relevance is True:
                stats["positive"] += 1
            else:
                stats["negative"] += 1

    summary_rows = []

    for combination in sorted(grouped):
        stats = grouped[combination]
        valid = stats["valid"]
        positive = stats["positive"]

        summary_rows.append(
            {
                "combination": combination,
                "pairs": stats["pairs"],
                "valid": valid,
                "missing": stats["missing"],
                "positive": positive,
                "negative": stats["negative"],
                "positive_rate": positive / valid if valid else None,
            }
        )

    return summary_rows


def write_summary_csv(rows, output_path):
    """Write aggregate counts for each three-judge combination."""
    fieldnames = [
        "combination",
        "pairs",
        "valid",
        "missing",
        "positive",
        "negative",
        "positive_rate",
    ]

    write_csv_atomic(rows, output_path, fieldnames)


def aggregation_settings_from_config(config):
    """Read Stage 4 aggregation settings from the benchmark config."""
    llm_config = config["llm_annotation"]

    models = [model.strip() for model in llm_config["models"]]

    if len(models) < 3:
        raise ValueError(
            "At least three LLM judges are required for Stage 4 aggregation."
        )

    return {
        "models": models,
        "relevance_mapping": dict(llm_config["relevance_mapping"]),
    }


def parse_args():
    """Parse Stage 4 judge-aggregation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate all three-judge LLM combinations and apply strict 2-out-of-3 majority voting."
    )

    parser.add_argument("--input", "-i", default="data/llm_annotations", help="Directory containing Stage 3 annotated query_*.json files.")
    parser.add_argument("--output", "-o", default="results/judge_aggregations.csv", help="Output CSV containing pair-level three-judge aggregations.")
    parser.add_argument("--benchmark-config", default="config/benchmark_config.json", help="Benchmark configuration JSON used for aggregation settings.")
    parser.add_argument("--models-config", default="config/models.json", help="Model registry JSON used to resolve configured LLM judges.")
    parser.add_argument("--pair-filter", default=None, help="Optional CSV or JSONL containing the query-dataset pairs to aggregate.")

    return parser.parse_args()


def print_aggregation_summary(
    models,
    combinations,
    selected_pair_count,
    aggregation_rows,
    output_path,
    summary_path,
):
    """Print summary information for the generated judge aggregations."""
    print("Judge aggregation completed.")
    print(f" - Models: {len(models)}")
    print(f" - Three-judge combinations: {len(combinations)}")
    print(f" - Query-dataset pairs: {selected_pair_count}")
    print(f" - Aggregation rows: {len(aggregation_rows)}")
    print(f" - Pair-level output: {output_path}")
    print(f" - Summary output: {summary_path}")


def main():
    """Run Stage 4 three-judge aggregation over the LLM annotations."""
    args = parse_args()

    benchmark_config = load_benchmark_config(args.benchmark_config)
    aggregation_config = aggregation_settings_from_config(benchmark_config)

    configured_models = set(load_model_ids(args.models_config))
    models = aggregation_config["models"]

    unknown_models = [
        model
        for model in models
        if model not in configured_models
    ]

    if unknown_models:
        raise ValueError(
            "Aggregation judge(s) are not present in models configuration: "
            f"{unknown_models}"
        )

    input_dir = Path(args.input)
    output_path = Path(args.output)
    pair_filter = load_pair_filter(args.pair_filter)
    pairs = load_annotated_pairs(input_dir)

    validate_models(pairs, models)

    aggregation_rows, combinations = build_aggregations(
        pairs,
        models,
        aggregation_config["relevance_mapping"],
        pair_filter=pair_filter,
    )

    write_aggregations_csv(aggregation_rows, output_path)

    summary_rows = build_summary(aggregation_rows)
    summary_path = output_path.parent / f"{output_path.stem}_summary.csv"

    write_summary_csv(summary_rows, summary_path)

    selected_pair_count = (
        len(pair_filter)
        if pair_filter is not None
        else len(pairs)
    )

    print_aggregation_summary(
        models=models,
        combinations=combinations,
        selected_pair_count=selected_pair_count,
        aggregation_rows=aggregation_rows,
        output_path=output_path,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()
