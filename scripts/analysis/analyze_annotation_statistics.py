#!/usr/bin/env python3
"""Compute LLM annotation statistics from Stage 3 JSON files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPOSITORY_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (
    annotation_reason_and_relevance,
    extract_query_number,
    list_query_json_files,
    load_benchmark_config,
    load_models_config,
    make_pair_key,
    parse_dataset_id,
    require_positive_int,
    write_csv_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LLM annotation statistics."
    )

    parser.add_argument("--llm-annotations", type=Path, default=Path("data/llm_annotations"), help="Directory containing Stage 3 query_*.json files.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/annotation_statistics"), help="Output directory for annotation statistics CSV files.")
    parser.add_argument("--benchmark-config", type=Path, default=Path("config/benchmark_config.json"), help="Benchmark methodology configuration JSON.")
    parser.add_argument("--models-config", type=Path, default=Path("config/models.json"), help="Model registry JSON.")

    return parser.parse_args()


def analysis_settings_from_config(config):
    """Read annotation-analysis settings from the benchmark config."""
    llm_config = config["llm_annotation"]

    return {
        "models": [model.strip() for model in llm_config["models"]],
        "reason_labels": [reason.strip() for reason in llm_config["reason_labels"]],
        "relevance_mapping": dict(llm_config["relevance_mapping"]),
        "supporting_fields": [
            field.strip()
            for field in llm_config["positive_evidence_fields"]
        ],
    }


def model_display_names_from_config(path: Path) -> dict[str, str]:
    config = load_models_config(path)
    display_names = {}

    for model in config["models"]:
        model_id = model["id"].strip()
        short_name = str(model.get("short_name", "")).strip()
        display_names[model_id] = short_name or model_id

    return display_names


def apply_model_display_names(
    rows: list[dict[str, Any]],
    display_names: dict[str, str],
) -> list[dict[str, Any]]:
    output_rows = []

    for row in rows:
        output_row = dict(row)
        output_row["model"] = display_names.get(
            row["model"],
            row["model"],
        )
        output_rows.append(output_row)

    return output_rows


def load_and_count(
    input_dir: Path,
    models: list[str],
    reason_labels: list[str],
    relevance_mapping: dict[str, bool],
    supporting_fields: list[str],
) -> tuple[
    Counter,
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    dict[str, int],
]:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"LLM annotation directory does not exist: {input_dir}"
        )

    input_files = list_query_json_files(input_dir)
    model_set = set(models)

    model_counts = Counter()
    reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)

    pair_keys: set[str] = set()
    annotation_slots = 0

    for path in input_files:
        payload = __import__("json").loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(payload, dict):
            raise ValueError(
                f"Annotation file must contain a JSON object: {path}"
            )

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

        seen_dataset_ids: set[str] = set()

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

            if dataset_id in seen_dataset_ids:
                raise ValueError(
                    f"{path}: duplicate dataset ID {dataset_id!r}."
                )

            seen_dataset_ids.add(dataset_id)

            pair_key = make_pair_key(
                query_id,
                dataset_id,
            )

            if pair_key in pair_keys:
                raise ValueError(
                    f"Duplicate query-dataset pair: {pair_key}"
                )

            pair_keys.add(pair_key)

            annotations = hit.get("annotations")

            if not isinstance(annotations, dict):
                raise ValueError(
                    f"Invalid annotations object for {pair_key}"
                )

            if set(annotations) != model_set:
                missing_models = model_set - set(annotations)
                extra_models = set(annotations) - model_set

                raise ValueError(
                    f"Unexpected model set for {pair_key}: "
                    f"missing={sorted(missing_models)}, "
                    f"extra={sorted(extra_models)}"
                )

            for model in models:
                annotation_slots += 1
                annotation = annotations[model]

                if (
                    not isinstance(annotation, dict)
                    or "error" in annotation
                ):
                    model_counts[(model, "invalid")] += 1
                    continue

                reason, relevance = annotation_reason_and_relevance(
                    annotation,
                    relevance_mapping,
                )

                if reason is None or relevance is None:
                    raise ValueError(
                        f"Invalid annotation for {pair_key} / {model}"
                    )

                if reason not in reason_labels:
                    raise ValueError(
                        f"Unexpected reason for {pair_key} / {model}: "
                        f"{reason!r}"
                    )

                reason_counts[model][reason] += 1

                if relevance is True:
                    model_counts[(model, "positive")] += 1

                    supporting_field = annotation.get(
                        "supporting_field"
                    )

                    if supporting_field not in supporting_fields:
                        raise ValueError(
                            "Invalid supporting_field for positive "
                            f"annotation {pair_key} / {model}: "
                            f"{supporting_field!r}"
                        )

                    field_counts[model][supporting_field] += 1

                else:
                    model_counts[(model, "negative")] += 1

    summary = {
        "query_files": len(input_files),
        "pairs": len(pair_keys),
        "annotation_slots": annotation_slots,
    }

    return (
        model_counts,
        reason_counts,
        field_counts,
        summary,
    )


def build_model_label_rows(
    model_counts: Counter,
    models: list[str],
    total_pairs: int,
) -> list[dict[str, Any]]:
    rows = []

    for model in models:
        positive = model_counts[(model, "positive")]
        negative = model_counts[(model, "negative")]
        invalid = model_counts[(model, "invalid")]
        valid = positive + negative

        if valid + invalid != total_pairs:
            raise ValueError(
                f"Unexpected annotation count for {model}: "
                f"{valid + invalid} != {total_pairs}"
            )

        rows.append(
            {
                "model": model,
                "positive": positive,
                "negative": negative,
                "invalid": invalid,
                "valid": valid,
                "positive_label_rate": (
                    positive / float(valid)
                    if valid
                    else None
                ),
                "valid_output_rate": (
                    valid / float(total_pairs)
                    if total_pairs
                    else None
                ),
            }
        )

    return rows


def build_reason_rows(
    reason_counts: dict[str, Counter[str]],
    models: list[str],
    reason_labels: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for model in models:
        valid = sum(
            reason_counts[model][reason]
            for reason in reason_labels
        )

        row: dict[str, Any] = {
            "model": model,
            "valid": valid,
        }

        for reason in reason_labels:
            row[f"{reason}_count"] = (
                reason_counts[model][reason]
            )

        for reason in reason_labels:
            row[f"{reason}_rate"] = (
                reason_counts[model][reason] / float(valid)
                if valid
                else None
            )

        rows.append(row)

    return rows


def build_supporting_field_rows(
    model_counts: Counter,
    field_counts: dict[str, Counter[str]],
    models: list[str],
    supporting_fields: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for model in models:
        positive = model_counts[(model, "positive")]

        field_total = sum(
            field_counts[model][field]
            for field in supporting_fields
        )

        if field_total != positive:
            raise ValueError(
                f"{model} positive supporting-field counts "
                f"do not sum to positive annotations: "
                f"{field_total} != {positive}"
            )

        row: dict[str, Any] = {
            "model": model,
            "positive": positive,
        }

        for field in supporting_fields:
            row[f"{field}_count"] = (
                field_counts[model][field]
            )

        for field in supporting_fields:
            row[f"{field}_rate"] = (
                field_counts[model][field] / float(positive)
                if positive
                else None
            )

        rows.append(row)

    return rows


def print_summary(
    model_rows: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    reason_labels: list[str],
    supporting_fields: list[str],
    input_summary: dict[str, int],
) -> None:
    print("Annotation statistics computed.")
    print(f" - Query files: {input_summary['query_files']}")
    print(f" - Query-dataset pairs: {input_summary['pairs']}")
    print(f" - Annotation slots: {input_summary['annotation_slots']}")

    print("Model label distribution:")
    for row in model_rows:
        print(
            f" - {row['model']}: "
            f"positive={row['positive']}, "
            f"negative={row['negative']}, "
            f"invalid={row['invalid']}, "
            f"valid={row['valid']}, "
            f"positive_label_rate="
            f"{row['positive_label_rate']:.3f}, "
            f"valid_output_rate="
            f"{row['valid_output_rate']:.3f}"
        )

    print("Reason distribution:")
    for row in reason_rows:
        print(
            f" - {row['model']}: "
            + ", ".join(
                f"{reason}={row[f'{reason}_rate']:.3f}"
                for reason in reason_labels
            )
        )

    print("Supporting-field distribution:")
    for row in field_rows:
        print(
            f" - {row['model']}: "
            + ", ".join(
                f"{field}={row[f'{field}_rate']:.3f}"
                for field in supporting_fields
            )
        )


def main() -> None:
    args = parse_args()

    benchmark_config = load_benchmark_config(
        args.benchmark_config
    )
    analysis_config = analysis_settings_from_config(
        benchmark_config
    )

    model_display_names = model_display_names_from_config(
        args.models_config
    )
    registered_models = set(model_display_names)
    models = analysis_config["models"]

    unknown_models = [
        model
        for model in models
        if model not in registered_models
    ]

    if unknown_models:
        raise ValueError(
            "Annotation-analysis model(s) are not present "
            f"in models configuration: {unknown_models}"
        )

    reason_labels = analysis_config["reason_labels"]
    relevance_mapping = analysis_config["relevance_mapping"]
    supporting_fields = analysis_config["supporting_fields"]

    (
        model_counts,
        reason_counts,
        field_counts,
        input_summary,
    ) = load_and_count(
        args.llm_annotations,
        models,
        reason_labels,
        relevance_mapping,
        supporting_fields,
    )

    model_rows = build_model_label_rows(
        model_counts,
        models,
        input_summary["pairs"],
    )

    reason_rows = build_reason_rows(
        reason_counts,
        models,
        reason_labels,
    )

    field_rows = build_supporting_field_rows(
        model_counts,
        field_counts,
        models,
        supporting_fields,
    )

    model_rows = apply_model_display_names(
        model_rows,
        model_display_names,
    )
    reason_rows = apply_model_display_names(
        reason_rows,
        model_display_names,
    )
    field_rows = apply_model_display_names(
        field_rows,
        model_display_names,
    )

    model_fieldnames = [
        "model",
        "positive",
        "negative",
        "invalid",
        "valid",
        "positive_label_rate",
        "valid_output_rate",
    ]

    reason_fieldnames = [
        "model",
        "valid",
        *[
            f"{reason}_count"
            for reason in reason_labels
        ],
        *[
            f"{reason}_rate"
            for reason in reason_labels
        ],
    ]

    field_fieldnames = [
        "model",
        "positive",
        *[
            f"{field}_count"
            for field in supporting_fields
        ],
        *[
            f"{field}_rate"
            for field in supporting_fields
        ],
    ]

    write_csv_atomic(
        model_rows,
        args.output_dir / "model_label_distribution.csv",
        model_fieldnames,
    )

    write_csv_atomic(
        reason_rows,
        args.output_dir / "reason_distribution.csv",
        reason_fieldnames,
    )

    write_csv_atomic(
        field_rows,
        args.output_dir / "supporting_field_distribution.csv",
        field_fieldnames,
    )

    print_summary(
        model_rows,
        reason_rows,
        field_rows,
        reason_labels,
        supporting_fields,
        input_summary,
    )


if __name__ == "__main__":
    main()
