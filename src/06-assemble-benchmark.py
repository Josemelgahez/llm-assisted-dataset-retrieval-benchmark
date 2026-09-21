#!/usr/bin/env python3
"""Assemble the canonical benchmark from LLM and human annotations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_utils import (
    annotation_reason_and_relevance,
    extract_query_number,
    list_query_json_files,
    load_benchmark_config,
    load_json_object,
    load_model_ids,
    make_pair_key,
    parse_dataset_id,
    parse_row_pair_key,
    require_positive_int,
)

DEFAULT_HUMAN_ANNOTATIONS = (
    "h1=data/human_annotations/annotator_1.csv",
    "h2=data/human_annotations/annotator_2.csv",
    "h3=data/human_annotations/annotator_3.csv",
)

DATASET_FIELDS = (
    "id",
    "title",
    "description",
    "header",
    "content",
    "metadato_fileName",
    "resource_fileName",
    "retrieval",
    "annotations",
)


def parse_args() -> argparse.Namespace:
    """Parse Stage 6 benchmark-assembly command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble the canonical benchmark from Stage 3 LLM annotations and the configured human annotations."
    )

    parser.add_argument("--benchmark-config", type=Path, default=Path("config/benchmark_config.json"), help="Benchmark methodology configuration JSON.")
    parser.add_argument("--models-config", type=Path, default=Path("config/models.json"), help="Model registry JSON.")
    parser.add_argument("--llm-annotations", type=Path, default=Path("data/llm_annotations"), help="Directory containing canonical Stage 3 query_*.json files.")
    parser.add_argument("--human-majority", type=Path, default=Path("data/human_annotations/human_majority_labels.csv"), help="CSV file with binary human-majority labels.")
    parser.add_argument("--human-label", action="append", default=None, help="Human annotation CSV in NAME=PATH format. Repeat once per annotator.")
    parser.add_argument("--output", type=Path, default=Path("data/benchmark/benchmark.jsonl"), help="Output JSONL benchmark path.")
    parser.add_argument("--copy-judge-aggregations-from", type=Path, default=None, help="Optional judge-aggregation CSV to copy alongside the benchmark artifacts.")
    parser.add_argument("--judge-aggregations-output", type=Path, default=Path("data/derived/judge_aggregations.csv"), help="Destination for the optional judge-aggregation CSV copy.")

    return parser.parse_args()


def fail(message: str) -> None:
    """Terminate Stage 6 with a user-facing validation error."""
    raise SystemExit(f"ERROR: {message}")


def parse_bool(value: Any, context: str) -> bool:
    """Parse a strict Boolean value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    fail(f"Invalid Boolean value in {context}: {value!r}")


def assembly_settings_from_config(config):
    """Read Stage 6 assembly settings from the benchmark config."""
    llm_config = config["llm_annotation"]
    human_config = config["human_audit"]
    benchmark_config = config["benchmark"]

    models = [model.strip() for model in llm_config["models"]]
    default_model = benchmark_config["default_relevance_source"].strip()

    if default_model not in models:
        raise ValueError(
            "benchmark.default_relevance_source must be one of the "
            "configured LLM judges."
        )

    return {
        "models": models,
        "relevance_mapping": dict(llm_config["relevance_mapping"]),
        "human_audit_sample_size": human_config["sample_size"],
        "human_annotators": human_config["annotators"],
        "default_relevance_model": default_model,
    }


def validate_annotation(
    model: str,
    annotation: Any,
    pair_key: str,
    relevance_mapping: dict[str, bool],
) -> None:
    """Validate one stored LLM annotation against the relevance protocol."""
    if not isinstance(annotation, dict):
        raise ValueError(
            f"Annotation for {model} at {pair_key} is not an object."
        )

    if "error" in annotation:
        return

    if "relevance" not in annotation:
        raise ValueError(
            f"Missing relevance for {model} at {pair_key}."
        )

    reason, relevance = annotation_reason_and_relevance(
        annotation,
        relevance_mapping,
    )

    if reason is None or relevance is None:
        raise ValueError(
            f"Invalid annotation for {model} at {pair_key}."
        )


def normalize_human_annotation(
    row: dict[str, str],
    label_name: str,
    relevance_mapping: dict[str, bool],
) -> dict[str, Any]:
    """Normalize and validate one human annotation for benchmark release."""
    relevance = parse_bool(
        row.get("relevance"),
        f"{label_name} at {row.get('pair_key')!r}",
    )

    reason = row.get("reason", "").strip()
    if not reason:
        fail(f"Missing human reason for {label_name} at {row.get('pair_key')!r}.")
    if reason not in relevance_mapping:
        raise ValueError(
            f"Invalid human reason for {label_name}: {reason!r}"
        )

    derived = relevance_mapping[reason]
    if derived != relevance:
        fail(
            "Inconsistent human reason/relevance for "
            f"{label_name} at {row.get('pair_key')!r}: "
            f"{reason!r} -> {derived}, stored {relevance}."
        )

    def clean(value: Any) -> Any:
        """Convert empty optional CSV fields to null."""
        if value is None:
            return None
        text = str(value)
        return text if text != "" else None

    return {
        "timestamp_utc": clean(row.get("timestamp_utc")),
        "annotator": clean(row.get("annotator")),
        "main_resource": clean(row.get("main_resource")),
        "scope": clean(row.get("scope")),
        "supporting_field": clean(row.get("supporting_field")),
        "quote": clean(row.get("quote")),
        "reason": reason,
        "relevance": relevance,
        "notes": clean(row.get("notes")),
    }


def parse_human_label_arg(value: str) -> tuple[str, Path]:
    """Parse one NAME=PATH human-annotation argument."""
    if "=" not in value:
        fail(f"--human-label must use NAME=PATH format: {value!r}")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        fail(f"--human-label must use NAME=PATH format: {value!r}")
    return name, Path(path)


def load_stage3(
    llm_annotations_dir: Path,
    configured_models: list[str],
    default_model: str,
    relevance_mapping: dict[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and validate Stage 3 records for benchmark assembly."""
    if not llm_annotations_dir.is_dir():
        fail(
            f"Stage 3 annotation directory does not exist: "
            f"{llm_annotations_dir}"
        )

    records: list[dict[str, Any]] = []
    source_evidence: dict[str, dict[str, Any]] = {}
    seen_pair_keys: set[str] = set()

    files = list_query_json_files(llm_annotations_dir)
    configured_model_set = set(configured_models)

    for path in files:
        data = load_json_object(path)
        for required_key in ("id_query", "query", "hits"):
            if required_key not in data:
                fail(f"Missing {required_key!r} in {path}")
        filename_query_id = extract_query_number(path.name)
        id_query = require_positive_int(
            data["id_query"],
            f"{path}: id_query",
        )
        if id_query != filename_query_id:
            fail(f"Filename/query ID mismatch in {path}: {filename_query_id} != {id_query}")
        query = data["query"]
        if not isinstance(query, str) or not query.strip():
            fail(f"Invalid query text in {path}: {query!r}")
        hits = data["hits"]
        if not isinstance(hits, list):
            fail(f"Invalid hits list in {path}")

        seen_dataset_ids: set[str] = set()
        for hit_index, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                fail(f"Invalid hit #{hit_index} in {path}")
            missing_fields = [field for field in DATASET_FIELDS if field not in hit]
            if missing_fields:
                fail(
                    f"Missing fields in {path}, pair id {hit.get('id')!r}: "
                    f"{missing_fields}"
                )

            id_dataset = parse_dataset_id(hit["id"], f"{path}, hit {hit_index}")
            if id_dataset in seen_dataset_ids:
                fail(f"Duplicate dataset ID {id_dataset!r} in {path}")
            seen_dataset_ids.add(id_dataset)
            pair_key = make_pair_key(id_query, id_dataset)
            if pair_key in seen_pair_keys:
                fail(f"Duplicate query-dataset pair: {pair_key}")
            seen_pair_keys.add(pair_key)

            annotations = hit["annotations"]
            if not isinstance(annotations, dict):
                fail(f"Annotations are not an object at {pair_key}")
            missing_models = [
                model for model in configured_models if model not in annotations
            ]
            if missing_models:
                fail(f"Missing models at {pair_key}: {missing_models}")
            extra_models = [
                model for model in annotations if model not in configured_model_set
            ]
            if extra_models:
                fail(f"Unexpected models at {pair_key}: {extra_models}")

            for model in configured_models:
                validate_annotation(
                    model,
                    annotations[model],
                    pair_key,
                    relevance_mapping,
                )

            # Invalid annotations from non-default judges are preserved, but the model
            # supplying the benchmark's default relevance label must be valid.
            default_annotation = annotations[default_model]
            if not isinstance(default_annotation, dict) or "error" in default_annotation:
                fail(f"Default model annotation is invalid at {pair_key}")
            _, default_relevance = annotation_reason_and_relevance(
                default_annotation,
                relevance_mapping,
            )

            if default_relevance is None:
                fail(f"Default model annotation is invalid at {pair_key}")

            source_evidence[pair_key] = {
                "retrieval": hit["retrieval"],
                "annotations": annotations,
            }
            records.append(
                {
                    "pair_key": pair_key,
                    "id_query": id_query,
                    "query": query,
                    "id_dataset": id_dataset,
                    "title": hit["title"],
                    "description": hit["description"],
                    "header": hit["header"],
                    "content": hit["content"],
                    "metadato_fileName": hit["metadato_fileName"],
                    "resource_fileName": hit["resource_fileName"],
                    "retrieval": hit["retrieval"],
                    "annotations": annotations,
                    "default_relevance_source": default_model,
                    "default_relevance": default_relevance,
                    "human_audit": None,
                }
            )

    return records, source_evidence


def load_human_labels(
    label_args: list[str],
    relevance_mapping: dict[str, bool],
    expected_annotators: int,
) -> dict[str, dict[str, Any]]:
    """Load complete human annotations grouped by query-dataset pair."""
    if len(label_args) != expected_annotators:
        fail(
            "Unexpected number of human annotators: "
            f"{len(label_args)} != {expected_annotators}"
        )

    labels: dict[str, dict[str, Any]] = {}
    label_names: set[str] = set()

    for arg in label_args:
        label_name, path = parse_human_label_arg(arg)
        if label_name in label_names:
            fail(f"Duplicate human label name: {label_name}")
        label_names.add(label_name)

        if not path.is_file():
            fail(f"Human label file does not exist: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"pair_key", "id_query", "id_dataset", "reason", "relevance"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                fail(f"Missing columns in {path}: {sorted(missing)}")

            seen_in_file: set[str] = set()
            for row in reader:
                pair_key, _, _ = parse_row_pair_key(row, str(path))
                if pair_key in seen_in_file:
                    fail(f"Duplicate pair in {path}: {pair_key}")
                seen_in_file.add(pair_key)

                labels.setdefault(pair_key, {})[label_name] = (
                    normalize_human_annotation(row, label_name, relevance_mapping)
                )

    expected_label_names = set(label_names)
    for pair_key, pair_labels in labels.items():
        if set(pair_labels) != expected_label_names:
            fail(
                f"Human pair {pair_key} does not have all labels: "
                f"{sorted(pair_labels)} != {sorted(expected_label_names)}"
            )

    return labels


def load_human_majority(path: Path) -> dict[str, bool]:
    """Load binary human-majority labels keyed by query-dataset pair."""
    if not path.is_file():
        fail(f"Human majority file does not exist: {path}")

    majority: dict[str, bool] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_key", "id_query", "id_dataset", "human_majority"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            fail(f"Missing columns in {path}: {sorted(missing)}")

        for row in reader:
            pair_key, _, _ = parse_row_pair_key(row, str(path))
            if pair_key in majority:
                fail(f"Duplicate human majority pair: {pair_key}")
            value = parse_bool(row.get("human_majority"), f"human majority at {pair_key}")
            majority[pair_key] = value
    return majority


def strict_bool_majority(values: list[bool], pair_key: str) -> bool:
    """Return the strict binary majority or fail when no majority exists."""
    positive = sum(value is True for value in values)
    negative = sum(value is False for value in values)
    threshold = len(values) / 2.0
    if positive > threshold:
        return True
    if negative > threshold:
        return False
    fail(f"Human relevance votes have no strict majority at {pair_key}")


def strict_reason_majority(values: Any) -> str | None:
    """Return a reason supported by a strict majority, if one exists."""
    values = list(values)
    counts = Counter(value for value in values if value is not None)
    threshold = len(values) / 2.0
    for reason, count in counts.items():
        if count > threshold:
            return reason
    return None


def attach_human_audit(
    records: list[dict[str, Any]],
    human_labels: dict[str, dict[str, Any]],
    human_majority: dict[str, bool],
    expected_human_audit: int,
) -> None:
    """Attach validated human annotations to the sampled benchmark records."""

    # Human annotations are attached only to the frozen sampled pairs; all other
    # benchmark records retain human_audit=None.
    benchmark_keys = {record["pair_key"] for record in records}
    human_keys = set(human_labels)
    majority_keys = set(human_majority)

    unknown_label_keys = human_keys - benchmark_keys
    if unknown_label_keys:
        fail(
            "Human labels contain IDs not present in the benchmark: "
            f"{sorted(unknown_label_keys)[:10]}"
        )

    unknown_majority_keys = majority_keys - benchmark_keys
    if unknown_majority_keys:
        fail(
            "Human majority contains IDs not present in the benchmark: "
            f"{sorted(unknown_majority_keys)[:10]}"
        )

    if human_keys != majority_keys:
        fail(
            "Human labels and human majority do not cover the same pairs: "
            f"labels_only={len(human_keys - majority_keys)}, "
            f"majority_only={len(majority_keys - human_keys)}"
        )

    if len(human_keys) != expected_human_audit:
        fail(
            f"Unexpected human-audit pair count: "
            f"{len(human_keys)} != {expected_human_audit}"
        )

    for record in records:
        pair_key = record["pair_key"]
        if pair_key in human_keys:
            labels = dict(human_labels[pair_key])
            human_relevances = [
                labels[label]["relevance"]
                for label in sorted(labels)
            ]
            expected_majority = strict_bool_majority(human_relevances, pair_key)
            if expected_majority != human_majority[pair_key]:
                fail(
                    "Human majority label does not match strict-majority "
                    f"human relevance votes at {pair_key}: "
                    f"expected {expected_majority}, "
                    f"stored {human_majority[pair_key]}."
                )

            majority_reason = strict_reason_majority(
                labels[label]["reason"]
                for label in sorted(labels)
            )

            labels["majority_relevance"] = human_majority[pair_key]
            labels["majority_reason"] = majority_reason
            record["human_audit"] = labels


def validate_records(
    records,
    expected_human_audit,
    configured_models,
    default_model,
    relevance_mapping,
):
    """Validate the assembled benchmark and return release summary statistics."""
    pair_keys = [record["pair_key"] for record in records]
    duplicated = [
        key for key, count in Counter(pair_keys).items() if count > 1
    ]
    if duplicated:
        fail(f"Duplicate pair_key values: {duplicated[:10]}")

    query_count = len({record["id_query"] for record in records})

    default_source_count = sum(
        1
        for record in records
        if record.get("default_relevance_source") == default_model
    )
    if default_source_count != len(records):
        fail(
            f"Unexpected default source count: "
            f"{default_source_count} != {len(records)}"
        )

    default_positive = sum(record["default_relevance"] is True for record in records)
    default_negative = sum(record["default_relevance"] is False for record in records)
    default_missing = sum(record["default_relevance"] is None for record in records)

    human_audit_count = sum(record["human_audit"] is not None for record in records)
    if human_audit_count != expected_human_audit:
        fail(
            f"Unexpected human_audit count: "
            f"{human_audit_count} != {expected_human_audit}"
        )

    invalid_by_model = Counter()
    present_by_model = Counter()
    majority_relevance_positive = 0
    majority_relevance_negative = 0
    majority_reason_defined = 0
    majority_reason_null = 0
    for record in records:
        annotations = record["annotations"]
        if set(annotations) != set(configured_models):
            fail(f"Configured model set mismatch at {record['pair_key']}")
        for model in configured_models:
            present_by_model[model] += int(model in annotations)
            annotation = annotations[model]
            if isinstance(annotation, dict) and "error" in annotation:
                invalid_by_model[model] += 1

        human_audit = record["human_audit"]
        if human_audit is not None:
            if human_audit["majority_relevance"] is True:
                majority_relevance_positive += 1
            elif human_audit["majority_relevance"] is False:
                majority_relevance_negative += 1
            else:
                fail(f"Invalid majority_relevance at {record['pair_key']}")

            if human_audit["majority_reason"] is None:
                majority_reason_null += 1
            elif human_audit["majority_reason"] in relevance_mapping:
                majority_reason_defined += 1
            else:
                fail(f"Invalid majority_reason at {record['pair_key']}")

    return {
        "records_total": len(records),
        "unique_pair_keys": len(set(pair_keys)),
        "distinct_queries": query_count,
        "human_audit_pairs": human_audit_count,
        "non_human_audit_pairs": len(records) - human_audit_count,
        "default_relevance_source": default_model,
        "default_relevance_source_count": default_source_count,
        "default_positive": default_positive,
        "default_negative": default_negative,
        "default_missing": default_missing,
        "majority_relevance_positive": majority_relevance_positive,
        "majority_relevance_negative": majority_relevance_negative,
        "majority_reason_defined": majority_reason_defined,
        "majority_reason_null": majority_reason_null,
        "llm_annotation_presence": dict(present_by_model),
        "llm_invalid_annotation_counts": dict(invalid_by_model),
    }


def write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write benchmark records atomically as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=False,
                    )
                    + "\n"
                )

        temporary_path.replace(output_path)

    except Exception:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise


def validate_stage3_equivalence(
    source_by_pair: dict[str, dict[str, Any]],
    benchmark_path: Path,
) -> None:
    """Verify that Stage 3 retrieval provenance and annotations are unchanged."""

    # Stage 6 may add release-level fields, but retrieval provenance and raw LLM
    # annotations must remain identical to their Stage 3 source records.
    output_pair_keys: set[str] = set()
    with benchmark_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pair_key = record["pair_key"]
            if pair_key in output_pair_keys:
                fail(f"Duplicate output pair at line {line_number}: {pair_key}")
            output_pair_keys.add(pair_key)
            source = source_by_pair.get(pair_key)
            if source is None:
                fail(f"Output pair not found in Stage 3 at line {line_number}: {pair_key}")
            if record["retrieval"] != source["retrieval"]:
                fail(f"Retrieval changed for {pair_key}")
            if record["annotations"] != source["annotations"]:
                fail(f"Annotations changed for {pair_key}")
    if output_pair_keys != set(source_by_pair):
        fail(
            "Output pair set does not match Stage 3 source pair set: "
            f"source_only={len(set(source_by_pair) - output_pair_keys)}, "
            f"output_only={len(output_pair_keys - set(source_by_pair))}"
        )


def maybe_copy_judge_aggregations(source: Path | None, destination: Path) -> bool:
    """Optionally copy the Stage 4 aggregation artifact."""
    if source is None:
        return False

    if not source.is_file():
        fail(f"Judge aggregation source is not a file: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if (
        destination.exists()
        and source.resolve() == destination.resolve()
    ):
        return False

    temporary_path = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    except Exception:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise

    return True


def print_assembly_summary(
    summary,
    models,
    output_path,
    copied_aggregations,
    aggregations_output,
):
    """Print summary statistics for the assembled benchmark."""
    print("Canonical benchmark assembled successfully.")
    print(f"Output: {output_path}")
    print(f"Records total: {summary['records_total']}")
    print(f"Unique pair_keys: {summary['unique_pair_keys']}")
    print(f"Distinct queries: {summary['distinct_queries']}")
    print(f"Human-audit pairs: {summary['human_audit_pairs']}")
    print(f"Non-human-audit pairs: {summary['non_human_audit_pairs']}")
    print(
        "Default relevance source: "
        f"{summary['default_relevance_source']} "
        f"({summary['default_relevance_source_count']} records)"
    )
    print(f"Default positives: {summary['default_positive']}")
    print(f"Default negatives: {summary['default_negative']}")
    print(f"Default missing: {summary['default_missing']}")
    print(
        "Human majority relevance positives: "
        f"{summary['majority_relevance_positive']}"
    )
    print(
        "Human majority relevance negatives: "
        f"{summary['majority_relevance_negative']}"
    )
    print(
        "Human majority reason defined: "
        f"{summary['majority_reason_defined']}"
    )
    print(
        "Human majority reason null: "
        f"{summary['majority_reason_null']}"
    )

    print("LLM annotation presence:")
    for model in models:
        print(
            f" - {model}: "
            f"{summary['llm_annotation_presence'].get(model, 0)}"
        )

    print("Invalid historical annotations preserved:")
    for model in models:
        print(
            f" - {model}: "
            f"{summary['llm_invalid_annotation_counts'].get(model, 0)}"
        )

    print("Stage 3 retrieval and annotations equivalence: verified")

    if copied_aggregations:
        print(f"Judge aggregations copied to: {aggregations_output}")

    
def main() -> None:
    """Run Stage 6 canonical benchmark assembly."""
    args = parse_args()

    benchmark_config = load_benchmark_config(args.benchmark_config)
    assembly_config = assembly_settings_from_config(benchmark_config)

    registered_models = set(load_model_ids(args.models_config))
    models = assembly_config["models"]
    expected_human_annotators = assembly_config["human_annotators"]

    unknown_models = [
        model
        for model in models
        if model not in registered_models
    ]

    if unknown_models:
        raise ValueError(
            "Assembly judge(s) are not present in models configuration: "
            f"{unknown_models}"
        )

    default_model = assembly_config["default_relevance_model"]
    expected_human_audit = assembly_config["human_audit_sample_size"]
    relevance_mapping = assembly_config["relevance_mapping"]

    human_label_args = list(
        args.human_label or DEFAULT_HUMAN_ANNOTATIONS
    )

    records, source_evidence = load_stage3(
        args.llm_annotations,
        models,
        default_model,
        relevance_mapping,
    )
    human_labels = load_human_labels(human_label_args, relevance_mapping, expected_human_annotators)
    human_majority = load_human_majority(args.human_majority)
    attach_human_audit(
        records,
        human_labels,
        human_majority,
        expected_human_audit,
    )
    summary = validate_records(
        records,
        expected_human_audit,
        models,
        default_model,
        relevance_mapping,
    )

    # Publish the assembled benchmark only after all source and human-label
    # consistency checks have succeeded.
    write_jsonl(records, args.output)
    validate_stage3_equivalence(source_evidence, args.output)
    copied_aggregations = maybe_copy_judge_aggregations(
        args.copy_judge_aggregations_from,
        args.judge_aggregations_output,
    )

    print_assembly_summary(
        summary=summary,
        models=models,
        output_path=args.output,
        copied_aggregations=copied_aggregations,
        aggregations_output=args.judge_aggregations_output,
    )

if __name__ == "__main__":
    main()
