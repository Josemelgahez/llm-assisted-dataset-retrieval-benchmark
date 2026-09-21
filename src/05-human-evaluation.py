#!/usr/bin/env python3
"""Evaluate human agreement and LLM judgments on the frozen human sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from itertools import combinations
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
    write_csv_atomic,
    write_json_atomic,
)

DEFAULT_HUMAN_ANNOTATIONS = (
    "h1=data/human_annotations/annotator_1.csv",
    "h2=data/human_annotations/annotator_2.csv",
    "h3=data/human_annotations/annotator_3.csv",
)


def parse_human_bool(value: Any) -> bool:
    """Parse a strict Boolean value from a human annotation field."""
    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        raise ValueError(f"Invalid Boolean value: {value!r}")

    text = value.strip().lower()

    if text == "true":
        return True
    if text == "false":
        return False

    raise ValueError(f"Invalid Boolean value: {value!r}")


def parse_aggregation_bool(value: Any, context: str) -> bool | None:
    """Parse a Stage 4 aggregate relevance value, preserving missing labels."""
    if value is None:
        raise ValueError(f"{context}: missing aggregated_relevance.")
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{context}: invalid aggregated_relevance {value!r}.")
    text = value.strip()
    if text == "":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    raise ValueError(f"{context}: invalid aggregated_relevance {value!r}.")


def human_reason_and_relevance(
    row: dict[str, Any],
    relevance_mapping: dict[str, bool],
) -> tuple[str, bool]:
    """Validate one human reason/relevance judgment against the protocol."""
    if "relevance" not in row:
        raise ValueError("Missing human relevance value.")

    relevance = parse_human_bool(row.get("relevance"))
    reason = str(row.get("reason", "")).strip()

    if reason not in relevance_mapping:
        raise ValueError(f"Invalid human reason: {reason!r}")

    derived = relevance_mapping[reason]

    if relevance != derived:
        raise ValueError(
            "Human relevance is inconsistent with its reason: "
            f"reason={reason!r}, relevance={relevance!r}"
        )

    return reason, relevance


def load_sample(path: Path) -> list[dict[str, Any]]:
    """Load and validate the frozen query-dataset sample used for human evaluation."""
    if not path.is_file():
        raise FileNotFoundError(f"Human sample does not exist: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Human sample CSV has no header: {path}")
            required = {"id_query", "id_dataset"}
            if not required.issubset(reader.fieldnames):
                raise ValueError(f"Human sample CSV must contain {sorted(required)}.")
            rows = list(reader)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Invalid sample row at {path}:{line_number}: expected object."
                        )
                    rows.append(row)
    else:
        raise ValueError("Human sample must be CSV or JSONL.")

    sample_rows = []
    for file_order, row in enumerate(rows):
        context = f"{path}, row {file_order + 2 if path.suffix.lower() == '.csv' else file_order + 1}"
        pair_key, query_id, dataset_id = parse_row_pair_key(row, context)

        order_value = row.get("order")
        if order_value is None or str(order_value).strip() == "":
            order = file_order
        else:
            try:
                order = int(str(order_value).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}: invalid order {order_value!r}.") from exc

        sample_rows.append(
            {
                "order": order,
                "pair_key": pair_key,
                "id_query": str(query_id),
                "id_dataset": dataset_id,
            }
        )

    sample_rows.sort(key=lambda row: row["order"])
    keys = [row["pair_key"] for row in sample_rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate pairs in sample: {duplicates[:10]}")
    orders = [row["order"] for row in sample_rows]
    duplicate_orders = [order for order, count in Counter(orders).items() if count > 1]
    if duplicate_orders:
        raise ValueError(f"Duplicate sample order values: {duplicate_orders[:10]}")
    return sample_rows


def parse_human_label_arguments(values: list[str]) -> dict[str, Path]:
    """Parse repeated NAME=PATH human-annotation arguments."""
    human_files: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--human-label must use NAME=PATH format.")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("--human-label must use NAME=PATH format.")
        if name in human_files:
            raise ValueError(f"Human annotator specified more than once: {name}")
        human_files[name] = Path(path)

    if len(human_files) < 2:
        raise ValueError("At least two human annotators are required.")
    return human_files


def load_human_labels(
    human_files: dict[str, Path],
    sample_rows: list[dict[str, Any]],
    relevance_mapping: dict[str, bool],
) -> tuple[dict[str, list[bool]], dict[str, list[str]]]:
    """Load complete human judgments aligned to the frozen sample order."""
    sample_keys = [row["pair_key"] for row in sample_rows]
    sample_key_set = set(sample_keys)
    labels: dict[str, list[bool]] = {}
    reasons: dict[str, list[str]] = {}

    for human_name, path in human_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Human annotation file does not exist: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Human annotation CSV has no header: {path}")
            required = {"id_query", "id_dataset", "reason", "relevance"}
            if not required.issubset(reader.fieldnames):
                raise ValueError(f"{path}: missing required columns {sorted(required)}.")
            rows = list(reader)

        judgments: dict[str, tuple[str, bool]] = {}
        for row_number, row in enumerate(rows, start=2):
            pair_key, _, _ = parse_row_pair_key(row, f"{path}, row {row_number}")
            if pair_key in judgments:
                raise ValueError(f"{path}: duplicate human judgment for {pair_key}")
            judgments[pair_key] = human_reason_and_relevance(row, relevance_mapping)

        missing = sample_key_set - set(judgments)
        extra = set(judgments) - sample_key_set
        # Every human annotator must judge exactly the same frozen sample so all
        # agreement and majority calculations remain pair-aligned.
        if missing:
            raise ValueError(f"{path}: missing {len(missing)} sample pair(s).")
        if extra:
            raise ValueError(f"{path}: contains {len(extra)} pair(s) outside sample.")

        reasons[human_name] = [judgments[pair_key][0] for pair_key in sample_keys]
        labels[human_name] = [judgments[pair_key][1] for pair_key in sample_keys]

    return labels, reasons


def load_llm_labels_and_reasons(
    input_dir: Path,
    sample_rows: list[dict[str, Any]],
    requested_models: list[str],
    relevance_mapping: dict[str, bool],
) -> tuple[dict[str, list[bool | None]], dict[str, list[str | None]]]:
    """Load LLM judgments for the frozen human-evaluation sample."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"LLM annotation directory does not exist: {input_dir}")

    sample_keys = [row["pair_key"] for row in sample_rows]
    sample_key_set = set(sample_keys)
    pair_annotations: dict[str, dict[str, Any]] = {}
    all_pair_keys: set[str] = set()
    observed_models: set[str] = set()

    input_files = list_query_json_files(input_dir)

    for input_path in input_files:
        payload = load_json_object(input_path)
        filename_query_id = extract_query_number(input_path.name)
        for required_key in ("id_query", "query", "hits"):
            if required_key not in payload:
                raise ValueError(f"{input_path}: missing {required_key!r}.")
        query_id = require_positive_int(
            payload["id_query"],
            f"{input_path}: id_query",
        )
        if query_id != filename_query_id:
            raise ValueError(
                f"{input_path}: filename ID {filename_query_id} != id_query {query_id}."
            )
        query = payload["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{input_path}: invalid query text {query!r}.")
        hits = payload["hits"]
        if not isinstance(hits, list):
            raise ValueError(f"'hits' must be a list in {input_path}")
        seen_dataset_ids: set[str] = set()
        for hit_index, hit in enumerate(hits, start=1):
            context = f"{input_path}, hit {hit_index}"
            if not isinstance(hit, dict):
                raise ValueError(f"{context}: hit must be an object.")
            if "id" not in hit:
                raise ValueError(f"{context}: missing dataset ID.")
            dataset_id = parse_dataset_id(hit["id"], context)
            if dataset_id in seen_dataset_ids:
                raise ValueError(f"{input_path}: duplicate dataset ID {dataset_id!r}.")
            seen_dataset_ids.add(dataset_id)
            pair_key = make_pair_key(query_id, dataset_id)
            if pair_key in all_pair_keys:
                raise ValueError(f"Duplicate LLM annotation entry for {pair_key}")
            all_pair_keys.add(pair_key)
            if "annotations" not in hit:
                raise ValueError(f"{context}: missing annotations.")
            annotations = hit["annotations"]
            if not isinstance(annotations, dict):
                raise ValueError(f"Invalid annotations object for {pair_key}")
            if pair_key not in sample_key_set:
                continue
            pair_annotations[pair_key] = annotations
            observed_models.update(annotations.keys())

    missing_pairs = sample_key_set - set(pair_annotations)
    if missing_pairs:
        raise ValueError(
            f"LLM annotations are missing {len(missing_pairs)} sample pair(s)."
        )

    missing_models = [model for model in requested_models if model not in observed_models]
    if missing_models:
        raise ValueError(f"Requested LLM judges were not found: {missing_models}")

    labels = {model: [] for model in requested_models}
    reasons = {model: [] for model in requested_models}
    # Preserve the frozen sample order across every model so metric vectors refer
    # to exactly the same query-dataset pairs.
    for pair_key in sample_keys:
        annotations = pair_annotations[pair_key]
        for model in requested_models:
            reason, relevance = annotation_reason_and_relevance(
                annotations.get(model),
                relevance_mapping,
            )
            reasons[model].append(reason)
            labels[model].append(relevance)

    return labels, reasons


def load_aggregation_labels(
    path: Path,
    sample_rows: list[dict[str, Any]],
) -> dict[str, list[bool | None]]:
    """Load Stage 4 aggregate labels aligned to the frozen human sample."""
    if not path.is_file():
        raise FileNotFoundError(f"Judge aggregation file does not exist: {path}")

    sample_keys = [row["pair_key"] for row in sample_rows]
    sample_key_set = set(sample_keys)
    values_by_combination: dict[str, dict[str, bool | None]] = {}
    seen_combination_pairs: set[tuple[str, str]] = set()

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Judge aggregation CSV has no header: {path}")
        required = {
            "pair_key",
            "id_query",
            "id_dataset",
            "combination",
            "aggregated_relevance",
            "valid_votes",
        }
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(required)}.")

        for row_number, row in enumerate(reader, start=2):
            context = f"{path}, row {row_number}"
            combination = str(row.get("combination", "")).strip()
            if not combination:
                raise ValueError(f"{context}: missing combination.")

            pair_key, _, _ = parse_row_pair_key(row, context)
            duplicate_key = (combination, pair_key)
            if duplicate_key in seen_combination_pairs:
                raise ValueError(f"{path}: duplicate aggregation for {combination}, {pair_key}")
            seen_combination_pairs.add(duplicate_key)

            try:
                valid_votes = int(str(row.get("valid_votes")).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}: invalid valid_votes {row.get('valid_votes')!r}.") from exc
            if valid_votes < 0 or valid_votes > 3:
                raise ValueError(f"{context}: valid_votes must be between 0 and 3.")

            aggregated_relevance = parse_aggregation_bool(
                row.get("aggregated_relevance"),
                context,
            )
            if aggregated_relevance is None and valid_votes == 3:
                raise ValueError(f"{context}: missing aggregate with valid_votes == 3.")
            if aggregated_relevance is not None and valid_votes != 3:
                raise ValueError(f"{context}: aggregate is present but valid_votes != 3.")

            if pair_key not in sample_key_set:
                continue

            combination_values = values_by_combination.setdefault(combination, {})
            combination_values[pair_key] = aggregated_relevance

    if not values_by_combination:
        raise ValueError(f"No aggregation rows for the sample in {path}")

    output = {}
    for combination, values in values_by_combination.items():
        missing_pairs = sample_key_set - set(values)
        if missing_pairs:
            raise ValueError(
                f"Combination {combination!r} is missing {len(missing_pairs)} sample pair(s)."
            )
        output[combination] = [values[pair_key] for pair_key in sample_keys]
    return output


def cohen_kappa(labels_a: list[Any], labels_b: list[Any]) -> dict[str, float | int | None]:
    """Compute Cohen's kappa over pairs with two available judgments."""
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}

    observed = sum(a == b for a, b in pairs) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else 0.0
    else:
        kappa = (observed - expected) / denominator
    return {"n": n, "agreement": observed, "kappa": kappa}


def fleiss_kappa(items: list[list[Any]]) -> float | None:
    """Compute Fleiss' kappa for all categories observed in usable items."""
    usable_items = [item for item in items if all(value is not None for value in item)]
    if not usable_items:
        return None

    n_items = len(usable_items)
    n_raters = len(usable_items[0])
    if n_raters < 2:
        return None

    category_totals: Counter[Any] = Counter()
    item_agreements = []
    for item in usable_items:
        counts = Counter(item)
        category_totals.update(counts)
        agreement = (
            sum(count * count for count in counts.values()) - n_raters
        ) / (n_raters * (n_raters - 1))
        item_agreements.append(agreement)

    observed = sum(item_agreements) / n_items
    expected = sum(
        (count / float(n_items * n_raters)) ** 2
        for count in category_totals.values()
    )
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / denominator


def exact_agreement(label_sets: list[list[Any]]) -> float | None:
    """Compute the fraction of items with complete unanimous agreement."""
    if not label_sets or not label_sets[0]:
        return None
    agreeing = 0
    for index in range(len(label_sets[0])):
        values = [labels[index] for labels in label_sets]
        if all(value is not None for value in values) and len(set(values)) == 1:
            agreeing += 1
    return agreeing / len(label_sets[0])


def majority_vote(values: list[bool | None]) -> bool | None:
    """Return the strict binary majority among available judgments."""
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    positive = sum(value is True for value in valid_values)
    negative = sum(value is False for value in valid_values)
    if positive > negative:
        return True
    if negative > positive:
        return False
    return None


def majority_reason(values: list[str]) -> str | None:
    """Return the reason label supported by a strict majority, if any."""
    counts = Counter(values)
    threshold = len(values) / 2.0
    for reason, count in counts.items():
        if count > threshold:
            return reason
    return None


def build_human_majority(human_labels: dict[str, list[bool]]) -> list[bool]:
    """Build the complete human-majority binary reference."""
    humans = list(human_labels)
    n = len(human_labels[humans[0]])
    majority = []
    for index in range(n):
        result = majority_vote([human_labels[human][index] for human in humans])
        if result is None:
            raise ValueError(f"Human-majority reference missing at index {index}.")
        majority.append(result)
    return majority


def write_human_majority_labels(
    sample_rows: list[dict[str, Any]],
    human_labels: dict[str, list[bool]],
    human_majority: list[bool],
    output_path: Path,
) -> None:
    """Write pair-level human labels and their derived majority label."""
    if len(human_majority) != len(sample_rows):
        raise ValueError(
            "Human-majority labels are not aligned with the frozen sample."
        )

    human_names = list(human_labels)

    for human_name in human_names:
        if len(human_labels[human_name]) != len(sample_rows):
            raise ValueError(
                f"Human labels for {human_name!r} are not aligned "
                "with the frozen sample."
            )

    rows = []

    for index, sample_row in enumerate(sample_rows):
        row = {
            "pair_key": sample_row["pair_key"],
            "id_query": sample_row["id_query"],
            "id_dataset": sample_row["id_dataset"],
        }

        for human_name in human_names:
            row[human_name] = human_labels[human_name][index]

        row["human_majority"] = human_majority[index]
        rows.append(row)

    fieldnames = [
        "pair_key",
        "id_query",
        "id_dataset",
        *human_names,
        "human_majority",
    ]

    write_csv_atomic(
        rows,
        output_path,
        fieldnames,
    )


def binary_metrics(reference: list[bool], prediction: list[bool | None]) -> dict[str, Any]:
    """Compute binary metrics over pairs with an available prediction."""
    # Missing model predictions are excluded rather than coerced to either class.
    pairs = [
        (reference_value, predicted_value)
        for reference_value, predicted_value in zip(reference, prediction)
        if reference_value is not None and predicted_value is not None
    ]
    tp = sum(ref is True and pred is True for ref, pred in pairs)
    tn = sum(ref is False and pred is False for ref, pred in pairs)
    fp = sum(ref is False and pred is True for ref, pred in pairs)
    fn = sum(ref is True and pred is False for ref, pred in pairs)
    n = len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / n if n else None
    kappa = cohen_kappa(reference, prediction)["kappa"]
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "kappa": kappa,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def subset(values: list[Any], indices: list[int]) -> list[Any]:
    """Select values at the supplied bootstrap indices."""
    return [values[index] for index in indices]


def percentile(values: list[float | None], quantile: float) -> float | None:
    """Compute an interpolated percentile after removing missing values."""
    clean_values = sorted(
        value
        for value in values
        if value is not None
        and not (isinstance(value, float) and math.isnan(value))
    )
    if not clean_values:
        return None
    position = (len(clean_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean_values[lower]
    return clean_values[lower] + (clean_values[upper] - clean_values[lower]) * (
        position - lower
    )


def generate_bootstrap_samples(n: int, iterations: int, seed: int) -> list[list[int]]:
    """Generate reproducible paired bootstrap index samples."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(iterations)]


def bootstrap_ci(metric_function, bootstrap_samples: list[list[int]]) -> tuple[float | None, float | None]:
    """Compute a percentile bootstrap confidence interval."""
    values = [metric_function(indices) for indices in bootstrap_samples]
    return percentile(values, 0.025), percentile(values, 0.975)


def mean_pairwise_human_kappa(
    prediction: list[bool | None],
    human_labels: dict[str, list[bool]],
) -> float | None:
    """Compute mean Cohen's kappa between one prediction and all humans."""
    kappas = []
    for human_values in human_labels.values():
        kappa = cohen_kappa(prediction, human_values)["kappa"]
        if kappa is None:
            return None
        kappas.append(kappa)
    return sum(kappas) / len(kappas) if kappas else None


def analyze_human_agreement(
    human_labels: dict[str, list[bool]],
    bootstrap_samples: list[list[int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute pairwise and multi-rater human agreement statistics."""
    humans = list(human_labels)
    pairwise_rows = []
    for human_a, human_b in combinations(humans, 2):
        values_a = human_labels[human_a]
        values_b = human_labels[human_b]
        result = cohen_kappa(values_a, values_b)
        ci = bootstrap_ci(
            lambda indices: cohen_kappa(subset(values_a, indices), subset(values_b, indices))["kappa"],
            bootstrap_samples,
        )
        pairwise_rows.append(
            {
                "annotator_a": human_a,
                "annotator_b": human_b,
                "n": result["n"],
                "observed_agreement": result["agreement"],
                "cohen_kappa": result["kappa"],
                "cohen_kappa_ci_low": ci[0],
                "cohen_kappa_ci_high": ci[1],
            }
        )

    n = len(next(iter(human_labels.values())))
    human_items = [[human_labels[human][index] for human in humans] for index in range(n)]
    fleiss = fleiss_kappa(human_items)
    fleiss_ci = bootstrap_ci(
        lambda indices: fleiss_kappa([human_items[index] for index in indices]),
        bootstrap_samples,
    )
    exact = exact_agreement([human_labels[human] for human in humans])
    exact_ci = bootstrap_ci(
        lambda indices: exact_agreement(
            [subset(human_labels[human], indices) for human in humans]
        ),
        bootstrap_samples,
    )
    return pairwise_rows, {
        "fleiss_kappa": fleiss,
        "fleiss_kappa_ci_low": fleiss_ci[0],
        "fleiss_kappa_ci_high": fleiss_ci[1],
        "exact_all_human_agreement": exact,
        "exact_all_human_agreement_ci_low": exact_ci[0],
        "exact_all_human_agreement_ci_high": exact_ci[1],
    }


def analyze_predictions_vs_humans(
    predictions: dict[str, list[bool | None]],
    human_labels: dict[str, list[bool]],
    bootstrap_samples: list[list[int]],
    name_column: str,
) -> list[dict[str, Any]]:
    """Compare each prediction source independently with every human annotator."""
    humans = list(human_labels)
    rows = []
    for name, values in predictions.items():
        row = {name_column: name, "n_valid": sum(value is not None for value in values)}
        kappas = []
        for human in humans:
            kappa = cohen_kappa(values, human_labels[human])["kappa"]
            row[f"kappa_{human.lower()}"] = kappa
            kappas.append(kappa)
        row["mean_pairwise_cohen_kappa"] = (
            sum(kappas) / len(kappas)
            if kappas and all(value is not None for value in kappas)
            else None
        )
        ci = bootstrap_ci(
            lambda indices: mean_pairwise_human_kappa(
                subset(values, indices),
                {human: subset(human_labels[human], indices) for human in humans},
            ),
            bootstrap_samples,
        )
        row["mean_pairwise_cohen_kappa_ci_low"] = ci[0]
        row["mean_pairwise_cohen_kappa_ci_high"] = ci[1]
        rows.append(row)
    return rows


def analyze_predictions_vs_majority(
    predictions: dict[str, list[bool | None]],
    human_majority: list[bool],
    bootstrap_samples: list[list[int]],
    name_column: str,
) -> list[dict[str, Any]]:
    """Evaluate each prediction source against the human-majority reference."""
    rows = []
    for name, values in predictions.items():
        row = {name_column: name, **binary_metrics(human_majority, values)}
        for metric_name in ("kappa", "precision", "recall", "f1"):
            ci = bootstrap_ci(
                lambda indices, metric_name=metric_name: binary_metrics(
                    subset(human_majority, indices),
                    subset(values, indices),
                )[metric_name],
                bootstrap_samples,
            )
            row[f"{metric_name}_ci_low"] = ci[0]
            row[f"{metric_name}_ci_high"] = ci[1]
        rows.append(row)
    return rows


def build_reason_analysis(
    human_reasons: dict[str, list[str]],
    human_labels: dict[str, list[bool]],
    human_majority: list[bool],
    bootstrap_samples: list[list[int]],
    relevance_mapping: dict[str, bool],
) -> dict[str, Any]:
    """Analyze agreement and disagreement patterns among human reason labels."""
    humans = list(human_reasons)
    n = len(next(iter(human_reasons.values())))
    reason_items = [[human_reasons[human][index] for human in humans] for index in range(n)]

    unanimous_reason_pairs = 0
    majority_defined = 0
    majority_undefined = 0
    non_unanimous_reason_but_unanimous_binary = 0
    direct_match_subtype_disagreement_pairs = 0

    for index, reasons in enumerate(reason_items):
        relevances = [relevance_mapping[reason] for reason in reasons]
        if majority_vote(relevances) != human_majority[index]:
            raise ValueError(f"Human majority mismatch at sample index {index}.")
        if majority_vote([human_labels[human][index] for human in humans]) != human_majority[index]:
            raise ValueError(f"Human label majority mismatch at sample index {index}.")

        reason_unanimous = len(set(reasons)) == 1
        binary_unanimous = len(set(relevances)) == 1
        if reason_unanimous:
            unanimous_reason_pairs += 1
        elif binary_unanimous:
            non_unanimous_reason_but_unanimous_binary += 1

        if majority_reason(reasons) is None:
            majority_undefined += 1
        else:
            majority_defined += 1

        if not reason_unanimous and "direct_match" in reasons and "subtype" in reasons:
            direct_match_subtype_disagreement_pairs += 1

    kappa = fleiss_kappa(reason_items)
    ci = bootstrap_ci(
        lambda indices: fleiss_kappa([reason_items[index] for index in indices]),
        bootstrap_samples,
    )
    return {
        "pairs": n,
        "fleiss_kappa_reason": kappa,
        "fleiss_kappa_reason_ci_low": ci[0],
        "fleiss_kappa_reason_ci_high": ci[1],
        "unanimous_reason_pairs": unanimous_reason_pairs,
        "unanimous_reason_rate": unanimous_reason_pairs / float(n),
        "majority_reason_defined": majority_defined,
        "majority_reason_undefined": majority_undefined,
        "non_unanimous_reason_but_unanimous_binary": non_unanimous_reason_but_unanimous_binary,
        "direct_match_subtype_disagreement_pairs": direct_match_subtype_disagreement_pairs,
    }


def build_false_negative_rows(
    llm_labels: dict[str, list[bool | None]],
    llm_reasons: dict[str, list[str | None]],
    human_reasons: dict[str, list[str]],
    human_majority: list[bool],
    individual_vs_majority: list[dict[str, Any]],
    reason_labels: list[str],
    relevance_mapping: dict[str, bool],
) -> list[dict[str, Any]]:
    """Summarize reason patterns among LLM false negatives."""

    # False-negative reason analysis is restricted to pairs considered relevant by
    # the human-majority reference but labeled non-relevant by the LLM judge.
    negative_reasons = {
        reason
        for reason in reason_labels
        if relevance_mapping[reason] is False
    }

    expected_fn = {row["judge"]: int(row["fn"]) for row in individual_vs_majority}
    humans = list(human_reasons)
    n = len(human_majority)
    human_majority_reasons = [
        majority_reason([human_reasons[human][index] for human in humans])
        for index in range(n)
    ]

    rows = []
    for model in llm_labels:
        false_negatives_total = 0
        false_negatives_with_majority_reason = 0
        human_reason_counts: Counter[str] = Counter()
        model_reason_counts: Counter[str] = Counter()

        for index in range(n):
            if human_majority[index] is not True:
                continue
            if llm_labels[model][index] is not False:
                continue

            false_negatives_total += 1
            model_reason = llm_reasons[model][index]
            if model_reason not in negative_reasons:
                raise ValueError(f"False negative has invalid model reason: {model}, index {index}")

            human_reason = human_majority_reasons[index]
            if human_reason is None:
                continue
            if human_reason not in reason_labels:
                raise ValueError(f"Invalid human majority reason at index {index}: {human_reason!r}")

            false_negatives_with_majority_reason += 1
            human_reason_counts[human_reason] += 1
            model_reason_counts[model_reason] += 1

        if model in expected_fn and false_negatives_total != expected_fn[model]:
            raise ValueError(
                f"{model} false_negatives_total mismatch: "
                f"{false_negatives_total} != {expected_fn[model]}"
            )

        row = {
            "model": model,
            "false_negatives_total": false_negatives_total,
            "false_negatives_with_majority_reason": false_negatives_with_majority_reason,
        }
        for reason in reason_labels:
            row[f"human_majority_{reason}"] = human_reason_counts[reason]
            row[f"model_reason_{reason}"] = model_reason_counts[reason]
        row["generic_among_fn_with_majority_reason"] = model_reason_counts["generic"]
        row["generic_rate_among_fn_with_majority_reason"] = (
            model_reason_counts["generic"] / float(false_negatives_with_majority_reason)
            if false_negatives_with_majority_reason
            else 0.0
        )
        rows.append(row)
    return rows


def evaluation_settings_from_config(config):
    """Read Stage 5 evaluation settings from the benchmark config."""
    llm_config = config["llm_annotation"]
    human_config = config["human_audit"]
    bootstrap_config = config["evaluation"]["bootstrap"]

    return {
        "models": [model.strip() for model in llm_config["models"]],
        "reason_labels": [reason.strip() for reason in llm_config["reason_labels"]],
        "relevance_mapping": dict(llm_config["relevance_mapping"]),
        "sample_size": human_config["sample_size"],
        "human_annotators": human_config["annotators"],
        "bootstrap_iterations": bootstrap_config["iterations"],
        "bootstrap_seed": bootstrap_config["seed"],
    }


def parse_args():
    """Parse Stage 5 human-evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze human agreement, LLM judges, three-judge aggregations, human reason agreement, and false-negative reason patterns."
    )

    parser.add_argument("--sample", type=Path, default=Path("data/human_annotations/human_annotation_subset_mapping.csv"), help="Frozen human annotation sample.")
    parser.add_argument("--human-label", action="append", default=None, help="Human annotation CSV in NAME=PATH format. Repeat once per annotator.")
    parser.add_argument("--llm-annotations", type=Path, default=Path("data/llm_annotations"), help="Directory containing Stage 3 query_*.json annotation files.")
    parser.add_argument("--aggregations", type=Path, default=Path("results/judge_aggregations.csv"), help="CSV generated by Stage 4 judge aggregation.")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("results/human_evaluation"), help="Directory used to store human-evaluation outputs.")
    parser.add_argument("--human-majority-output", type=Path, default=Path("data/human_annotations/human_majority_labels.csv"),
    help=(
        "Output CSV containing pair-level human labels and "
        "the derived human-majority relevance label."
    ),
)
    parser.add_argument("--benchmark-config", type=Path, default=Path("config/benchmark_config.json"), help="Benchmark configuration JSON.")
    parser.add_argument("--models-config", type=Path, default=Path("config/models.json"), help="Model registry JSON.")

    return parser.parse_args()


def write_evaluation_summary(
    sample_rows,
    human_labels,
    llm_labels,
    aggregation_labels,
    human_majority,
    human_agreement_summary,
    reason_row,
    bootstrap_iterations,
    bootstrap_seed,
    sample_path,
    llm_annotations_path,
    aggregations_path,
    human_label_args,
    human_majority_output,
    output_dir,
):
    """Write and print the Stage 5 evaluation summary."""
    human_positive = sum(value is True for value in human_majority)
    human_negative = sum(value is False for value in human_majority)

    summary = {
        "sample_size": len(sample_rows),
        "human_annotators": list(human_labels),
        "individual_llm_judges": list(llm_labels),
        "three_judge_combinations": list(aggregation_labels),
        "human_majority": {
            "positive": human_positive,
            "negative": human_negative,
            "positive_rate": human_positive / len(sample_rows),
        },
        "human_agreement": human_agreement_summary,
        "reason_agreement": reason_row,
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "resampling_unit": "query_dataset_pair",
            "confidence_interval": "percentile_95",
        },
        "inputs": {
            "sample": str(sample_path),
            "llm_annotations": str(llm_annotations_path),
            "aggregations": str(aggregations_path),
            "human_labels": human_label_args,
        },
        "outputs": {
            "human_majority_labels": str(human_majority_output),
        },
    }

    write_json_atomic(
        summary,
        output_dir / "human_evaluation_summary.json",
    )

    print("Human evaluation completed.")
    print(f" - Human-evaluated pairs: {len(sample_rows)}")
    print(f" - Human-majority positive: {human_positive}")
    print(f" - Human-majority negative: {human_negative}")
    print(f" - Binary Fleiss' kappa: {human_agreement_summary['fleiss_kappa']:.6f}")
    print(f" - Reason Fleiss' kappa: {reason_row['fleiss_kappa_reason']:.6f}")
    print(f" - Outputs: {output_dir}")


def main() -> None:
    """Run Stage 5 human agreement and LLM evaluation analysis."""
    args = parse_args()

    benchmark_config = load_benchmark_config(args.benchmark_config)
    evaluation_config = evaluation_settings_from_config(benchmark_config)

    expected_human_pairs = evaluation_config["sample_size"]
    bootstrap_iterations = evaluation_config["bootstrap_iterations"]
    bootstrap_seed = evaluation_config["bootstrap_seed"]

    registered_models = set(load_model_ids(args.models_config))
    models = evaluation_config["models"]

    unknown_models = [
        model
        for model in models
        if model not in registered_models
    ]

    if unknown_models:
        raise ValueError(
            "Evaluation judge(s) are not present in models configuration: "
            f"{unknown_models}"
        )

    human_label_args = list(args.human_label or DEFAULT_HUMAN_ANNOTATIONS)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = load_sample(args.sample)

    if len(sample_rows) != expected_human_pairs:
        raise ValueError(
            "Unexpected human sample size: "
            f"{len(sample_rows)} != {expected_human_pairs}"
        )

    human_files = parse_human_label_arguments(human_label_args)
    expected_human_annotators = evaluation_config["human_annotators"]

    if len(human_files) != expected_human_annotators:
        raise ValueError(
            "Unexpected number of human annotators: "
            f"{len(human_files)} != {expected_human_annotators}"
        )

    human_labels, human_reasons = load_human_labels(
        human_files,
        sample_rows,
        evaluation_config["relevance_mapping"],
    )

    llm_labels, llm_reasons = load_llm_labels_and_reasons(
        args.llm_annotations,
        sample_rows,
        models,
        evaluation_config["relevance_mapping"],
    )

    aggregation_labels = load_aggregation_labels(
        args.aggregations,
        sample_rows,
    )

    # Reuse the same bootstrap index samples across judges and metrics so all
    # confidence intervals are based on the same paired resamples.
    bootstrap_samples = generate_bootstrap_samples(
        len(sample_rows),
        bootstrap_iterations,
        bootstrap_seed,
    )
    human_majority = build_human_majority(human_labels)

    human_pairwise_rows, human_agreement_summary = analyze_human_agreement(
        human_labels,
        bootstrap_samples,
    )
    individual_vs_humans = analyze_predictions_vs_humans(
        llm_labels,
        human_labels,
        bootstrap_samples,
        name_column="judge",
    )
    individual_vs_majority = analyze_predictions_vs_majority(
        llm_labels,
        human_majority,
        bootstrap_samples,
        name_column="judge",
    )
    combinations_vs_humans = analyze_predictions_vs_humans(
        aggregation_labels,
        human_labels,
        bootstrap_samples,
        name_column="combination",
    )
    combinations_vs_majority = analyze_predictions_vs_majority(
        aggregation_labels,
        human_majority,
        bootstrap_samples,
        name_column="combination",
    )
    reason_row = build_reason_analysis(
        human_reasons,
        human_labels,
        human_majority,
        bootstrap_samples,
        evaluation_config["relevance_mapping"],
    )
    fn_rows = build_false_negative_rows(
        llm_labels,
        llm_reasons,
        human_reasons,
        human_majority,
        individual_vs_majority,
        evaluation_config["reason_labels"],
        evaluation_config["relevance_mapping"],
    )

    write_human_majority_labels(sample_rows, human_labels, human_majority, args.human_majority_output)
    write_csv_atomic(human_pairwise_rows, output_dir / "human_pairwise_agreement.csv")
    write_csv_atomic(individual_vs_humans, output_dir / "individual_judges_vs_humans.csv")
    write_csv_atomic(individual_vs_majority, output_dir / "individual_judges_vs_human_majority.csv")
    write_csv_atomic(combinations_vs_humans, output_dir / "three_judge_combinations_vs_humans.csv")
    write_csv_atomic(combinations_vs_majority, output_dir / "three_judge_combinations_vs_human_majority.csv")
    write_csv_atomic([reason_row], output_dir / "reason_analysis.csv")
    write_csv_atomic(fn_rows, output_dir / "false_negative_reason_analysis.csv")

    write_evaluation_summary(
        sample_rows=sample_rows,
        human_labels=human_labels,
        llm_labels=llm_labels,
        aggregation_labels=aggregation_labels,
        human_majority=human_majority,
        human_agreement_summary=human_agreement_summary,
        reason_row=reason_row,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        sample_path=args.sample,
        llm_annotations_path=args.llm_annotations,
        aggregations_path=args.aggregations,
        human_label_args=human_label_args,
        human_majority_output=args.human_majority_output,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
