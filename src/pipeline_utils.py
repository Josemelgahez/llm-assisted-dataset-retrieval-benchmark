# src/pipeline_utils.py

import csv
import json
import math
import re
import os
from pathlib import Path
from typing import Any

def load_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open(encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")

    return value


def require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object.")
    return value


def require_list(value: Any, path: str, *, non_empty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list.")
    if non_empty and not value:
        raise ValueError(f"{path} must be a non-empty list.")
    return value


def require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value


def require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be Boolean.")
    return value


def require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def require_positive_int(value: Any, path: str) -> int:
    value = require_int(value, path)
    if value <= 0:
        raise ValueError(f"{path} must be a positive integer.")
    return value


def require_non_negative_int(value: Any, path: str) -> int:
    value = require_int(value, path)
    if value < 0:
        raise ValueError(f"{path} must be a non-negative integer.")
    return value


def require_number(value: Any, path: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path} must be a finite number.")
    return value


def require_allowed(value: Any, path: str, allowed: set[str]) -> str:
    value = require_string(value, path).strip()
    if value not in allowed:
        allowed_text = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(f"{path} must be one of: {allowed_text}.")
    return value


def require_no_extra_keys(value: dict[str, Any], path: str, allowed: set[str]) -> None:
    extra_keys = sorted(set(value) - allowed)

    if extra_keys:
        raise ValueError(f"{path} contains unsupported key(s): {extra_keys}")


def require_unique_string_list(value: Any, path: str, *, non_empty: bool = True) -> list[str]:
    values = require_list(value, path, non_empty=non_empty)
    normalized_values = []
    seen_values: set[str] = set()

    for index, item in enumerate(values):
        item = require_string(item, f"{path}[{index}]").strip()

        if item in seen_values:
            raise ValueError(f"Duplicate value {item!r} in {path}.")

        seen_values.add(item)
        normalized_values.append(item)

    return normalized_values


def validate_benchmark_config(config: dict[str, Any]) -> None:
    require_object(config, "benchmark_config")

    benchmark = require_object(config.get("benchmark"), "benchmark")
    require_string(
        benchmark.get("default_relevance_source"),
        "benchmark.default_relevance_source",
    )

    collection = require_object(config.get("collection"), "collection")
    filtering = require_object(
        collection.get("filtering"),
        "collection.filtering",
    )
    require_bool(
        filtering.get("require_title"),
        "collection.filtering.require_title",
    )
    require_bool(
        filtering.get("require_description"),
        "collection.filtering.require_description",
    )
    require_positive_int(
        filtering.get("minimum_data_rows"),
        "collection.filtering.minimum_data_rows",
    )

    resource_selection = require_object(
        collection.get("resource_selection"),
        "collection.resource_selection",
    )
    require_positive_int(
        resource_selection.get("resources_per_dataset"),
        "collection.resource_selection.resources_per_dataset",
    )

    content_sampling = require_object(
        collection.get("content_sampling"),
        "collection.content_sampling",
    )
    require_positive_int(
        content_sampling.get("maximum_rows"),
        "collection.content_sampling.maximum_rows",
    )
    require_int(
        content_sampling.get("random_seed"),
        "collection.content_sampling.random_seed",
    )
    require_allowed(
        content_sampling.get("file_order"),
        "collection.content_sampling.file_order",
        {"filesystem", "sorted"},
    )

    candidate_pool = require_object(config.get("candidate_pool"), "candidate_pool")
    require_positive_int(
        candidate_pool.get("pooling_depth"),
        "candidate_pool.pooling_depth",
    )

    llm_annotation = require_object(
        config.get("llm_annotation"),
        "llm_annotation",
    )
    require_non_negative_int(
        llm_annotation.get("content_character_limit"),
        "llm_annotation.content_character_limit",
    )
    require_positive_int(
        llm_annotation.get("maximum_attempts"),
        "llm_annotation.maximum_attempts",
    )
    few_shots = require_bool(
        llm_annotation.get("few_shots"),
        "llm_annotation.few_shots",
    )

    few_shot_examples = require_non_negative_int(
        llm_annotation.get("few_shot_examples"),
        "llm_annotation.few_shot_examples",
    )

    if few_shots and few_shot_examples == 0:
        raise ValueError(
            "llm_annotation.few_shot_examples must be greater than 0 "
            "when llm_annotation.few_shots is true."
        )
    require_string(
        llm_annotation.get("prompt_blueprint"),
        "llm_annotation.prompt_blueprint",
    )
    require_unique_string_list(llm_annotation.get("models"), "llm_annotation.models")

    input_components = require_unique_string_list(
        llm_annotation.get("input_components"),
        "llm_annotation.input_components",
    )
    if "query" not in input_components:
        raise ValueError(
            "llm_annotation.input_components must contain 'query'."
        )
    
    reason_labels = require_unique_string_list(
        llm_annotation.get("reason_labels"),
        "llm_annotation.reason_labels",
    )

    relevance_mapping = require_object(
        llm_annotation.get("relevance_mapping"),
        "llm_annotation.relevance_mapping",
    )
    reason_label_set = set(reason_labels)
    relevance_mapping_keys = set(relevance_mapping)

    missing_relevance_reasons = sorted(reason_label_set - relevance_mapping_keys)
    if missing_relevance_reasons:
        raise ValueError(
            "llm_annotation.relevance_mapping is missing reason(s): "
            f"{missing_relevance_reasons}"
        )

    extra_relevance_reasons = sorted(relevance_mapping_keys - reason_label_set)
    if extra_relevance_reasons:
        raise ValueError(
            "llm_annotation.relevance_mapping contains unexpected reason(s): "
            f"{extra_relevance_reasons}"
        )

    for reason in reason_labels:
        require_bool(
            relevance_mapping.get(reason),
            f"llm_annotation.relevance_mapping.{reason}",
        )

    positive_evidence_fields = require_unique_string_list(
        llm_annotation.get("positive_evidence_fields"),
        "llm_annotation.positive_evidence_fields",
    )
    input_component_set = set(input_components)

    for field in positive_evidence_fields:
        if field == "query":
            raise ValueError(
                "llm_annotation.positive_evidence_fields cannot contain 'query'."
            )

        if field not in input_component_set:
            raise ValueError(
                "llm_annotation.positive_evidence_fields contains field not "
                f"declared in llm_annotation.input_components: {field!r}"
            )

    negative_evidence_normalization = require_object(
        llm_annotation.get("negative_evidence_normalization"),
        "llm_annotation.negative_evidence_normalization",
    )
    require_no_extra_keys(
        negative_evidence_normalization,
        "llm_annotation.negative_evidence_normalization",
        {"supporting_field", "quote"},
    )

    if set(negative_evidence_normalization) != {"supporting_field", "quote"}:
        missing_keys = sorted(
            {"supporting_field", "quote"} - set(negative_evidence_normalization)
        )
        raise ValueError(
            "llm_annotation.negative_evidence_normalization is missing key(s): "
            f"{missing_keys}"
        )

    for key in ("supporting_field", "quote"):
        if negative_evidence_normalization[key] is not None:
            raise ValueError(
                f"llm_annotation.negative_evidence_normalization.{key} must be null."
            )

    invalid_output_policy = require_string(
        llm_annotation.get("invalid_output_policy"),
        "llm_annotation.invalid_output_policy",
    ).strip()

    if invalid_output_policy != "missing":
        raise ValueError("llm_annotation.invalid_output_policy must be 'missing'.")

    human_audit = require_object(
        config.get("human_audit"),
        "human_audit",
    )

    require_positive_int(
        human_audit.get("sample_size"),
        "human_audit.sample_size",
    )
    require_int(
        human_audit.get("sample_seed"),
        "human_audit.sample_seed",
    )
    require_positive_int(
        human_audit.get("annotators"),
        "human_audit.annotators",
    )
    require_allowed(
        human_audit.get("sampling"),
        "human_audit.sampling",
        {"simple_random_without_replacement"},
    )

    human_majority = require_object(
        human_audit.get("human_majority"),
        "human_audit.human_majority",
    )
    require_allowed(
        human_majority.get("binary_rule"),
        "human_audit.human_majority.binary_rule",
        {"2_out_of_3"},
    )

    judge_aggregation = require_object(
        config.get("judge_aggregation"),
        "judge_aggregation",
    )

    combination_size = require_positive_int(
        judge_aggregation.get("judge_combination_size"),
        "judge_aggregation.judge_combination_size",
    )

    if combination_size != 3:
        raise ValueError(
            "judge_aggregation.judge_combination_size must be 3."
        )

    require_positive_int(
        judge_aggregation.get("number_of_combinations"),
        "judge_aggregation.number_of_combinations",
    )

    require_allowed(
        judge_aggregation.get("aggregation_rule"),
        "judge_aggregation.aggregation_rule",
        {"2_out_of_3_majority_vote"},
    )

    require_all_three = require_bool(
        judge_aggregation.get("require_all_three_valid_judgments"),
        "judge_aggregation.require_all_three_valid_judgments",
    )

    if not require_all_three:
        raise ValueError(
            "judge_aggregation.require_all_three_valid_judgments "
            "must be true."
        )

    require_allowed(
        judge_aggregation.get("missing_judgment_policy"),
        "judge_aggregation.missing_judgment_policy",
        {"missing"},
    )

    evaluation = require_object(
        config.get("evaluation"),
        "evaluation",
    )

    bootstrap = require_object(
        evaluation.get("bootstrap"),
        "evaluation.bootstrap",
    )

    confidence_level = require_number(
        bootstrap.get("confidence_level"),
        "evaluation.bootstrap.confidence_level",
    )

    if not math.isclose(
        float(confidence_level),
        0.95,
    ):
        raise ValueError(
            "evaluation.bootstrap.confidence_level must be 0.95."
        )

    require_allowed(
        bootstrap.get("method"),
        "evaluation.bootstrap.method",
        {"percentile"},
    )

    require_positive_int(
        bootstrap.get("iterations"),
        "evaluation.bootstrap.iterations",
    )
    require_int(
        bootstrap.get("seed"),
        "evaluation.bootstrap.seed",
    )

    require_allowed(
        bootstrap.get("resampling_unit"),
        "evaluation.bootstrap.resampling_unit",
        {"query_dataset_pair"},
    )

    paired = require_bool(
        bootstrap.get("paired"),
        "evaluation.bootstrap.paired",
    )

    if not paired:
        raise ValueError(
            "evaluation.bootstrap.paired must be true."
        )


def platform_from_model_entry(model: dict[str, Any]) -> str:
    platform_text = str(model.get("platform", "")).lower()
    execution_text = str(model.get("execution", "")).lower()

    if "ollama" in platform_text or execution_text == "local":
        return "ollama"
    if "openai" in platform_text or "openai" in execution_text:
        return "openai"
    if (
        "vertex" in platform_text
        or "gemini" in platform_text
        or "google" in platform_text
    ):
        return "vertex"

    raise ValueError(
        "Could not derive platform/backend from model configuration: "
        f"{model}"
    )


def validate_models_config(config: dict[str, Any]) -> None:
    require_object(config, "models_config")

    models = require_list(config.get("models"), "models", non_empty=True)
    seen_ids: set[str] = set()

    for index, model in enumerate(models):
        path = f"models[{index}]"
        model = require_object(model, path)

        model_id = require_string(model.get("id"), f"{path}.id").strip()
        if model_id in seen_ids:
            raise ValueError(f"Duplicate model id {model_id!r} in models.")
        seen_ids.add(model_id)

        require_string(model.get("platform"), f"{path}.platform")
        require_string(model.get("execution"), f"{path}.execution")

        generation = require_object(model.get("generation"), f"{path}.generation")
        backend = platform_from_model_entry(model)

        if backend == "ollama":
            require_no_extra_keys(
                generation,
                f"{path}.generation",
                {
                    "num_ctx",
                    "temperature",
                    "top_p",
                    "repeat_penalty",
                    "max_output_tokens",
                    "structured_output",
                },
            )
            require_positive_int(generation.get("num_ctx"), f"{path}.generation.num_ctx")
            require_number(generation.get("temperature"), f"{path}.generation.temperature")
            require_number(generation.get("top_p"), f"{path}.generation.top_p")
            require_number(
                generation.get("repeat_penalty"),
                f"{path}.generation.repeat_penalty",
            )
            require_positive_int(
                generation.get("max_output_tokens"),
                f"{path}.generation.max_output_tokens",
            )
            require_bool(
                generation.get("structured_output"),
                f"{path}.generation.structured_output",
            )

        elif backend == "vertex":
            require_no_extra_keys(
                generation,
                f"{path}.generation",
                {
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "response_mime_type",
                },
            )
            require_number(generation.get("temperature"), f"{path}.generation.temperature")
            require_number(generation.get("top_p"), f"{path}.generation.top_p")
            require_positive_int(
                generation.get("max_output_tokens"),
                f"{path}.generation.max_output_tokens",
            )
            require_string(
                generation.get("response_mime_type"),
                f"{path}.generation.response_mime_type",
            )

        elif backend == "openai":
            if generation:
                raise ValueError(
                    f"{path}.generation must be empty for OpenAI models."
                )


QUERY_FILENAME_RE = re.compile(r"^query_(\d+)\.json$")


def extract_query_number(filename):
    """Extract the numeric query ID from a query_<id>.json filename."""
    match = QUERY_FILENAME_RE.fullmatch(filename)

    if not match:
        raise ValueError(f"Invalid query filename: {filename!r}")

    query_id = int(match.group(1))

    if query_id <= 0:
        raise ValueError(
            f"Query filename ID must be greater than 0: {filename!r}"
        )

    return query_id


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config = load_json_object(path)
    validate_benchmark_config(config)
    return config


def load_models_config(path: str | Path) -> dict[str, Any]:
    config = load_json_object(path)
    validate_models_config(config)
    return config


def load_model_ids(path: str | Path) -> list[str]:
    """Load validated model IDs from the model registry."""
    config = load_models_config(path)
    return [model["id"].strip() for model in config["models"]]


def parse_query_id(value: Any, context: str) -> int:
    """Parse a positive query ID from an integer or numeric string."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid query ID in {context}: {value!r}")

    if isinstance(value, int):
        query_id = value
    elif isinstance(value, str):
        text = value.strip()

        if not text or not text.isdigit():
            raise ValueError(f"Invalid query ID in {context}: {value!r}")

        query_id = int(text)
    else:
        raise ValueError(f"Invalid query ID in {context}: {value!r}")

    if query_id <= 0:
        raise ValueError(f"Invalid query ID in {context}: {value!r}")

    return query_id


def parse_dataset_id(value: Any, context: str) -> str:
    """Parse and normalize a dataset ID."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid dataset ID in {context}: {value!r}")

    return value.strip()


def make_pair_key(query_id: int, dataset_id: str) -> str:
    """Build a query-dataset pair identifier."""
    return f"{query_id}||{dataset_id}"


def parse_pair_key(value: Any, context: str) -> str:
    """Parse and normalize a query-dataset pair identifier."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid pair_key in {context}: {value!r}")

    parts = value.strip().split("||")

    if len(parts) != 2:
        raise ValueError(f"Invalid pair_key in {context}: {value!r}")

    query_text, dataset_text = (part.strip() for part in parts)
    query_id = parse_query_id(query_text, context)
    dataset_id = parse_dataset_id(dataset_text, context)

    return make_pair_key(query_id, dataset_id)


def annotation_reason_and_relevance(
    annotation: Any,
    relevance_mapping: dict[str, bool],
) -> tuple[str | None, bool | None]:
    if not isinstance(annotation, dict) or "error" in annotation:
        return None, None

    reason = annotation.get("reason")

    if not isinstance(reason, str):
        return None, None

    reason = reason.strip()

    if reason not in relevance_mapping:
        return None, None

    derived = relevance_mapping[reason]

    if "relevance" in annotation:
        stored = annotation.get("relevance")

        if not isinstance(stored, bool):
            raise ValueError(
                f"Stored LLM relevance must be Boolean: {stored!r}"
            )

        if stored != derived:
            raise ValueError(
                "Stored LLM relevance is inconsistent with its reason: "
                f"reason={reason!r}, stored={stored!r}, derived={derived!r}"
            )

    return reason, derived


def list_query_json_files(input_dir: str | Path) -> list[Path]:
    """List query_<id>.json files in numeric query-ID order."""
    input_dir = Path(input_dir)
    files_by_query_id: dict[int, Path] = {}
    malformed_files = []

    for path in input_dir.iterdir():
        if not path.is_file():
            continue

        if path.name.startswith("query_") and path.suffix == ".json":
            try:
                query_id = extract_query_number(path.name)
            except ValueError:
                malformed_files.append(path.name)
                continue

            if query_id in files_by_query_id:
                raise ValueError(
                    "Duplicate numeric query ID in query filenames: "
                    f"{files_by_query_id[query_id].name!r} and {path.name!r}"
                )

            files_by_query_id[query_id] = path

    if malformed_files:
        raise ValueError(
            f"Malformed query filename(s): {sorted(malformed_files)}"
        )

    if not files_by_query_id:
        raise FileNotFoundError(
            f"No valid query_<positive integer>.json files found in {input_dir}"
        )

    return [
        files_by_query_id[query_id]
        for query_id in sorted(files_by_query_id)
    ]


def write_csv_atomic(
    rows: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Write rows atomically to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        if not rows and fieldnames is None:
            temporary_path.write_text("", encoding="utf-8")
        else:
            if fieldnames is None:
                fieldnames = list(rows[0].keys())

            with temporary_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        temporary_path.replace(path)

    except Exception:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise


def parse_row_pair_key(
    row: dict[str, Any],
    context: str,
) -> tuple[str, int, str]:
    """Parse and validate the query-dataset identifiers in a row."""
    if "id_query" not in row:
        raise ValueError(f"{context}: missing id_query.")
    if "id_dataset" not in row:
        raise ValueError(f"{context}: missing id_dataset.")

    query_id = parse_query_id(row.get("id_query"), context)
    dataset_id = parse_dataset_id(row.get("id_dataset"), context)
    pair_key = make_pair_key(query_id, dataset_id)

    if "pair_key" in row and str(row.get("pair_key", "")).strip():
        stored_pair_key = parse_pair_key(
            row.get("pair_key"),
            context,
        )

        if stored_pair_key != pair_key:
            raise ValueError(
                f"{context}: inconsistent pair_key "
                f"{stored_pair_key!r} != {pair_key!r}"
            )

    return pair_key, query_id, dataset_id


def write_json_atomic(
    payload: Any,
    path: str | Path,
) -> None:
    """Write UTF-8 JSON atomically with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        temporary_path.replace(path)

    except Exception:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise
