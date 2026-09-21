#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from pipeline_utils import load_benchmark_config

EXCLUSION_FILENAMES = {
    "duplicate_metadata": "duplicate_metadata.json",
    "empty_metadata": "empty_metadata.json",
    "invalid_resources": "invalid_resources.json",
    "duplicate_resources": "duplicate_resources.json",
}

csv.field_size_limit(sys.maxsize)


def collection_settings_from_config(config):
    """Read Stage 1 settings from the collection section of the config."""
    collection = config["collection"]
    content_sampling = collection["content_sampling"]
    filtering = collection["filtering"]
    resource_selection = collection["resource_selection"]

    return {
        "sample_rows": content_sampling["maximum_rows"],
        "random_seed": content_sampling["random_seed"],
        "minimum_data_rows": filtering["minimum_data_rows"],
        "file_order": content_sampling["file_order"],
        "require_title": filtering["require_title"],
        "require_description": filtering["require_description"],
        "resources_per_dataset": resource_selection["resources_per_dataset"],
    }


def list_metadata_files(collection_dir, file_order):
    """Return metadata filenames using the requested traversal order."""
    filenames = [name for name in os.listdir(collection_dir) if name.lower().startswith("meta_")]

    # Sorting provides portable deterministic traversal. Filesystem order is
    # retained as an explicit compatibility option for the released snapshot.
    if file_order == "sorted":
        filenames.sort()

    return filenames


def iter_resources(metadata):
    """
    Iterate over resources while preserving repository-provided order.

    Yields tuples of (resource_key, resource, container_type), where
    container_type is either "dict" or "list".
    """
    resources = metadata.get("resources", [])

    if isinstance(resources, dict):
        for key, resource in resources.items():
            if isinstance(resource, dict):
                yield key, resource, "dict"
        return

    if isinstance(resources, list):
        for index, resource in enumerate(resources):
            if isinstance(resource, dict):
                yield index, resource, "list"


def preferred_multilingual_text(value, preferred_languages=("es", "en")):
    """
    Return the preferred non-empty text from a multilingual field.

    Priority:
      1. Spanish
      2. English
      3. First non-empty value
    """

    def first_nonempty_from_list(values):
        """Return the first non-empty textual value from a multilingual list."""
        for item in values:
            if isinstance(item, str) and item.strip():
                return item.strip()

            if isinstance(item, dict) and item.get("value"):
                text = str(item.get("value", "")).strip()
                if text:
                    return text

        return ""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        # Support both {"language": ..., "value": ...} entries and mappings
        # such as {"es": ..., "en": ...} produced by different metadata sources.
        if "language" in value and "value" in value:
            text = str(value.get("value", "")).strip()
            if text:
                return text

        for language in preferred_languages:
            if language not in value or not value[language]:
                continue

            candidate = value[language]

            if isinstance(candidate, list):
                text = first_nonempty_from_list(candidate)
                if text:
                    return text

            elif isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        # If neither preferred language is available, retain the first usable
        # textual value instead of discarding otherwise valid metadata.
        for candidate in value.values():
            if isinstance(candidate, list):
                text = first_nonempty_from_list(candidate)
                if text:
                    return text

            elif isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return ""

    if isinstance(value, list):
        for language in preferred_languages:
            for item in value:
                if (
                    isinstance(item, dict)
                    and item.get("language") == language
                    and item.get("value")
                ):
                    text = str(item.get("value", "")).strip()
                    if text:
                        return text

        return first_nonempty_from_list(value)

    return ""


def raw_file_md5(path, chunk_size=1024 * 1024):
    """Compute the raw-byte MD5 used only for metadata-level deduplication."""
    if not path.exists():
        return None

    try:
        digest = hashlib.md5()

        with path.open("rb") as file:
            while True:
                block = file.read(chunk_size)
                if not block:
                    break
                digest.update(block)

        return digest.hexdigest()

    except Exception as exc:
        print(f"[WARN] Could not hash {path}: {exc}")
        return None


def resource_filename(resource):
    """Return the physical resource filename for a resource entry."""
    filename = resource.get("fileName")
    if filename:
        return str(filename)

    path_value = resource.get("path")
    if path_value:
        return Path(str(path_value)).name

    return ""


def metadata_functional_key(metadata, collection_dir):
    """
    Build the metadata deduplication key.

    The key contains:
      - preferred title
      - preferred description
      - serialized temporal coverage
      - normalized geographic coverage
      - sorted raw MD5 hashes of downloaded resources

    This preserves the deduplication semantics used by the benchmark.
    """
    # Normalize the metadata fields that define semantic equivalence.
    title = preferred_multilingual_text(metadata.get("title", []))
    description = preferred_multilingual_text(metadata.get("description", []))
    temporal = json.dumps(
        metadata.get("temporal", {}),
        sort_keys=True,
    )
    geo = str(metadata.get("geo", "")).strip().lower()

    # Include raw resource hashes so records with different downloaded data
    # are not collapsed solely because their metadata is identical.
    resource_hashes = []

    for _, resource, _ in iter_resources(metadata):
        # `path` marks the resource as downloaded, while `fileName` is the
        # physical filename used inside the local crawler snapshot directory.
        if not resource.get("path"):
            continue

        filename = resource.get("fileName")
        if not filename:
            continue

        hash_value = raw_file_md5(collection_dir / str(filename))

        if hash_value is not None:
            resource_hashes.append(hash_value)

    return json.dumps(
        {
            "title": title,
            "description": description,
            "temporal": temporal,
            "geo": geo,
            "hashes_csv": sorted(resource_hashes),
        },
        sort_keys=True,
    )


def detect_duplicate_metadata(collection_dir, metadata_files):
    """
    Detect duplicate metadata records using the functional key.

    The first record encountered for each key is retained. Therefore, when
    file_order="filesystem", representative selection depends on filesystem
    enumeration order.
    """
    # Map each functional key to the first metadata record encountered.
    seen_keys = {}
    duplicates = []
    groups = {}

    # Representative selection is intentionally first-seen: changing traversal
    # order can therefore change which duplicate metadata record is retained.
    for filename in tqdm(
        metadata_files,
        desc="Detecting duplicate metadata",
    ):
        path = collection_dir / filename

        try:
            with path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

            key = metadata_functional_key(
                metadata,
                collection_dir,
            )

        except Exception as exc:
            print(f"[WARN] Could not build deduplication key for {filename}: {exc}")
            continue

        if key in seen_keys:
            duplicates.append(filename)

            groups.setdefault(
                key,
                [seen_keys[key]],
            ).append(filename)

        else:
            seen_keys[key] = filename

    duplicate_groups = [members for members in groups.values() if len(members) > 1]

    return duplicates, duplicate_groups


def has_insufficient_nonempty_lines(path, encoding, minimum_data_rows):
    """
    Return True when a resource is missing, empty, unreadable, or contains
    fewer than the required header plus data rows.

    A valid tabular resource must contain one header line plus the configured
    minimum number of data rows.
    """
    required_nonempty_lines = minimum_data_rows + 1

    try:
        if not path.exists() or path.stat().st_size == 0:
            return True

        nonempty_lines = 0

        with path.open("r", encoding=encoding, errors="ignore") as file:
            for line in file:
                if line.strip():
                    nonempty_lines += 1

                    if nonempty_lines >= required_nonempty_lines:
                        return False

        return True

    except Exception as exc:
        print(f"[WARN] Could not validate resource {path}: {exc}")
        return True


def invalid_resource_identifier(resource_key, resource, container_type):
    """
    Return the identifier stored in invalid_resources.json.

    For dictionary-based resource containers, the manifest uses the resource
    key. For list-based containers, it uses fileName/path.
    """
    if container_type == "dict":
        return str(resource_key)

    return resource_filename(resource) or str(resource.get("path") or "")


def detect_invalid_resources(
    collection_dir,
    metadata_files,
    minimum_data_rows,
):
    """
    Detect invalid resources and metadata records without any valid resource.

    A resource is invalid when:
      - it has no downloaded path;
      - its physical file has fewer than the required non-empty lines; or
      - the metadata has no schema for the resource.
    """
    empty_metadata = []
    invalid_resources = []

    counters = {
        "metadata_processed": 0,
        "resources_processed": 0,
        "valid_resources": 0,
        "invalid_resources": 0,
    }

    for filename in tqdm(
        metadata_files,
        desc="Validating resources",
    ):
        counters["metadata_processed"] += 1

        path = collection_dir / filename

        try:
            with path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

        except Exception as exc:
            print(f"[WARN] Could not read metadata {filename}: {exc}")
            continue

        valid_count = 0
        total_count = 0

        # A metadata record is retained only if at least one resource
        # passes the file and schema checks.
        for resource_key, resource, container_type in iter_resources(metadata):
            total_count += 1
            counters["resources_processed"] += 1

            invalid_identifier = invalid_resource_identifier(
                resource_key,
                resource,
                container_type,
            )

            if not resource.get("path"):
                invalid_resources.append(invalid_identifier)
                counters["invalid_resources"] += 1
                continue

            filename_on_disk = resource_filename(resource)

            if not filename_on_disk:
                invalid_resources.append(invalid_identifier)
                counters["invalid_resources"] += 1
                continue

            resource_path = collection_dir / filename_on_disk

            invalid_file = has_insufficient_nonempty_lines(
                resource_path,
                resource.get("encoding"),
                minimum_data_rows,
            )

            # Schema availability is required because downstream representation
            # construction relies on column information from the metadata.
            missing_schema = not resource.get("schema")

            if invalid_file or missing_schema:
                invalid_resources.append(invalid_identifier)
                counters["invalid_resources"] += 1
                continue

            valid_count += 1
            counters["valid_resources"] += 1

        if total_count == 0 or valid_count == 0:
            empty_metadata.append(filename)

    return (
        empty_metadata,
        invalid_resources,
        counters,
    )


def canonical_cell(value):
    """Canonicalize one table cell for resource deduplication."""
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    return " ".join(text.split())


def canonical_csv_md5(
    path,
    encoding,
    delimiter,
    include_header=True,
):
    """
    Compute the normalized table-content hash used for resource deduplication.

    Cells are normalized before hashing so superficial whitespace and line-ending
    differences do not prevent equivalent tabular resources from matching.
    """
    try:
        with path.open(
            "r",
            encoding=encoding,
            errors="ignore",
            newline="",
        ) as file:
            reader = csv.reader(file, delimiter=delimiter)

            digest = hashlib.md5()
            # Explicit separators prevent different row/column layouts from
            # producing the same byte sequence after cell normalization.
            column_separator = b"\x1f"
            row_separator = b"\x1e"

            for row_index, row in enumerate(reader):
                if row_index == 0 and not include_header:
                    continue

                for cell in row:
                    digest.update(canonical_cell(cell).encode("utf-8"))
                    digest.update(column_separator)

                digest.update(row_separator)

            return digest.hexdigest()

    except Exception as exc:
        print(f"[WARN] Could not compute canonical hash for {path}: {exc}")
        return None


def preferred_resource(resource_a, resource_b):
    """
    Select the preferred resource among equivalent tabular resources.

    Priority:
      1. encoding containing UTF-8
      2. semicolon delimiter
      3. lexicographically smaller path
    """
    a_utf8 = "UTF-8" in (resource_a.get("encoding") or "").upper()
    b_utf8 = "UTF-8" in (resource_b.get("encoding") or "").upper()

    if a_utf8 != b_utf8:
        if a_utf8:
            return resource_a
        return resource_b

    a_semicolon = (resource_a.get("delimiter") or "").strip() == ";"
    b_semicolon = (resource_b.get("delimiter") or "").strip() == ";"

    if a_semicolon != b_semicolon:
        if a_semicolon:
            return resource_a
        return resource_b

    if str(resource_a.get("path", "")) <= str(resource_b.get("path", "")):
        return resource_a

    return resource_b


def detect_duplicate_resources(
    collection_dir,
    metadata_files,
    duplicate_metadata,
    empty_metadata,
    invalid_resources,
):
    """
    Detect equivalent tabular resources within each retained metadata record.
    """
    duplicate_metadata = set(duplicate_metadata)
    empty_metadata = set(empty_metadata)
    invalid_resources = set(invalid_resources)

    duplicate_resources = []

    # Resource duplicates are resolved only within the same metadata record.
    # Equal tables belonging to different datasets are not collapsed.
    for metadata_filename in tqdm(
        metadata_files,
        desc="Detecting duplicate resources",
    ):
        if metadata_filename in duplicate_metadata or metadata_filename in empty_metadata:
            continue

        metadata_path = collection_dir / metadata_filename

        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        except Exception as exc:
            print(f"[WARN] Could not read metadata {metadata_filename}: {exc}")
            continue

        valid_resources = []
        resource_hashes = []

        for resource_key, resource, _ in iter_resources(metadata):
            if not resource.get("path"):
                continue

            filename = resource.get("fileName")

            # Invalid resources may be identified by resource key or filename,
            # depending on the resource container representation.
            if filename in invalid_resources or str(resource_key) in invalid_resources:
                continue

            if not filename:
                continue

            hash_value = canonical_csv_md5(
                collection_dir / str(filename),
                resource.get("encoding"),
                resource.get("delimiter"),
            )

            if hash_value is None:
                continue

            valid_resources.append(resource)
            resource_hashes.append(hash_value)

        # Keep one preferred resource for each canonical table-content hash.
        retained_by_hash = {}

        for index, hash_value in enumerate(resource_hashes):
            if hash_value not in retained_by_hash:
                retained_by_hash[hash_value] = index
                continue

            retained_index = retained_by_hash[hash_value]
            retained_resource = valid_resources[retained_index]
            candidate_resource = valid_resources[index]

            winner = preferred_resource(retained_resource, candidate_resource)

            if winner is candidate_resource:
                duplicate_resources.append(retained_resource.get("path"))

                retained_by_hash[hash_value] = index

            else:
                duplicate_resources.append(candidate_resource.get("path"))

    return [value for value in duplicate_resources if value]


def tabular_extension(resource):
    """Return csv/tsv for supported tabular resources, otherwise None."""
    name = str(resource.get("fileName") or resource.get("path") or "").lower()

    if name.endswith(".csv"):
        return "csv"

    if name.endswith(".tsv"):
        return "tsv"

    return None


def resource_delimiter(resource):
    """Resolve delimiter using the reducer fallback rules."""
    delimiter = str(resource.get("delimiter") or "").strip()

    if delimiter:
        return delimiter

    if tabular_extension(resource) == "tsv":
        return "\t"

    return ";"


def lines_without_trailing_padding(file_obj):
    """Strip trailing spaces, tabs, and line endings before CSV parsing."""
    for line in file_obj:
        yield line.rstrip(" \t\r\n")


def reduce_tabular_resource(
    input_path,
    output_path,
    encoding,
    delimiter,
    sample_rows,
    rng,
):
    """
    Keep the header and at most sample_rows data rows using reservoir sampling.

    The sampled rows are sorted back into their source relative order.
    """
    with input_path.open(
        "r",
        encoding=encoding,
        errors="ignore",
        newline="",
    ) as input_file:
        reader = csv.reader(
            lines_without_trailing_padding(input_file),
            delimiter=delimiter,
        )

        header = next(reader)

        # Reservoir sampling selects at most sample_rows without loading the
        # complete resource into memory.
        reservoir = []
        seen_rows = 0

        for row in reader:
            seen_rows += 1

            if len(reservoir) < sample_rows:
                reservoir.append((seen_rows, row))
            else:
                replacement_index = rng.randint(1, seen_rows)

                if replacement_index <= sample_rows:
                    reservoir[replacement_index - 1] = (seen_rows, row)
    # Restore sampled rows to their original relative order.
    reservoir.sort(key=lambda item: item[0])

    sampled_rows = [row for _, row in reservoir]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file,
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            quoting=csv.QUOTE_ALL,
        )

        writer.writerow(header)
        writer.writerows(sampled_rows)

    return {
        "source_data_rows": seen_rows,
        "sampled_data_rows": len(sampled_rows),
    }


def rebuild_resources_container(original_resources, retained_items):
    """Preserve the source list/dict representation of resources."""
    if isinstance(original_resources, dict):
        return dict(retained_items)

    return [resource for _, resource in retained_items]


def prepare_reduced_collection(
    collection_dir,
    output_dir,
    metadata_files,
    duplicate_metadata,
    empty_metadata,
    invalid_resources,
    duplicate_resources,
    sample_rows,
    random_seed,
):
    """
    Create the reduced collection used by the downstream indexing pipeline.
    """
    # Convert exclusion lists to sets because membership is checked repeatedly
    # while traversing the collection.
    duplicate_metadata = set(duplicate_metadata)
    empty_metadata = set(empty_metadata)
    invalid_resources = set(invalid_resources)
    duplicate_resources = set(duplicate_resources)
    
    # A single seeded RNG is shared across resources. Consequently, sampling is
    # deterministic for a fixed traversal order, seed, and collection snapshot.
    rng = random.Random(random_seed)

    report = {
        "metadata_written": 0,
        "resources_written": 0,
        "resources_excluded": 0,
        "resource_rows": {},
    }
    # Metadata records rejected during the previous checks are not processed
    # when constructing the reduced collection.
    retained_metadata_files = [
        filename
        for filename in metadata_files
        if (filename not in duplicate_metadata and filename not in empty_metadata)
    ]

    for metadata_filename in tqdm(retained_metadata_files, desc="Reducing collection"):
        metadata_path = collection_dir / metadata_filename

        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        except Exception as exc:
            print(f"[WARN] Could not read metadata {metadata_filename}: {exc}")
            continue

        original_resources = metadata.get("resources", [])

        retained_items = []
        # Keep only valid tabular resources and generate their sampled version.
        for resource_key, resource, _ in iter_resources(metadata):
            resource_path_value = resource.get("path")

            if not resource_path_value:
                report["resources_excluded"] += 1
                continue

            if not tabular_extension(resource):
                report["resources_excluded"] += 1
                continue

            filename = resource_filename(resource)

            if not filename:
                report["resources_excluded"] += 1
                continue

            if (
                filename in invalid_resources
                or str(resource_path_value) in invalid_resources
                or str(resource_key) in invalid_resources
                or filename in duplicate_resources
                or str(resource_path_value) in duplicate_resources
            ):
                report["resources_excluded"] += 1
                continue

            input_resource_path = collection_dir / filename
            output_resource_path = output_dir / filename

            encoding = resource.get(
                "encoding",
                "utf-8",
            )
            delimiter = resource_delimiter(resource)

            try:
                row_stats = reduce_tabular_resource(
                    input_path=input_resource_path,
                    output_path=output_resource_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    sample_rows=sample_rows,
                    rng=rng,
                )

            except Exception as exc:
                print(f"[WARN] Could not reduce resource {filename}: {exc}")
                report["resources_excluded"] += 1
                continue

            # The reduced resource is physically rewritten as UTF-8 using the resolved
            # delimiter, so its metadata must describe the normalized file rather than
            # the original crawler representation.
            normalized_resource = dict(resource)
            normalized_resource["encoding"] = "utf-8"
            normalized_resource["delimiter"] = delimiter

            retained_items.append(
                (resource_key, normalized_resource)
            )

            report["resources_written"] += 1
            report["resource_rows"][filename] = row_stats

        if not retained_items:
            continue
        # Preserve whether the source metadata represented resources as a
        # dictionary or as a list.
        metadata["resources"] = rebuild_resources_container(original_resources, retained_items)

        output_metadata_path = output_dir / metadata_filename

        with output_metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        report["metadata_written"] += 1

    return report


def normalized_component_checks(
    output_dir,
    file_order,
):
    """
    Validate the textual components expected by later stages without changing
    the prepared collection.

    This is intentionally diagnostic only: adding a new filtering rule here
    would alter the released benchmark.
    """
    metadata_files = list_metadata_files(
        output_dir,
        file_order,
    )

    checks = {
        "metadata_records": len(metadata_files),
        "missing_title": [],
        "missing_description": [],
        "metadata_without_resources": [],
        "resource_count_mismatches": [],
        "resources_without_schema": [],
    }

    for filename in metadata_files:
        path = output_dir / filename

        try:
            with path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)

        except Exception:
            continue

        if not preferred_multilingual_text(metadata.get("title")):
            checks["missing_title"].append(filename)

        if not preferred_multilingual_text(metadata.get("description")):
            checks["missing_description"].append(filename)

        resources = list(iter_resources(metadata))
        resource_count = len(resources)

        if not resources:
            checks["metadata_without_resources"].append(filename)

        checks.setdefault("resource_counts", {})[filename] = resource_count

        for _, resource, _ in resources:
            if not resource.get("schema"):
                checks["resources_without_schema"].append(
                    {
                        "metadata": filename,
                        "resource": resource_filename(resource),
                    }
                )

    return checks


def validate_prepared_collection_contract(
    component_checks,
    require_title,
    require_description,
    resources_per_dataset,
):
    """Fail if the prepared collection violates declared Stage 1 settings."""
    if require_title and component_checks["missing_title"]:
        raise ValueError(
            "Prepared collection contains metadata without title despite "
            "collection.filtering.require_title=true. Examples: "
            f"{component_checks['missing_title'][:10]}"
        )

    if require_description and component_checks["missing_description"]:
        raise ValueError(
            "Prepared collection contains metadata without description despite "
            "collection.filtering.require_description=true. Examples: "
            f"{component_checks['missing_description'][:10]}"
        )

    if resources_per_dataset is not None:
        mismatches = [
            {
                "metadata": filename,
                "resources": count,
            }
            for filename, count in component_checks["resource_counts"].items()
            if count != resources_per_dataset
        ]

        component_checks["resource_count_mismatches"] = mismatches

        if mismatches:
            raise ValueError(
                "Prepared collection violates "
                "collection.resource_selection.resources_per_dataset="
                f"{resources_per_dataset}. Examples: {mismatches[:10]}"
            )


def write_json(path, value):
    """Write UTF-8 JSON with stable indentation."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )


def ensure_empty_output_directory(
    output_dir,
    overwrite,
):
    """Create a clean output directory without silently mixing old outputs."""
    if output_dir.exists():
        has_content = any(output_dir.iterdir())

        if has_content:
            if not overwrite:
                raise FileExistsError(
                    f"Output directory is not empty: {output_dir}. "
                    "Use --overwrite to replace it."
                )

            shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def parse_args():
    """Parse Stage 1 collection-preparation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a dataset collection for benchmark construction by filtering invalid or duplicate resources and creating the reduced tabular representation."
    )

    parser.add_argument("--input", "-i", required=True, help="Raw crawler snapshot directory containing meta_*.json and resources.")
    parser.add_argument("--output", "-o", required=True, help="Output directory for the prepared reduced collection.")
    parser.add_argument("--work-dir", default="collection_preparation", help="Directory used for exclusion manifests and preparation reports.")
    parser.add_argument("--benchmark-config", default="config/benchmark_config.json", help="Benchmark configuration JSON containing the Stage 1 settings.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")

    return parser.parse_args()


def print_preparation_summary(
    metadata_files,
    manifests,
    preparation,
    component_checks,
    output_dir,
    report_path,
):
    """Print a summary of the prepared collection and recorded exclusions."""
    print("\nCollection preparation completed.")
    print(f" - Input metadata records: {len(metadata_files)}")
    print(f" - Duplicate metadata excluded: {len(manifests['duplicate_metadata'])}")
    print(f" - Metadata without valid resources excluded: {len(manifests['empty_metadata'])}")
    print(f" - Invalid resources recorded: {len(manifests['invalid_resources'])}")
    print(f" - Duplicate resources recorded: {len(manifests['duplicate_resources'])}")
    print(f" - Prepared metadata records: {preparation['metadata_written']}")
    print(f" - Prepared tabular resources: {preparation['resources_written']}")
    print(f" - Missing title after preparation: {len(component_checks['missing_title'])}")
    print(f" - Missing description after preparation: {len(component_checks['missing_description'])}")
    print(
        " - Resource-count mismatches after preparation: "
        f"{len(component_checks['resource_count_mismatches'])}"
    )
    print(f" - Prepared collection: {output_dir}")
    print(f" - Preparation report: {report_path}")


def main():
    """Run Stage 1 collection filtering, reduction, and reporting."""
    args = parse_args()

    benchmark_config = load_benchmark_config(args.benchmark_config)
    collection_settings = collection_settings_from_config(benchmark_config)

    sample_rows = collection_settings["sample_rows"]
    random_seed = collection_settings["random_seed"]
    minimum_data_rows = collection_settings["minimum_data_rows"]
    file_order = collection_settings["file_order"]
    require_title = collection_settings["require_title"]
    require_description = collection_settings["require_description"]
    resources_per_dataset = collection_settings["resources_per_dataset"]

    collection_dir = Path(args.input).expanduser().resolve()

    output_dir = Path(args.output).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()

    if not collection_dir.is_dir():
        raise FileNotFoundError(f"Input collection does not exist: {collection_dir}")

    ensure_empty_output_directory(
        output_dir,
        args.overwrite,
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_files = list_metadata_files(
        collection_dir,
        file_order,
    )

    print(f"Metadata files found: {len(metadata_files)}")

    if file_order == "filesystem":
        print(
            "[INFO] Using filesystem traversal order. This reproduces the "
            "order used to construct the released snapshot but may not be "
            "portable across filesystem layouts."
        )

    duplicate_metadata, duplicate_groups = detect_duplicate_metadata(collection_dir, metadata_files)

    empty_metadata, invalid_resources, invalid_counters = detect_invalid_resources(
        collection_dir, metadata_files, minimum_data_rows
    )

    duplicate_resources = detect_duplicate_resources(
        collection_dir=collection_dir,
        metadata_files=metadata_files,
        duplicate_metadata=duplicate_metadata,
        empty_metadata=empty_metadata,
        invalid_resources=invalid_resources,
    )

    manifests = {
        "duplicate_metadata": duplicate_metadata,
        "empty_metadata": empty_metadata,
        "invalid_resources": invalid_resources,
        "duplicate_resources": duplicate_resources,
    }

    for key, values in manifests.items():
        write_json(
            work_dir / EXCLUSION_FILENAMES[key],
            values,
        )

    preparation = prepare_reduced_collection(
        collection_dir=collection_dir,
        output_dir=output_dir,
        metadata_files=metadata_files,
        duplicate_metadata=manifests["duplicate_metadata"],
        empty_metadata=manifests["empty_metadata"],
        invalid_resources=manifests["invalid_resources"],
        duplicate_resources=manifests["duplicate_resources"],
        sample_rows=sample_rows,
        random_seed=random_seed,
    )

    component_checks = normalized_component_checks(output_dir, file_order)
    validate_prepared_collection_contract(
        component_checks=component_checks,
        require_title=require_title,
        require_description=require_description,
        resources_per_dataset=resources_per_dataset,
    )

    report = {
        "input": str(collection_dir),
        "output": str(output_dir),
        "configuration": {
            "benchmark_config": str(Path(args.benchmark_config).expanduser().resolve()),
            "sample_rows": sample_rows,
            "random_seed": random_seed,
            "minimum_data_rows": minimum_data_rows,
            "file_order": file_order,
            "require_title": require_title,
            "require_description": require_description,
            "resources_per_dataset": resources_per_dataset,
        },
        "input_metadata_files": len(metadata_files),
        "exclusions": {
            "duplicate_metadata": len(manifests["duplicate_metadata"]),
            "empty_metadata": len(manifests["empty_metadata"]),
            "invalid_resources": len(manifests["invalid_resources"]),
            "duplicate_resources": len(manifests["duplicate_resources"]),
            "duplicate_metadata_groups": len(duplicate_groups),
        },
        "invalid_resource_scan": invalid_counters,
        "prepared_collection": {
            key: value for key, value in preparation.items() if key != "resource_rows"
        },
        "component_checks": {
            "metadata_records": component_checks["metadata_records"],
            "missing_title_count": len(component_checks["missing_title"]),
            "missing_description_count": len(component_checks["missing_description"]),
            "metadata_without_resources_count": len(component_checks["metadata_without_resources"]),
            "resource_count_mismatch_count": len(component_checks["resource_count_mismatches"]),
            "resources_without_schema_count": len(component_checks["resources_without_schema"]),
            "missing_title_examples": component_checks["missing_title"][:10],
            "missing_description_examples": component_checks["missing_description"][:10],
            "resource_count_mismatch_examples": component_checks["resource_count_mismatches"][:10],
        },
    }

    report_path = work_dir / "collection_preparation_report.json"
    write_json(report_path, report)
    print_preparation_summary(
        metadata_files=metadata_files,
        manifests=manifests,
        preparation=preparation,
        component_checks=component_checks,
        output_dir=output_dir,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
