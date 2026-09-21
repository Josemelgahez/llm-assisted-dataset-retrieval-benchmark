import argparse
import csv
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Real
from pathlib import Path

import requests
from tqdm import tqdm

from pipeline_utils import load_benchmark_config, require_positive_int

DATASET_FIELDS = (
    "id",
    "title",
    "description",
    "header",
    "content",
    "metadato_fileName",
    "resource_fileName",
)


# All profiles returning the same dataset ID must expose the same dataset
# representation; only retrieval rank and score may differ across profiles.
REPRESENTATION_FIELDS = tuple(field for field in DATASET_FIELDS if field != "id")


def parse_endpoints(endpoints_argument):
    """
    Parse one or more named retrieval endpoints.

    Format: name1=url1,name2=url2
    """
    if not endpoints_argument:
        raise ValueError(
            "No retrieval endpoints supplied. Use --endpoints name=url[,name=url...]."
        )

    endpoints = []
    seen_names = set()

    raw_endpoints = [value.strip() for value in endpoints_argument.split(",")]

    for raw_endpoint in raw_endpoints:
        if not raw_endpoint or "=" not in raw_endpoint:
            raise ValueError(
                "Every retrieval endpoint must use name=url format. "
                f"Invalid item: {raw_endpoint!r}"
            )

        name, url = raw_endpoint.split("=", 1)
        name = name.strip()
        url = url.strip()

        if not name:
            raise ValueError(f"Retrieval profile name cannot be empty: {raw_endpoint!r}")

        if not url:
            raise ValueError(
                f"Retrieval endpoint URL cannot be empty for profile {name!r}."
            )

        if name in seen_names:
            raise ValueError(f"Duplicate retrieval profile name: {name!r}")

        seen_names.add(name)
        endpoints.append({"name": name, "url": url})

    return endpoints


def _extract_hits(response_data):
    """Extract retrieval hits from the retrieval backend response."""
    if not isinstance(response_data, dict):
        raise ValueError("Search response must be a JSON object.")

    if "hits" not in response_data:
        raise ValueError("Search response must contain a 'hits' field.")

    hits = response_data["hits"]

    if not isinstance(hits, list):
        raise ValueError("Search response field 'hits' must be a list.")

    for index, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            raise ValueError(f"Search response hit #{index} must be an object.")

    return hits


def prepare_candidate_pool_output(output_dir, overwrite):
    """Protect the candidate-pool output from stale query_*.json files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Only Stage 2 query outputs are managed here; unrelated files in the
    # destination directory are left untouched.
    existing_query_files = sorted(output_path.glob("query_*.json"))

    if existing_query_files and not overwrite:
        raise FileExistsError(
            f"{output_path} already contains {len(existing_query_files)} "
            "query_*.json file(s). Re-run with --overwrite to replace only "
            "those files."
        )

    return existing_query_files


def _is_missing(value):
    """Return True for missing or empty string values."""
    return value is None or (isinstance(value, str) and not value.strip())


def load_query_table(path):
    """Load the query CSV/TSV and return its rows and column names."""
    path = Path(path)

    with path.open(encoding="utf-8", newline="") as file:
        sample = file.read(4096)

        if not sample:
            raise ValueError(f"Query file is empty: {path}")

        file.seek(0)
        # Infer comma/tab separation from the input rather than requiring a
        # separate CLI option for CSV and TSV query files.
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(file, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError(f"Query file has no header row: {path}")

        return list(reader), list(reader.fieldnames)


def _normalize_query_id(value, row_number):
    """Parse a query-table ID as a positive integer."""
    if _is_missing(value):
        raise ValueError(f"Invalid ID at row {row_number}: value is missing.")

    text = value.strip()

    if not text.isdecimal():
        raise ValueError(
            f"Invalid ID at row {row_number}: {value!r} is not an integer."
        )

    normalized = int(text)

    if normalized <= 0:
        raise ValueError(
            f"Invalid ID at row {row_number}: {normalized!r} must be greater than 0."
        )

    return normalized


def validate_queries(rows, columns, queries_path):
    """Validate and normalize the query table used for retrieval."""
    required_columns = {"ID", "QUERY"}
    missing_columns = sorted(required_columns - set(columns))

    if missing_columns:
        raise ValueError(
            f"Missing required query column(s) in {queries_path}: "
            f"{missing_columns}. Detected columns: {columns}"
        )

    normalized_rows = []
    seen_ids = {}
    # Normalize query IDs and text while enforcing unique positive IDs.
    for row_number, row in enumerate(rows, start=2):
        query_id = _normalize_query_id(row["ID"], row_number)

        if query_id in seen_ids:
            raise ValueError(
                f"Duplicate query ID {query_id!r} at rows "
                f"{seen_ids[query_id]} and {row_number}."
            )

        seen_ids[query_id] = row_number

        query_value = row["QUERY"]

        if _is_missing(query_value):
            raise ValueError(
                f"Invalid QUERY at row {row_number}: value is missing or empty."
            )

        # All profiles returning the same dataset ID must expose the same dataset
        # representation; only retrieval rank and score may differ across profiles.
        normalized_rows.append(
            {
                "ID": query_id,
                "QUERY": query_value.strip(),
            }
        )

    if not normalized_rows:
        raise ValueError("Query file contains no query rows.")

    return normalized_rows


def _validated_dataset_fields(hit, profile_name, rank):
    """
    Validate and keep the dataset fields required by the candidate pool and
    subsequent annotation stages.
    """
    missing_fields = [field for field in DATASET_FIELDS if field not in hit]

    if missing_fields:
        raise ValueError(
            f"Hit #{rank} from profile {profile_name!r} is missing required "
            f"dataset field(s): {missing_fields}."
        )

    dataset_id = hit["id"]

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(
            f"Hit #{rank} from profile {profile_name!r} must have a non-empty "
            "string dataset id."
        )

    if "score" not in hit:
        raise ValueError(
            f"Hit #{rank} from profile {profile_name!r} is missing score."
        )

    # Retrieval scores are preserved as returned by the backend, but must be
    # finite numeric values so provenance remains valid and serializable.
    score = hit["score"]

    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError(
            f"Hit #{rank} from profile {profile_name!r} has non-numeric score: "
            f"{score!r}."
        )

    numeric_score = float(score)

    if not math.isfinite(numeric_score):
        raise ValueError(
            f"Hit #{rank} from profile {profile_name!r} has invalid score: "
            f"{score!r}."
        )

    return {field: hit[field] for field in DATASET_FIELDS}, score


def _merge_hits_with_retrieval(hits_by_endpoint):
    """
    Merge the retained results from all retrieval profiles by dataset ID.

    Each dataset appears once in the resulting candidate pool.

    For every retrieval profile that contributed the dataset, the output
    preserves:
      - rank: the position returned by that endpoint.
      - score: the score returned by that endpoint.

    This function expects each endpoint list to have already been truncated
    to its top-k results. Therefore, every profile stored under `retrieval`
    represents an actual contribution to the candidate pool.
    """
    merged_hits = {}
    hit_order = []
    # Merge profiles in endpoint order so the first occurrence of each dataset
    # determines its position in the final candidate pool.
    for endpoint_name, hits in hits_by_endpoint.items():
        seen_dataset_ids = set()

        for rank, hit in enumerate(hits, start=1):
            dataset_fields, score = _validated_dataset_fields(
                hit,
                endpoint_name,
                rank,
            )

            dataset_id = dataset_fields["id"]

            if dataset_id in seen_dataset_ids:
                raise ValueError(
                    f"Profile {endpoint_name!r} returned duplicate dataset ID "
                    f"{dataset_id!r}."
                )

            seen_dataset_ids.add(dataset_id)

            if dataset_id not in merged_hits:
                merged_hit = dataset_fields
                merged_hit["retrieval"] = {}
                merged_hits[dataset_id] = merged_hit
                hit_order.append(dataset_id)

            else:
                merged_hit = merged_hits[dataset_id]
                # The same dataset ID must expose the same representation
                # regardless of the retrieval profile that returned it.
                for field in REPRESENTATION_FIELDS:
                    if merged_hit[field] != dataset_fields[field]:
                        existing_profiles = ", ".join(
                            merged_hit["retrieval"].keys()
                        )
                        raise ValueError(
                            f"Inconsistent representation for dataset ID "
                            f"{dataset_id!r}, field {field!r}: profiles "
                            f"{existing_profiles!r} returned "
                            f"{merged_hit[field]!r}, but profile "
                            f"{endpoint_name!r} returned {dataset_fields[field]!r}."
                        )

            # Store profile-specific provenance without duplicating the dataset
            # representation in the pooled candidate record.
            merged_hit["retrieval"][endpoint_name] = {
                "rank": rank,
                "score": score,
            }

    return [merged_hits[dataset_id] for dataset_id in hit_order]


def _process_query(
    row,
    endpoints,
    output_dir,
    auth_token,
    pooling_depth,
):
    """
    Execute one query against all retrieval profiles and directly build
    its diversified candidate pool using the configured pooling depth.
    """
    output_dir = Path(output_dir)
    query_value = row["QUERY"]
    query_id = row["ID"]

    params = {"q": query_value}
    headers = {"Accept": "application/json"}

    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    hits_by_endpoint = {}
    # Profiles are queried sequentially within each query. Parallelism is
    # applied across queries, so endpoint order remains deterministic.
    for endpoint in endpoints:
        endpoint_name = endpoint["name"]
        endpoint_url = endpoint["url"]

        try:
            response = requests.get(
                endpoint_url,
                headers=headers,
                params=params,
                timeout=300,
            )

            response.raise_for_status()

            response_data = response.json()
            hits = _extract_hits(response_data)

        except Exception as exc:
            raise RuntimeError(
                f"Query {query_value!r} failed for retrieval profile "
                f"{endpoint_name!r} at {endpoint_url!r}: {exc}"
            ) from exc

        # Truncate before merging so the candidate pool is exactly the union
        # of each profile's configured top-k contribution.
        hits_by_endpoint[endpoint_name] = hits[:pooling_depth]

    candidate_hits = _merge_hits_with_retrieval(hits_by_endpoint)

    candidate_pool = {
        "id_query": query_id,
        "query": query_value,
        "hits": candidate_hits,
    }

    result_path = output_dir / f"query_{query_id}.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(candidate_pool, file, indent=2, ensure_ascii=False)


def validate_temporary_outputs(temporary_output_dir, query_rows):
    """Verify that every input query produced exactly one output file."""
    temporary_output_path = Path(temporary_output_dir)

    expected_filenames = {f"query_{row['ID']}.json" for row in query_rows}
    generated_filenames = {
        path.name for path in temporary_output_path.glob("query_*.json")
    }
    # Do not replace the existing candidate pool unless every query produced
    # exactly one expected output file.
    if generated_filenames != expected_filenames:
        missing = sorted(expected_filenames - generated_filenames)
        extra = sorted(generated_filenames - expected_filenames)

        raise RuntimeError(
            "Temporary candidate-pool output is incomplete or inconsistent. "
            f"Missing files: {missing}. Extra files: {extra}."
        )


def positive_int(value):
    """Parse a positive integer for argparse."""
    try:
        value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than 0") from exc

    try:
        return require_positive_int(value, "--concurrency")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args():
    """Parse Stage 2 candidate-pool generation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run queries against multiple retrieval profiles and build a diversified candidate pool."
    )

    parser.add_argument("--endpoints", required=True, help="Comma-separated retrieval endpoints in name=url format.")
    parser.add_argument("--queries", "-q", default="data/queries/queries.csv", help="CSV or TSV file containing the queries.")
    parser.add_argument("--output", "-o", default="data/candidate_pool", help="Output directory for the generated candidate pool.")
    parser.add_argument("--concurrency", "-c", type=positive_int, default=1, help="Number of queries executed concurrently.")
    parser.add_argument("--auth", "-a", default=None, help='Authorization token for protected APIs, without the "Bearer " prefix.')
    parser.add_argument("--benchmark-config", default="config/benchmark_config.json", help="Benchmark configuration JSON containing the Stage 2 settings.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing query_*.json files in the output directory.")

    return parser.parse_args()


def print_generation_summary(output_dir, endpoints, elapsed_time):
    """Print summary statistics from the generated candidate pool."""
    print(f"Total execution time: {elapsed_time:.2f} s")

    total_candidate_pairs = 0
    queries_with_results = 0
    profile_counts = {endpoint["name"]: 0 for endpoint in endpoints}

    result_files = sorted(Path(output_dir).glob("query_*.json"))
    # Re-read the published files so the summary reflects the actual candidate
    # pool on disk rather than intermediate in-memory results.
    for result_file in result_files:
        with result_file.open(encoding="utf-8") as file:
            data = json.load(file)

        hits = data.get("hits", [])
        total_candidate_pairs += len(hits)

        if hits:
            queries_with_results += 1

        for hit in hits:
            for profile_name in hit.get("retrieval", {}):
                if profile_name in profile_counts:
                    profile_counts[profile_name] += 1

    print("\nSummary:")
    print(f" - Unique candidate pairs: {total_candidate_pairs}")
    print(
        f" - Queries with at least one result: "
        f"{queries_with_results} of {len(result_files)}"
    )
    print(" - Retained pairs by retrieval profile:")

    for profile_name, count in profile_counts.items():
        print(f"   - {profile_name}: {count}")

    print("=" * 80)


def main():
    """Run Stage 2 retrieval and diversified candidate-pool construction."""
    args = parse_args()
    start_time = time.time()

    endpoints = parse_endpoints(args.endpoints)

    queries_path = args.queries

    output_dir = args.output
    concurrency = args.concurrency
    auth_token = args.auth

    benchmark_config = load_benchmark_config(args.benchmark_config)
    candidate_pool_config = benchmark_config["candidate_pool"]
    pooling_depth = candidate_pool_config["pooling_depth"]

    configured_profiles = list(candidate_pool_config["profiles"])
    endpoint_profiles = [
        endpoint["name"]
        for endpoint in endpoints
    ]

    if endpoint_profiles != configured_profiles:
        raise ValueError(
            "Retrieval endpoint profiles must match the configured profiles "
            "in the same order: "
            f"{endpoint_profiles} != {configured_profiles}"
        )

    query_rows_raw, query_columns = load_query_table(queries_path)

    query_rows = validate_queries(query_rows_raw, query_columns, queries_path)

    existing_query_files = prepare_candidate_pool_output(output_dir, args.overwrite)

    print(
        f"[ENDPOINTS] Running {len(endpoints)} retrieval profile(s): "
        f"{[endpoint['name'] for endpoint in endpoints]}"
    )

    print(f"[POOLING] Top-k per retrieval profile: {pooling_depth}")

    output_path = Path(output_dir)
    # Generate the complete candidate pool in a temporary directory so an
    # incomplete run cannot overwrite a previously valid output.
    with tempfile.TemporaryDirectory(
        prefix="candidate_pool_",
        dir=str(output_path.parent),
    ) as temporary_output_dir:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _process_query,
                    row,
                    endpoints,
                    temporary_output_dir,
                    auth_token,
                    pooling_depth,
                )
                for row in query_rows
            ]

            try:
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing queries",
                ):
                    future.result()
            except Exception:
                for future in futures:
                    future.cancel()
                raise

        validate_temporary_outputs(
            temporary_output_dir,
            query_rows,
        )
        # Publish the new candidate pool only after successful generation and
        # validation of all query outputs.
        if args.overwrite:
            for query_file in existing_query_files:
                query_file.unlink()

        for result_path in sorted(Path(temporary_output_dir).glob("query_*.json")):
            result_path.replace(output_path / result_path.name)

    print_generation_summary(
        output_dir,
        endpoints,
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()