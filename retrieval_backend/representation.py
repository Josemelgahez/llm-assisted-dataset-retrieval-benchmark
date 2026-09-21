"""Historical dataset representation used by the retrieval indexes."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from .common import (
    MAX_CELL_CHARS,
    MAX_CONTENT_LINES,
    MAX_CONTENT_TEXT_CHARS,
    clip_for_model,
    document_dense_preprocess,
)


csv.field_size_limit(sys.maxsize)


def extract_multilingual_text(value: Any, preferred: tuple[str, ...] = ("es", "en")) -> str:
    """Extract multilingual text using the historical es -> en -> fallback order."""

    def first_non_empty(items: Iterable[Any]) -> str:
        for item in items:
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
        if "language" in value and "value" in value:
            text = str(value.get("value", "")).strip()
            if text:
                return text
        for language in preferred:
            language_value = value.get(language)
            if isinstance(language_value, list):
                text = first_non_empty(language_value)
                if text:
                    return text
            if isinstance(language_value, str) and language_value.strip():
                return language_value.strip()
        for item in value.values():
            if isinstance(item, list):
                text = first_non_empty(item)
                if text:
                    return text
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    if isinstance(value, list):
        for language in preferred:
            for item in value:
                if (
                    isinstance(item, dict)
                    and item.get("language") == language
                    and item.get("value")
                ):
                    text = str(item.get("value", "")).strip()
                    if text:
                        return text
        return first_non_empty(value)
    return ""


def iter_resources(metadata: dict[str, Any]):
    """Yield resources preserving the order already present in metadata."""
    resources = metadata.get("resources") or []
    if isinstance(resources, dict):
        for key, resource in resources.items():
            yield key, resource
    elif isinstance(resources, list):
        for index, resource in enumerate(resources):
            yield index, resource


def resource_name(resource: dict[str, Any]) -> str:
    return resource.get("fileName") or resource.get("name") or resource.get("title") or "Recurso"


def resource_path_value(resource: dict[str, Any]) -> str:
    return resource.get("path") or resource.get("downloadURL") or resource.get("accessURL") or ""


def resource_delimiter(resource: dict[str, Any]) -> str:
    delimiter = (resource.get("delimiter") or "").strip()
    name = (resource.get("fileName") or resource.get("path") or "").lower()
    if name.endswith(".tsv"):
        return "\t"
    return delimiter or ","


def extract_header(resource: dict[str, Any]) -> str:
    """Compose the historical schema-based header string."""
    schema = resource.get("schema") or {}
    fields = schema.get("fields") or []
    if not fields:
        return ""

    names = []
    for field in fields:
        name = str(field.get("name", "")).strip()
        if not name:
            continue
        if any(char in name for char in [" ", ";", ",", '"']):
            names.append(f'"{name}"')
        else:
            names.append(name)
    return resource_delimiter(resource).join(names)


def _append_limited(parts: list[str], current_length: int, text: str) -> int:
    if not text:
        return current_length
    remaining = MAX_CONTENT_TEXT_CHARS - current_length
    if remaining <= 0:
        return current_length
    addition = text + "\n"
    if len(addition) <= remaining:
        parts.append(addition)
        return current_length + len(addition)
    parts.append(addition[:remaining])
    return MAX_CONTENT_TEXT_CHARS


def _csv_path(collection_dir: Path, resource: dict[str, Any]) -> Path:
    filename = resource.get("fileName")
    if not filename:
        return collection_dir / str(resource.get("path") or "")
    return collection_dir / str(filename)


def extract_rows_text(collection_dir: Path, resource: dict[str, Any]) -> str:
    """Extract row payload text as in the historical indexer."""
    path = _csv_path(collection_dir, resource)
    delimiter = resource_delimiter(resource)
    encoding = resource.get("encoding") or "utf-8"
    parts: list[str] = []
    current_length = 0

    with path.open("r", encoding=encoding, errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"', escapechar="\\")
        fields = (resource.get("schema") or {}).get("fields") or []
        if fields:
            next(reader, None)
        rows_seen = 0
        for row in reader:
            rows_seen += 1
            raw_row = delimiter.join(str(cell)[:MAX_CELL_CHARS] for cell in row)
            current_length = _append_limited(parts, current_length, raw_row)
            if MAX_CONTENT_LINES and rows_seen >= MAX_CONTENT_LINES:
                break

    return "".join(parts)


def extract_columns_text(collection_dir: Path, resource: dict[str, Any]) -> str:
    """Extract column payload text as in the historical indexer."""
    path = _csv_path(collection_dir, resource)
    delimiter = resource_delimiter(resource)
    encoding = resource.get("encoding") or "utf-8"

    with path.open("r", encoding=encoding, errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"', escapechar="\\")
        rows = []
        for index, row in enumerate(reader):
            rows.append(row)
            if MAX_CONTENT_LINES and index + 1 >= MAX_CONTENT_LINES:
                break

    if not rows:
        return ""

    num_columns = max(len(row) for row in rows)
    columns = [[] for _ in range(num_columns)]
    for row in rows:
        for index in range(num_columns):
            columns[index].append(row[index] if index < len(row) else "")

    parts: list[str] = []
    current_length = 0
    for column in columns:
        column_text = " ".join(str(cell)[:MAX_CELL_CHARS] for cell in column)
        column_text = document_dense_preprocess(clip_for_model(column_text)).lower()
        if column_text:
            current_length = _append_limited(parts, current_length, column_text)
    return "".join(parts)


def compose_content(rows_text: str, columns_text: str) -> str:
    rows_text = (rows_text or "").strip()
    columns_text = (columns_text or "").strip()
    if rows_text and columns_text:
        return rows_text if rows_text == columns_text else f"{rows_text}\n{columns_text}"
    return rows_text or columns_text


def select_first_valid_resource(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first resource with a path, preserving SOLO_UN_RECURSO=True."""
    for _, resource in iter_resources(metadata):
        if isinstance(resource, dict) and resource.get("path"):
            return resource
    return None


def build_dataset_record(metadata_path: Path, collection_dir: Path | None = None) -> dict[str, str] | None:
    """Build the payload and index texts for one prepared metadata record."""
    collection_dir = collection_dir or metadata_path.parent
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    dataset_id = metadata.get("identifier") or metadata_path.name
    title = extract_multilingual_text(metadata.get("title"))
    description = extract_multilingual_text(metadata.get("description"))
    if not title or not description:
        return None

    resource = select_first_valid_resource(metadata)
    header = ""
    content = ""
    resource_file_name = ""
    if resource:
        resource_file_name = resource_name(resource)
        delimiter = resource_delimiter(resource)
        header_raw = extract_header(resource)
        header = clip_for_model(
            document_dense_preprocess(header_raw.replace(delimiter, " ")).lower()
        )
        try:
            rows = clip_for_model(
                document_dense_preprocess(extract_rows_text(collection_dir, resource)).lower()
            )
        except Exception:
            rows = ""
        try:
            columns = clip_for_model(
                document_dense_preprocess(extract_columns_text(collection_dir, resource)).lower()
            )
        except Exception:
            columns = ""
        content = compose_content(rows, columns)

    return {
        "id": str(dataset_id),
        "title": title,
        "description": description,
        "header": header,
        "content": content,
        "metadato_fileName": str(metadata.get("fileName") or metadata_path.name),
        "resource_fileName": str(resource_file_name),
        "dense_title_text": clip_for_model(document_dense_preprocess(title)),
        "dense_description_text": clip_for_model(document_dense_preprocess(description)),
    }


def iter_dataset_records(collection_dir: Path):
    """Yield historical dataset records from a prepared collection."""
    for metadata_path in sorted(collection_dir.glob("meta_*.json")):
        record = build_dataset_record(metadata_path, collection_dir=collection_dir)
        if record is not None:
            yield record
