"""Index the minimal benchmark retrieval profiles in local Qdrant."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, Literal

from .common import (
    BM25_COLLECTION,
    BM25_DESC_VECTOR,
    BM25_TITLE_VECTOR,
    HARRIER_COLLECTION,
    HARRIER_DESC_VECTOR,
    HARRIER_MODEL_NAME,
    HARRIER_TITLE_VECTOR,
    QWEN_COLLECTION,
    QWEN_DESC_VECTOR,
    QWEN_MODEL_NAME,
    QWEN_TITLE_VECTOR,
    SPARSE_MODEL_NAME,
    preprocess_bm25_text,
)
from .representation import iter_dataset_records

POINT_NAMESPACE = uuid.NAMESPACE_DNS


def point_id(dataset_id: str) -> str:
    """Historical deterministic UUID derived from the dataset identifier."""
    return str(uuid.uuid5(POINT_NAMESPACE, str(dataset_id)))


def clean_payload(record: dict[str, str]) -> dict[str, str]:
    """Return only the public payload fields needed by downstream stages."""
    return {
        "dataset_uid": record["id"],
        "id": record["id"],
        "title": record["title"],
        "description": record["description"],
        "header": record["header"],
        "headers": record["header"],
        "content": record["content"],
        "rows": record["content"],
        "metadato_fileName": record["metadato_fileName"],
        "meta_fileName": record["metadato_fileName"],
        "resource_fileName": record["resource_fileName"],
    }


def recreate_lexical_collection(client, collection_name: str = BM25_COLLECTION) -> None:
    """Recreate the sparse BM25 collection with IDF weighting."""
    from qdrant_client import models

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            BM25_TITLE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
            BM25_DESC_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )


def recreate_dense_collection(
    client,
    collection_name: str,
    title_vector: str,
    description_vector: str,
    dimension: int,
) -> None:
    """Recreate one dense profile collection with cosine distance."""
    from qdrant_client import models

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            title_vector: models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
            description_vector: models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        },
    )


def _to_sparse_vector(sparse_embedding) -> models.SparseVector:
    from qdrant_client import models

    return models.SparseVector(
        indices=list(sparse_embedding.indices),
        values=list(sparse_embedding.values),
    )


def index_lexical(
    client,
    records: Iterable[dict[str, str]],
    collection_name: str = BM25_COLLECTION,
    batch_size: int = 64,
) -> None:
    """Index title and description BM25 vectors."""
    from fastembed import SparseTextEmbedding
    from qdrant_client import models

    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    batch: list[dict[str, str]] = []

    def flush() -> None:
        if not batch:
            return
        title_embeddings = list(
            sparse_model.embed([preprocess_bm25_text(record["title"]) for record in batch])
        )
        desc_embeddings = list(
            sparse_model.embed([preprocess_bm25_text(record["description"]) for record in batch])
        )
        points = []
        for record, title_embedding, desc_embedding in zip(batch, title_embeddings, desc_embeddings):
            points.append(
                models.PointStruct(
                    id=point_id(record["id"]),
                    payload=clean_payload(record),
                    vector={
                        BM25_TITLE_VECTOR: _to_sparse_vector(title_embedding),
                        BM25_DESC_VECTOR: _to_sparse_vector(desc_embedding),
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points, wait=True)
        batch.clear()

    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            flush()
    flush()


def load_sentence_transformer(model_name: str):
    """Load a historical dense embedding model."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, trust_remote_code=True, device=device)


def index_dense_profile(
    client,
    records: Iterable[dict[str, str]],
    collection_name: str,
    title_vector: str,
    description_vector: str,
    model_name: str,
    batch_size: int = 4,
    encoder_method: Literal["auto", "encode_document", "encode"] = "auto",
) -> None:
    """Index title and description dense vectors for one semantic profile."""
    from qdrant_client import models

    model = load_sentence_transformer(model_name)
    batch: list[dict[str, str]] = []

    def encode_documents(texts: list[str]):
        if encoder_method not in {"auto", "encode_document", "encode"}:
            raise ValueError(f"Unknown dense encoder method: {encoder_method}")
        if encoder_method == "encode_document" and not hasattr(model, "encode_document"):
            raise RuntimeError(
                f"Model {model_name!r} does not expose encode_document()."
            )
        if encoder_method in {"auto", "encode_document"} and hasattr(model, "encode_document"):
            return model.encode_document(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def flush() -> None:
        if not batch:
            return
        title_embeddings = encode_documents([record["dense_title_text"] for record in batch])
        desc_embeddings = encode_documents(
            [record["dense_description_text"] for record in batch]
        )
        points = []
        for record, title_embedding, desc_embedding in zip(batch, title_embeddings, desc_embeddings):
            points.append(
                models.PointStruct(
                    id=point_id(record["id"]),
                    payload=clean_payload(record),
                    vector={
                        title_vector: title_embedding.tolist(),
                        description_vector: desc_embedding.tolist(),
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points, wait=True)
        batch.clear()

    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            flush()
    flush()


def build_indexes(
    input_dir: Path,
    qdrant_url: str,
    profiles: set[str],
    batch_size: int = 64,
    dense_encoder_method: Literal["auto", "encode_document", "encode"] = "auto",
) -> None:
    """Build the requested benchmark indexes in local Qdrant."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)

    if "bm25" in profiles:
        recreate_lexical_collection(client)
        index_lexical(client, iter_dataset_records(input_dir), batch_size=batch_size)

    if "harrier" in profiles:
        model = load_sentence_transformer(HARRIER_MODEL_NAME)
        dimension = model.get_sentence_embedding_dimension()
        recreate_dense_collection(
            client,
            HARRIER_COLLECTION,
            HARRIER_TITLE_VECTOR,
            HARRIER_DESC_VECTOR,
            int(dimension),
        )
        del model
        index_dense_profile(
            client,
            iter_dataset_records(input_dir),
            HARRIER_COLLECTION,
            HARRIER_TITLE_VECTOR,
            HARRIER_DESC_VECTOR,
            HARRIER_MODEL_NAME,
            batch_size=min(batch_size, 4),
            encoder_method=dense_encoder_method,
        )

    if "qwen" in profiles:
        model = load_sentence_transformer(QWEN_MODEL_NAME)
        dimension = model.get_sentence_embedding_dimension()
        recreate_dense_collection(
            client,
            QWEN_COLLECTION,
            QWEN_TITLE_VECTOR,
            QWEN_DESC_VECTOR,
            int(dimension),
        )
        del model
        index_dense_profile(
            client,
            iter_dataset_records(input_dir),
            QWEN_COLLECTION,
            QWEN_TITLE_VECTOR,
            QWEN_DESC_VECTOR,
            QWEN_MODEL_NAME,
            batch_size=min(batch_size, 4),
            encoder_method=dense_encoder_method,
        )
