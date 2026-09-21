"""Minimal historical search semantics over Qdrant."""

from __future__ import annotations

from typing import Any

from .common import (
    DATASET_FIELDS,
    HARRIER_MODEL_NAME,
    K_MAX,
    QWEN_MODEL_NAME,
    SPARSE_MODEL_NAME,
    TOP_N,
    preprocess_bm25_text,
    profile_config,
    query_dense_preprocess,
    query_prompt,
)


def _payload_value(payload: dict[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return str(value)
    return ""


def format_hit(dataset_id: str, payload: dict[str, Any], score: float) -> dict[str, Any]:
    """Return the minimal downstream response schema."""
    return {
        "id": _payload_value(payload, "id", "dataset_uid") or str(dataset_id),
        "title": _payload_value(payload, "title", "text"),
        "description": _payload_value(payload, "description", "desc", "description_text"),
        "header": _payload_value(payload, "header", "headers", "header_text"),
        "content": _payload_value(payload, "content", "rows", "content_text"),
        "metadato_fileName": _payload_value(payload, "metadato_fileName", "meta_fileName", "meta_json"),
        "resource_fileName": _payload_value(payload, "resource_fileName"),
        "score": float(score),
    }


def fuse_field_results(
    title_results: list[dict[str, Any]],
    description_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse title and description results by dataset ID using score sums."""
    candidates: dict[str, dict[str, Any]] = {}

    for field, results in (("title", title_results), ("description", description_results)):
        for item in results:
            payload = item.get("payload") or {}
            dataset_id = str(payload.get("dataset_uid") or payload.get("id") or item.get("id"))
            candidate = candidates.setdefault(
                dataset_id,
                {
                    "payload": payload,
                    "scores": {"title": 0.0, "description": 0.0},
                },
            )
            if payload and not candidate["payload"]:
                candidate["payload"] = payload
            score = float(item.get("score") or 0.0)
            if score > candidate["scores"][field]:
                candidate["scores"][field] = score

    hits = []
    for dataset_id, candidate in candidates.items():
        score = candidate["scores"]["title"] + candidate["scores"]["description"]
        hits.append(format_hit(dataset_id, candidate["payload"], score))

    hits.sort(key=lambda hit: float(hit.get("score", 0.0)), reverse=True)
    return hits[:TOP_N]


class RetrievalSearcher:
    """Search the three public benchmark profiles."""

    def __init__(self, qdrant_url: str):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=qdrant_url)
        self._dense_models: dict[str, Any] = {}
        self._sparse_model: Any | None = None

    def _load_dense_model(self, profile: str):
        if profile not in self._dense_models:
            from sentence_transformers import SentenceTransformer
            import torch

            model_name = HARRIER_MODEL_NAME if profile == "harrier" else QWEN_MODEL_NAME
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._dense_models[profile] = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )
        return self._dense_models[profile]

    def _load_sparse_model(self):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            self._sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        return self._sparse_model

    def _query_qdrant(
        self,
        collection_name: str,
        vector_name: str,
        vector: Any,
        sparse: bool,
    ) -> list[dict[str, Any]]:
        from qdrant_client import models

        query_vector = (
            models.NamedSparseVector(name=vector_name, vector=vector)
            if sparse
            else models.NamedVector(name=vector_name, vector=vector)
        )
        points = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=K_MAX,
            with_payload=True,
        )
        return [
            {
                "id": point.id,
                "score": float(point.score),
                "payload": point.payload or {},
            }
            for point in points
        ]

    def search_lexical(self, query: str) -> list[dict[str, Any]]:
        from qdrant_client import models

        config = profile_config("bm25")
        model = self._load_sparse_model()
        sparse_embedding = next(model.embed([preprocess_bm25_text(query)]))
        sparse_vector = models.SparseVector(
            indices=list(sparse_embedding.indices),
            values=list(sparse_embedding.values),
        )
        title = self._query_qdrant(
            config["collection"],
            config["title_vector"],
            sparse_vector,
            sparse=True,
        )
        description = self._query_qdrant(
            config["collection"],
            config["description_vector"],
            sparse_vector,
            sparse=True,
        )
        return fuse_field_results(title, description)

    def search_semantic(self, profile: str, query: str) -> list[dict[str, Any]]:
        config = profile_config(profile)
        model = self._load_dense_model(profile)
        normalized_query = query_dense_preprocess(query)
        embedding = model.encode(
            normalized_query,
            normalize_embeddings=True,
            prompt=query_prompt(),
        )
        vector = embedding.tolist()
        title = self._query_qdrant(
            config["collection"],
            config["title_vector"],
            vector,
            sparse=False,
        )
        description = self._query_qdrant(
            config["collection"],
            config["description_vector"],
            vector,
            sparse=False,
        )
        return fuse_field_results(title, description)


def ensure_clean_schema(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the public fields expected by the candidate-pool generator."""
    return [
        {field: hit.get(field, "") for field in DATASET_FIELDS} | {"score": float(hit["score"])}
        for hit in hits
    ]
