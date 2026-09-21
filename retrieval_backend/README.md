# Clean retrieval backend for the datos.gob.es benchmark

This directory contains a minimal public-artifact implementation of the indexing and search backend used to build the Stage 2 candidate pool.

It preserves the historical experiment semantics while removing unrelated private infrastructure: no API keys, no TLS certificates, no Redis, no reranker, no generative model, no intent classifier, no cache, and no OpenSearch logging.

## Retrieval profiles

- `bm25`: BM25 sparse retrieval over `title` and `description`, using Qdrant `Modifier.IDF`.
- `harrier`: dense cosine retrieval with `microsoft/harrier-oss-v1-0.6b`.
- `qwen`: dense cosine retrieval with `Qwen/Qwen3-Embedding-0.6B`.

All three profiles search `title` and `description` separately and compute:

```text
final score = title score + description score
```

`header` and `content` are returned as payload fields for later LLM judging, but they do not affect ranking.

## Local Qdrant

Example:

```bash
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.15.5
```

The default URL is:

```text
http://localhost:6333
```

No authentication, API key, certificate, or TLS setting is used.

The Qdrant server version is pinned intentionally. The artifact dependencies use
`qdrant-client==1.15.1`, which is compatible with Qdrant 1.15.x. Avoid using the
floating `latest` tag, as newer servers can trigger client/server compatibility
warnings and may change near-tie ordering in reconstructed indexes.

## Build indexes

Use the prepared collection produced by `src/01_prepare_collection.py`.

```bash
python scripts/retrieval/build_retrieval_indexes.py \
  --input /path/to/prepared_collection \
  --profiles bm25,harrier,qwen \
  --qdrant-url http://localhost:6333
```

This creates the historical collection names:

- `datasets_bm25`
- `datasets_harrier`
- `datasets_qwen`

## Run the API

```bash
python scripts/retrieval/run_retrieval_api.py \
  --qdrant-url http://localhost:6333 \
  --host 127.0.0.1 \
  --port 8002
```

## Query endpoints

```bash
curl 'http://127.0.0.1:8002/search/bm25?q=cartografia'
curl 'http://127.0.0.1:8002/search/harrier/semantic?q=cartografia'
curl 'http://127.0.0.1:8002/search/qwen/semantic?q=cartografia'
```

Each endpoint returns:

```json
{
  "hits": [
    {
      "id": "...",
      "title": "...",
      "description": "...",
      "header": "...",
      "content": "...",
      "metadato_fileName": "...",
      "resource_fileName": "...",
      "score": 1.234
    }
  ]
}
```

Results are sorted exactly as in the historical backend:

```python
hits.sort(key=lambda hit: hit["score"], reverse=True)
```

No secondary dataset-ID tie-break is added.

## Candidate-pool generation

Point the clean candidate-pool generator to the three endpoints above and keep `top_k=50`.

The historical validation targets are:

- queries: 100
- unique candidate pairs: 11,779
- bm25 memberships: 3,832
- harrier memberships: 5,000
- qwen memberships: 5,000

These numbers are validation targets only; they are not hard-coded in this backend.