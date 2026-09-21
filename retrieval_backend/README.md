# Retrieval backend for the datos.gob.es benchmark

This directory contains the retrieval backend used to reconstruct the Stage 2 candidate pool of the benchmark.

The implementation provides the indexing and search functionality required by the three retrieval profiles used in the experiment.

## Retrieval profiles

- `bm25`: BM25 sparse retrieval over `title` and `description`, using Qdrant `Modifier.IDF`.
- `harrier`: dense cosine retrieval with `microsoft/harrier-oss-v1-0.6b`.
- `qwen`: dense cosine retrieval with `Qwen/Qwen3-Embedding-0.6B`.

All three profiles search `title` and `description` separately and combine the two scores as:

```text
final score = title score + description score
```

`header` and `content` are returned as payload fields for later relevance assessment, but they do not affect retrieval ranking.

## Local Qdrant

The retrieval backend uses a local Qdrant instance.

Start Qdrant with:

```bash
docker run --rm \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:v1.15.5
```

The default URL is:

```text
http://localhost:6333
```

The released setup uses Qdrant `v1.15.5` together with `qdrant-client==1.15.1`.

## Build indexes

Use the prepared collection produced by:

```text
src/01_prepare_collection.py
```

Build the three retrieval indexes with:

```bash
python scripts/retrieval/build_retrieval_indexes.py \
  --input /path/to/prepared_collection \
  --profiles bm25,harrier,qwen \
  --qdrant-url http://localhost:6333
```

This creates the following Qdrant collections:

```text
datasets_bm25
datasets_harrier
datasets_qwen
```

## Run the API

Start the retrieval API with:

```bash
python scripts/retrieval/run_retrieval_api.py \
  --qdrant-url http://localhost:6333 \
  --host 127.0.0.1 \
  --port 8002
```

## Query endpoints

The three retrieval profiles are exposed through:

```text
GET /search/bm25
GET /search/harrier/semantic
GET /search/qwen/semantic
```

Example requests:

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

Results are sorted by descending retrieval score:

```python
hits.sort(
    key=lambda hit: hit["score"],
    reverse=True,
)
```

No secondary dataset-ID tie-break is applied.

## Candidate-pool generation

Once the API is running, Stage 2 can query the three profiles with:

```bash
python src/02-search-results-generation.py \
  --queries data/queries/queries.csv \
  --endpoints \
    bm25=http://127.0.0.1:8002/search/bm25,harrier=http://127.0.0.1:8002/search/harrier/semantic,qwen=http://127.0.0.1:8002/search/qwen/semantic \
  --output data/candidate_pool
```

The pooling depth is defined in:

```text
config/benchmark_config.json
```

For the released benchmark configuration, Stage 2 produces:

```text
Queries:                    100
Unique candidate pairs:  11,779

Profile memberships:
  BM25:                   3,832
  Harrier:                5,000
  Qwen:                   5,000
```