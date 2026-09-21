# LLM-Assisted Benchmark Construction for Dataset Retrieval

This repository contains the artifact accompanying our work on scalable benchmark construction for heterogeneous dataset retrieval using multi-LLM relevance annotation and human validation.

The artifact provides:

* the processed `datos.gob.es` collection used in the experiments;
* the query set and pooled retrieval candidates;
* relevance annotations produced by five LLM judges;
* the human-annotated subset and individual human annotations;
* scripts implementing the complete benchmark-construction pipeline;
* the final assembled benchmark;
* analysis scripts used to reproduce the main experimental results;
* a minimal retrieval backend for reconstructing the candidate pool.

The final benchmark contains **100 queries** and **11,779 judged query-dataset pairs**, including a **300-pair human-annotated subset**.

---

## Repository structure

```text
.
├── src/
│   ├── 01_prepare_collection.py
│   ├── 02-search-results-generation.py
│   ├── 03-search-results-evaluation.py
│   ├── 04-judge-aggregation.py
│   ├── generate_human_annotation_subset.py
│   ├── 05-human-evaluation.py
│   ├── 06-assemble-benchmark.py
│   └── queries_llm.py
│
├── scripts/
│   ├── analysis/
│   │   ├── analyze_pool_diversity.py
│   │   ├── analyze_annotation_statistics.py
│   │   └── evaluate_retrieval.py
│   │
│   └── retrieval/
│       ├── build_retrieval_indexes.py
│       └── run_retrieval_api.py
│
├── retrieval_backend/
│   ├── README.md
│   └── ...
│
├── config/
│   ├── benchmark_config.json
│   └── models.json
│
├── data/
│   ├── normalized_collection/
│   ├── queries/
│   ├── candidate_pool/
│   ├── llm_annotations/
│   ├── human_annotations/
│   ├── derived/
│   └── benchmark/
│
└── results/
```

The repository separates three types of executable code:

* `src/`: the benchmark-construction pipeline;
* `scripts/analysis/`: scripts that reproduce analyses and numerical results reported in the paper;
* `scripts/retrieval/`: command-line entry points for building and running the local retrieval backend.

The retrieval backend implementation itself is contained in `retrieval_backend/`.

Unless otherwise noted, commands in this README are intended to be run from the repository root.

---

## Benchmark-construction pipeline

The methodology consists of **five conceptual stages** implemented through **six numbered pipeline scripts**.

Conceptual Stage 4 is implemented by two scripts: judge aggregation (`04`) and human evaluation (`05`).

A separate reproducibility utility, `generate_human_annotation_subset.py`, reconstructs the human-annotation subset used in Conceptual Stage 4.

| Script                            | Conceptual stage | Purpose                                     |
| --------------------------------- | ---------------- | ------------------------------------------- |
| `01_prepare_collection.py`        | Stage 1          | Collection normalization and representation |
| `02-search-results-generation.py` | Stage 2          | Diversified candidate pooling               |
| `03-search-results-evaluation.py` | Stage 3          | Multi-LLM relevance annotation              |
| `04-judge-aggregation.py`         | Stage 4          | Judge aggregation                           |
| `05-human-evaluation.py`          | Stage 4          | Human evaluation and aggregation assessment |
| `06-assemble-benchmark.py`        | Stage 5          | Benchmark assembly and release              |

### Stage 1 — Collection normalization and representation

```text
src/01_prepare_collection.py
```

Stage 1 prepares the dataset collection used by the retrieval pipeline.

The original `datos.gob.es` crawl contained **116,272 metadata records**. After filtering and normalization, the searchable collection contains **56,983 datasets**.

For tabular resources, at most 20 rows are sampled without replacement using seed 0, while preserving their original relative order.

The artifact distributes the processed collection used by the retrieval pipeline, not the complete raw portal dump.

The processed collection is available on Figshare:

```text
https://doi.org/10.6084/m9.figshare.33952177
```

Example:

```bash
python src/01_prepare_collection.py --help
```

See the command-line help for the available input and output options.

---

### Stage 2 — Diversified candidate pooling

```text
src/02-search-results-generation.py
```

Stage 2 queries multiple retrieval profiles and merges their top-ranked results into one candidate pool per query.

The experiment uses three retrieval profiles:

* BM25 lexical retrieval;
* Harrier (`microsoft/harrier-oss-v1-0.6b`);
* Qwen (`Qwen/Qwen3-Embedding-0.6B`).

Each profile contributes its top 50 results.

Candidate datasets are deduplicated by dataset identifier while preserving the retrieval provenance of every contributing profile.

Example:

```bash
python src/02-search-results-generation.py \
  --queries data/queries/queries.csv \
  --endpoints \
    bm25=http://127.0.0.1:8002/search/bm25,harrier=http://127.0.0.1:8002/search/harrier/semantic,qwen=http://127.0.0.1:8002/search/qwen/semantic \
  --output data/candidate_pool
```

The local backend required for reconstructing these endpoints is described in [`retrieval_backend/README.md`](retrieval_backend/README.md).

The released candidate pool contains **11,779 unique query-dataset pairs**.

---

### Stage 3 — Multi-LLM relevance annotation

```text
src/03-search-results-evaluation.py
```

Stage 3 annotates each pooled query-dataset pair using the LLM judges selected in:

```text
config/benchmark_config.json
```

Execution platforms and generation settings for these models are defined in:

```text
config/models.json
```

The experiment uses:

* `gpt-oss:20b`
* `gpt-4.1-mini`
* `gemma4:26b`
* `gemini-2.5-flash`
* `gpt-5.4-mini`

For each pair, the judge assigns one of five reasons:

```text
direct_match
subtype
generic
unrelated
insufficient_evidence
```

Binary relevance is derived deterministically:

```text
direct_match          → relevant
subtype               → relevant
generic               → not relevant
unrelated             → not relevant
insufficient_evidence → not relevant
```

Invalid or missing LLM outputs remain **missing** and are never converted to negative relevance labels.

The annotation prompt, few-shot examples, and output schema are implemented in:

```text
src/queries_llm.py
```

Example:

```bash
python src/03-search-results-evaluation.py \
  --input data/candidate_pool \
  --output data/llm_annotations \
  --model gemini-2.5-flash
```

Run the script once for each configured judge. Existing valid annotations are preserved, allowing interrupted annotation runs to be resumed safely.

---

### Stage 4a — Judge aggregation

```text
src/04-judge-aggregation.py
```

This script constructs all three-judge combinations and applies strict 2-out-of-3 majority voting.

If any of the three required judge labels is missing or invalid for a pair, the corresponding aggregate label is left missing rather than inferred from the remaining votes.

Default output:

```text
results/judge_aggregations.csv
```

Summary statistics are written to:

```text
results/human_evaluation/judge_aggregations_summary.csv
```

Example:

```bash
python src/04-judge-aggregation.py
```

With the released annotations, the five judges produce **10 three-judge combinations** over the **11,779 candidate pairs**.

---

### Stage 4 support utility — Human-annotation subset generation

```text
src/generate_human_annotation_subset.py
```

This utility reconstructs the 300-pair subset sent to human annotators.

The subset is a simple random sample without replacement over all query-dataset pairs in the candidate pool. The population is sorted deterministically before sampling, and the sample size and seed are read from:

```text
config/benchmark_config.json
```

The subset is selected directly from the Stage 2 candidate pool and is independent of Stage 3 LLM annotations and Stage 4 judge aggregations. No LLM relevance labels are used for sampling.

The experiment settings are:

```text
human_audit.sample_size = 300
human_audit.sample_seed = 1
```

Default outputs:

```text
data/human_annotations/human_annotation_subset_mapping.csv
data/human_annotations/human_annotation_subset.jsonl
```

Example:

```bash
python src/generate_human_annotation_subset.py
```

With the released candidate pool, this reproduces the distributed subset exactly, including pair order.

---

### Stage 4b — Human evaluation and aggregation assessment

```text
src/05-human-evaluation.py
```

Stage 4 also includes human validation.

The human annotation consists of **300 query-dataset pairs**, independently evaluated by three human annotators.

Inputs are located under:

```text
data/human_annotations/
```

including:

```text
human_annotation_subset_mapping.csv
annotator_1.csv
annotator_2.csv
annotator_3.csv
```

The script computes:

* pairwise Cohen's kappa between human annotators;
* Fleiss' kappa;
* exact human agreement;
* agreement between each LLM judge and the human annotations;
* agreement between three-judge combinations and the human annotations;
* precision, recall, and F1 against the human-majority reference;
* bootstrap confidence intervals;
* reason-level agreement;
* false-negative reason analysis.

Example:

```bash
python src/05-human-evaluation.py
```

Outputs are written to:

```text
results/human_evaluation/
```

including:

```text
human_pairwise_agreement.csv
individual_judges_vs_humans.csv
individual_judges_vs_human_majority.csv
three_judge_combinations_vs_humans.csv
three_judge_combinations_vs_human_majority.csv
reason_analysis.csv
false_negative_reason_analysis.csv
human_evaluation_summary.json
```

The script also writes the pair-level human-majority labels consumed by Stage 5 benchmark assembly:

```text
data/human_annotations/human_majority_labels.csv
```

---

### Stage 5 — Benchmark assembly

```text
src/06-assemble-benchmark.py
```

The final stage combines:

* the Stage 3 annotations;
* the human annotation records;
* the human-majority labels;
* the benchmark configuration.

The default relevance source is read from:

```text
config/benchmark_config.json
```

The model registry is read from:

```text
config/models.json
```

Example:

```bash
python src/06-assemble-benchmark.py
```

The canonical output is:

```text
data/benchmark/benchmark.jsonl
```

Each record contains the query and dataset representation, retrieval provenance, all LLM annotations, the selected default relevance label, and—when available—the complete human-annotation information.

For reproducibility, Stage 6 preserves the Stage 3 `retrieval` and `annotations` objects unchanged.

---

## Canonical benchmark

The distributed benchmark contains:

```text
Queries:                         100
Query-dataset pairs:          11,779
Human-annotated pairs:             300
LLM judges:                         5
```

The canonical benchmark file is:

```text
data/benchmark/benchmark.jsonl
```

---

## Retrieval backend

The artifact includes a minimal Qdrant-based retrieval implementation for reconstructing the Stage 2 candidate pool.

The retrieval profiles are:

| Profile   | Retrieval model                 |
| --------- | ------------------------------- |
| `bm25`    | BM25                            |
| `harrier` | `microsoft/harrier-oss-v1-0.6b` |
| `qwen`    | `Qwen/Qwen3-Embedding-0.6B`     |

All profiles search `title` and `description` separately and combine the two scores.

The command-line entry points are:

```text
scripts/retrieval/build_retrieval_indexes.py
scripts/retrieval/run_retrieval_api.py
```

Detailed setup instructions, including Qdrant requirements and endpoint definitions, are available in [`retrieval_backend/README.md`](retrieval_backend/README.md).

---

## Analysis and paper-result reproduction

Scripts under:

```text
scripts/analysis/
```

are not additional stages of benchmark construction. They reproduce analyses and numerical results reported for the released artifact.

### Candidate-pool diversity

```bash
python scripts/analysis/analyze_pool_diversity.py
```

Computes:

* profile membership counts;
* exclusive candidate contributions;
* exclusive rates;
* pairwise Jaccard overlap.

Default output:

```text
results/pool_diversity.csv
```

---

### LLM annotation statistics

```bash
python scripts/analysis/analyze_annotation_statistics.py
```

Computes:

* positive and negative label counts;
* invalid annotation counts;
* reason distributions;
* supporting-field distributions.

Outputs are written to:

```text
results/annotation_statistics/
```

---

### Retrieval effectiveness

```bash
python scripts/analysis/evaluate_retrieval.py
```

Reconstructs the stored BM25, Harrier, and Qwen rankings from the benchmark retrieval provenance and evaluates them using:

* nDCG@10;
* nDCG@50;
* Recall@10;
* Recall@50.

Default output:

```text
results/retrieval_effectiveness.csv
```

These retrieval systems were also used to construct the candidate pool. Their effectiveness results should therefore be interpreted as measurements over the judged pooled collection rather than as an independent evaluation on a separately constructed test set.

---

## Configuration

### `config/benchmark_config.json`

Contains the main benchmark-construction settings, including configuration for:

* collection preparation;
* queries;
* candidate pooling;
* LLM annotation;
* human annotation;
* judge aggregation;
* evaluation;
* retrieval evaluation;
* the selected default relevance source.

The LLM judges used by the experiment are selected in this file.

### `config/models.json`

Defines the LLM model registry, including model identifiers, execution platforms, and generation settings.

`benchmark_config.json` determines which judges participate in the experiment, while `models.json` provides the execution configuration for those models.

### Environment variables for LLM APIs

Stage 3 requires the corresponding credentials when running API-based LLM judges. Environment variables can be defined in the shell or in a `.env` file at the repository root.

For OpenAI models:

```env
OPENAI_API_KEY=...
```

For Gemini models through Google Vertex AI:

```env
VERTEX_PROJECT_ID=...
VERTEX_LOCATION=us-central1
```

`GOOGLE_CLOUD_PROJECT` can be used instead of `VERTEX_PROJECT_ID`. If `VERTEX_LOCATION` is omitted, `us-central1` is used by default.

Vertex AI authentication uses Google Application Default Credentials (ADC). For example:

```bash
gcloud auth application-default login
```

Local Ollama models do not require API credentials, but the corresponding models must be available in the local Ollama runtime.

---

## Data directories

### `data/normalized_collection/`

Processed collection used by retrieval.

The repository distributes the processed snapshot rather than the complete raw `datos.gob.es` crawl.

The processed collection can also be downloaded from Figshare:

```text
https://doi.org/10.6084/m9.figshare.33952177
```

### `data/queries/`

Benchmark query set.

### `data/candidate_pool/`

Released Stage 2 candidate pool.

Each query file contains the union of candidates retrieved by the configured profiles together with profile-specific ranks and scores.

### `data/llm_annotations/`

Stage 3 candidate-pool records enriched with the five LLM annotations.

### `data/human_annotations/`

Human-annotation sample, individual human judgments, and majority labels.

### `data/benchmark/`

Final assembled benchmark.

---

## Results directories

The `results/` directory contains reproducible analysis outputs, including:

```text
judge_aggregations.csv
human_evaluation/judge_aggregations_summary.csv
human_evaluation/
annotation_statistics/
pool_diversity.csv
retrieval_effectiveness.csv
```

These files can be regenerated from the distributed artifact inputs using the corresponding scripts.

---

## Reproducing the released artifact

Two reproduction paths are supported:

1. **Full pipeline reproduction**, starting from the processed collection and rebuilding retrieval, LLM annotations, aggregations, human-evaluation outputs, and the final benchmark.
2. **Downstream reproduction from the distributed artifact**, using the released intermediate data to reproduce aggregation, human evaluation, benchmark assembly, and analysis results without re-running retrieval or external LLM services.

### Full pipeline reproduction

A full reconstruction follows. Human-annotation sampling branches directly from the Stage 2 candidate pool:

```text
data/normalized_collection/
        │
        │   build_retrieval_indexes.py
        │   run_retrieval_api.py
        ↓
local retrieval backend
        │
        ↓
02-search-results-generation.py
        │
        ↓
data/candidate_pool/
        │
        ├──────────────────────────────────────┐
        │                                      │
        ↓                                      ↓
03-search-results-evaluation.py    generate_human_annotation_subset.py
        │                                      │
        ↓                                      ↓
data/llm_annotations/          human_annotation_subset.*
        │                                      │
        │                              [human annotation]
        │                                      │
        │                              annotator_1/2/3.csv
        │                                      │
        ├──→ 04-judge-aggregation.py           │
        │          │                           │
        │          ↓                           │
        │   judge_aggregations.csv             │
        │          │                           │
        │          └─────────────┐             │
        │                        ↓             ↓
        │                 05-human-evaluation.py
        │                        │
        │                        ├──→ results/human_evaluation/
        │                        │
        │                        └──→ human_majority_labels.csv
        │
        ├──────────────────────────────┐
        │                              │
        ↓                              ↓
data/llm_annotations/       human annotations
        │                   + human_majority_labels.csv
        └──────────────┬───────────────┘
                       ↓
              06-assemble-benchmark.py
                       ↓
           data/benchmark/benchmark.jsonl
```

Re-running Stage 3 requires access to the LLM execution environments configured through `config/benchmark_config.json` and `config/models.json`, together with the corresponding local runtimes or API credentials.

### Downstream reproduction from distributed artifacts

The distributed candidate pool, LLM annotations, and human annotations allow all stages after LLM annotation to be reproduced without contacting external LLM services.

The main downstream pipeline can be reproduced with:

```bash
python src/04-judge-aggregation.py
python src/05-human-evaluation.py
python src/06-assemble-benchmark.py
```

The reported analysis outputs can then be regenerated with:

```bash
python scripts/analysis/analyze_pool_diversity.py
python scripts/analysis/analyze_annotation_statistics.py
python scripts/analysis/evaluate_retrieval.py
```

---

## Main validation results

The released artifact reproduces the following benchmark-construction statistics:

```text
Candidate pairs:        11,779
Human-annotated pairs:     300

Human majority:
  relevant:                 173
  non-relevant:             127
```

The selected default relevance source is:

```text
gemini-2.5-flash
```

with:

```text
relevant pairs:          6,359
non-relevant pairs:      5,420
missing labels:              0
```

These values are reported as reproduction references; they are not hard-coded as execution requirements in the benchmark-assembly pipeline.

---

## Reproducibility notes

* The artifact uses a fixed processed snapshot of `datos.gob.es`.
* Retrieval is executed locally through Qdrant.
* LLM annotation outputs are preserved as generated; invalid outputs remain missing rather than being converted to negative labels.
* Human annotation is performed on a fixed 300-pair sample.
* Bootstrap confidence intervals use query-dataset pairs as the resampling unit.
* The distributed intermediate data allow downstream benchmark construction and analysis to be reproduced without re-running the LLM annotation stage.

For retrieval-backend-specific setup and version requirements, see [`retrieval_backend/README.md`](retrieval_backend/README.md).
