# Indexing Performance

## What Changed

The initial index has two phases:

1. Extract source units, build lexical search, and resolve graph edges.
2. Load the embedding model and fill missing dense vectors.

Normal `urag init --full` and `urag index` complete both phases. Use
`--defer-embeddings` to stop after the first phase, then
`urag index --embeddings-only` to complete the second without scanning source.
Completed vector batches are committed, so interrupted work can be resumed.

CLI and MCP retrieval initialize the model only when an embedding is needed.
Lexical queries, graph queries, and hybrid queries on an index with no vectors
remain usable without downloading the model.

Local passage inputs are grouped by character length before batching to reduce
token padding. The default local batch size is 8, rather than 64. Vector order
is restored before returning provider results, and indexing preserves unit IDs.
HTTP providers retain a default batch size of 64.

## Measurements

Measured on Windows with an AMD Ryzen 9 5900X (12 cores / 24 logical processors),
FastEmbed 0.8.0, ONNX Runtime 1.28.0, and the cached default
`BAAI/bge-base-en-v1.5` model. The measurements exclude model download time.

A 256-passage sample from this repository's index ranged from 57 to 915
characters. ONNX inference accounted for over 99% of measured embedding time.
Two repetitions per batch configuration gave:

| Passage ordering | Batch size | Median seconds | Passages/second |
| --- | ---: | ---: | ---: |
| Original index order | 8 | 8.425 | 30.39 |
| Original index order | 16 | 10.795 | 23.72 |
| Original index order | 32 | 13.755 | 18.61 |
| Original index order | 64 | 18.420 | 13.90 |
| Length grouped | 8 | 5.897 | 43.41 |
| Length grouped | 16 | 6.394 | 40.04 |
| Length grouped | 32 | 7.858 | 32.58 |
| Length grouped | 64 | 10.419 | 24.57 |

The best measured configuration was approximately 3.1 times faster than the
old 64-item, original-order passage workload. A separate 64-passage thread
sweep found automatic ONNX threading faster than forcing 1, 4, or 8 threads,
so automatic threading remains the default.

A fresh-index fixture creates Python functions with documentation sampled from
the same indexed passages. It exercises extraction, graph resolution, actual
Indexer batching, vector validation, and SQLite storage:

| Fixture size | Lexical/graph phase | Cached model load | Old dense phase | Tuned dense phase | Dense speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 units | 0.380s | 0.782s | 10.514s | 7.273s | 1.45x |
| 1,024 units | 0.506s | 0.378s | 43.840s | 29.645s | 1.48x |

For these fixtures, old and tuned vectors matched by unit ID with a minimum
cosine similarity of 1.0. The 1,024-unit fixture reduced dense-phase time by
about 32%. Deferred mode makes the lexical/graph phase usable without waiting
for either model loading or dense inference.

These are local measurements, not a promised speedup for every repository.
Model, input lengths, hardware, and provider affect throughput. Large projects
still incur inference cost proportional to their embedding-eligible units.

## Reproduce And Tune

From a source checkout with an existing index and cached local model:

```powershell
.\.venv\Scripts\python.exe scripts\profile_embeddings.py --samples 256 --threads 0 --batches 8,16,32,64 --repeats 2
.\.venv\Scripts\python.exe scripts\profile_embeddings.py --samples 256 --threads 0 --batches 8,16,32,64 --repeats 2 --sort
.\.venv\Scripts\python.exe scripts\profile_embeddings.py --samples 64 --threads 0,1,4,8 --batches 16,64 --repeats 1
.\.venv\Scripts\python.exe scripts\profile_embeddings.py --samples 1024 --index-benchmark
```

The passage profiler opens the project database read-only and uses cached-only
model loading. The index benchmark builds its source and database in a temporary
directory and compares the old pipeline with the current configured execution
settings. Neither modifies the project index. Run measurements without competing
CPU-heavy work.

Configure execution separately from model identity:

```powershell
urag embed --batch-size 16 --threads 8
urag embed --batch-size 0 --threads 0
```

`batch_size = 0` selects the provider default, and `threads = 0` selects
automatic ONNX threading. Changing these settings preserves existing vectors
and evaluation caches. FastEmbed's batch and thread controls are described in
its [performance reference](https://deepwiki.com/qdrant/fastembed/8-performance-optimization).
