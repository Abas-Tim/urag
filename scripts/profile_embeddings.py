"""Measure cached local embedding throughput without modifying the project index."""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import json
import math
import os
import pstats
import sqlite3
import statistics
import struct
import tempfile
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from urag.config import IndexConfig, default_model_cache_dir, load_config
from urag.db import Database
from urag.embed import Embedder, LocalEmbedder, NoopEmbedder
from urag.indexer import Indexer


def benchmark_index(cfg, texts: list[str]) -> None:
    class BaselineEmbedder(Embedder):
        @property
        def dimension(self):
            return tuned.dimension

        def embed_passages(self, texts):
            return [[float(x) for x in v] for v in tuned.model.embed(texts)]

        def embed_query(self, text):
            return tuned.embed_query(text)

    with tempfile.TemporaryDirectory(prefix="urag-profile-") as directory:
        root = Path(directory)
        for start in range(0, len(texts), 32):
            source = "\n\n".join(
                f"def sample_{i}(value):\n    {text!r}\n    return value\n"
                for i, text in enumerate(texts[start : start + 32], start)
            )
            (root / f"sample_{start}.py").write_text(source, encoding="utf-8")
        fixture_cfg = replace(cfg, project_root=root, index=IndexConfig(languages=["python"]))
        started = time.perf_counter()
        with closing(Database(fixture_cfg.db_path, cfg.embedding.dimension)) as db:
            indexer = Indexer(fixture_cfg, db, NoopEmbedder())
            indexer.index_all()
            lexical_seconds = time.perf_counter() - started
            started = time.perf_counter()
            tuned = LocalEmbedder(cfg.embedding)
            model_load_seconds = time.perf_counter() - started
            indexer.embedder = BaselineEmbedder()
            started = time.perf_counter()
            count = indexer.embed_pending()
            baseline_seconds = time.perf_counter() - started
            baseline = dict(db.conn.execute("SELECT unit_id, embedding FROM vec_units"))
            db.clear_embeddings()
            indexer.embedder = tuned
            started = time.perf_counter()
            indexer.embed_pending()
            tuned_seconds = time.perf_counter() - started
            cosines = []
            for uid, raw in db.conn.execute("SELECT unit_id, embedding FROM vec_units"):
                before = struct.unpack(f"{tuned.dimension}f", baseline[uid])
                after = struct.unpack(f"{tuned.dimension}f", raw)
                dot = sum(a * b for a, b in zip(before, after, strict=True))
                norm = math.sqrt(sum(a * a for a in before) * sum(b * b for b in after))
                cosines.append(dot / norm)
            print(
                json.dumps(
                    {
                        "benchmark": "fresh index fixture",
                        "units": count,
                        "lexical_seconds": round(lexical_seconds, 3),
                        "model_load_seconds": round(model_load_seconds, 3),
                        "baseline_embedding_seconds": round(baseline_seconds, 3),
                        "tuned_embedding_seconds": round(tuned_seconds, 3),
                        "embedding_speedup": round(baseline_seconds / tuned_seconds, 2),
                        "minimum_vector_cosine": min(cosines),
                    }
                ),
                flush=True,
            )
            if min(cosines) < 0.99:
                raise RuntimeError("embedding equivalence check failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", default="0,1,4,8")
    parser.add_argument("--batches", default="16,32,64")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--index-benchmark", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        parser.error("--samples and --repeats must be positive")
    from fastembed import TextEmbedding

    cfg = load_config(args.root.resolve())
    conn = sqlite3.connect(cfg.db_path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT u.*, f.path, f.language FROM units u JOIN files f ON f.id = u.file_id "
            "WHERE u.summary != '' OR u.concepts != '' OR u.qualname != '' OR u.signature != '' "
            "ORDER BY u.id"
        ).fetchall()
        stride = max(1, len(rows) // args.samples)
        rows = rows[::stride][: args.samples]
        texts = [
            f"file: {row['path']}\nlanguage: {row['language']}\n"
            + Database._row_to_unit(row).retrieval_key
            for row in rows
        ]
    finally:
        conn.close()
    if not texts:
        raise SystemExit("No embedding-eligible units found in the index")
    if args.index_benchmark:
        benchmark_index(cfg, texts)
        return
    if args.sort:
        texts.sort(key=len)
    print(
        json.dumps(
            {
                "model": cfg.embedding.model,
                "cpu_count": os.cpu_count(),
                "fastembed": importlib.metadata.version("fastembed"),
                "onnxruntime": importlib.metadata.version("onnxruntime"),
                "samples": len(texts),
                "chars_min": min(map(len, texts)),
                "chars_max": max(map(len, texts)),
                "sorted": args.sort,
            }
        ),
        flush=True,
    )
    for threads in map(int, args.threads.split(",")):
        started = time.perf_counter()
        model = TextEmbedding(
            model_name=cfg.embedding.model,
            cache_dir=str(default_model_cache_dir()),
            threads=threads or None,
            local_files_only=True,
        )
        load_seconds = time.perf_counter() - started
        list(model.embed(texts[:2], batch_size=2))
        for batch_size in map(int, args.batches.split(",")):
            durations = []
            profiler = cProfile.Profile()
            for _ in range(args.repeats):
                started = time.perf_counter()
                profiler.enable()
                vectors = list(model.embed(texts, batch_size=batch_size))
                profiler.disable()
                durations.append(time.perf_counter() - started)
                if len(vectors) != len(texts):
                    raise RuntimeError("embedding count mismatch")
            stats = pstats.Stats(profiler)
            inference_seconds = (
                sum(
                    value[2]
                    for key, value in stats.stats.items()
                    if key[2] == "run" and "onnxruntime" in key[0]
                )
                / args.repeats
            )
            median = statistics.median(durations)
            print(
                json.dumps(
                    {
                        "threads": threads or "default",
                        "batch_size": batch_size,
                        "load_seconds": round(load_seconds, 3),
                        "seconds": [round(s, 3) for s in durations],
                        "passages_per_second": round(len(texts) / median, 2),
                        "inference_seconds": round(inference_seconds, 3),
                    }
                ),
                flush=True,
            )
        del model


if __name__ == "__main__":
    main()
