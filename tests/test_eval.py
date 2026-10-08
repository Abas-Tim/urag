"""Tests for the eval harness (autogen, gold resolution, metrics, chunk mapping)."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from urag.config import load_config
from urag.db import Database
from urag.embed import Embedder, NoopEmbedder
from urag.eval import (
    ChunkBaseline,
    Hit,
    OracleBaseline,
    Question,
    RgBaseline,
    SystemRun,
    _metrics,
    aggregate,
    autogen_questions,
    judge_results,
    load_questions,
    reresolve_questions,
    resolve_question,
    run_eval,
    scan_import_aliases,
)
from urag.indexer import Indexer


@pytest.fixture
def db(tmp_path: Path):
    cfg = load_config(tmp_path)
    src = '''"""Mod."""

def alpha(x: int) -> int:
    """Alpha helper."""
    return x + 1

class Beta:
    def go(self):
        return alpha(1)
'''
    (tmp_path / "m.py").write_text(src, encoding="utf-8")
    db = Database(cfg.db_path, cfg.embedding.dimension)
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        yield db
    finally:
        db.close()


def test_autogen_and_resolve(db):
    qs = autogen_questions(db, 2)
    assert {q.label for q in qs} == {"definition", "call"}
    for q in qs:
        rq = resolve_question(db, q)
        assert rq.gold_unit_ids, q
        if rq.label == "definition":
            assert rq.gold_file == "m.py"
        else:
            assert rq.target == "alpha"
            assert rq.query == "what calls alpha"


def test_eval_errors_when_rg_missing(tmp_path, monkeypatch):
    """A missing ripgrep must fail loudly, not score the rg baseline 0.00."""
    monkeypatch.setattr("urag.eval.shutil.which", lambda name: None)
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    (tmp_path / "m.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    db = Database(cfg.db_path, cfg.embedding.dimension)
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        with pytest.raises(RuntimeError, match="ripgrep"):
            run_eval(cfg, db, NoopEmbedder(), autogen=1, systems="urag-auto,rg")
    finally:
        db.close()


def test_autogen_skips_ambiguous_definitions(db):
    root = db.db_path.parent.parent
    (root / "other.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
    Indexer(load_config(root), db, NoopEmbedder()).index_all()

    definitions = [q.query for q in autogen_questions(db, 20) if q.label == "definition"]

    assert "where is alpha defined" not in definitions


def test_reresolve_drops_ambiguous_definition(db):
    root = db.db_path.parent.parent
    (root / "other.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
    Indexer(load_config(root), db, NoopEmbedder()).index_all()

    question = Question(query="where is alpha defined", label="definition", gold_file="m.py")

    assert reresolve_questions(db, [question]) == []


def test_callers_gold(db):
    # what calls alpha -> caller = Beta.go
    qs = [q for q in autogen_questions(db, 10) if q.label == "call" and "alpha" in q.query]
    assert qs
    resolved = resolve_question(db, qs[0])
    u, _, _ = db.unit_by_id(resolved.gold_unit_ids[0])
    assert u.qualname == "Beta.go"


def test_metrics():
    q = Question(query="q", gold_unit_ids=[7], gold_file="m.py")
    run = SystemRun(
        "t",
        hits=[Hit("other.py", 1, 10), Hit("m.py", 7, 20)],
        seconds=0.01,
        tokens=30,
    )
    m = _metrics(run, q, top_k=2)
    assert m["unit_recall"] == 1.0
    assert m["file_recall"] == 1.0
    assert m["mrr"] == 0.5
    assert m["tokens"] == 30


def test_metrics_miss():
    q = Question(query="q", gold_unit_ids=[99], gold_file="m.py")
    run = SystemRun("t", hits=[Hit("other.py", 1, 10)], seconds=0.0, tokens=10)
    m = _metrics(run, q, top_k=5)
    assert m["unit_recall"] == 0.0
    assert m["mrr"] == 0.0


def test_aggregate():
    rows = [
        {
            "unit_recall": 1.0,
            "file_recall": 1.0,
            "mrr": 1.0,
            "tokens": 10,
            "seconds": 0.1,
            "n_hits": 1,
        },
        {
            "unit_recall": 0.0,
            "file_recall": 1.0,
            "mrr": 0.0,
            "tokens": 30,
            "seconds": 0.3,
            "n_hits": 1,
        },
    ]
    a = aggregate(rows)
    assert a["unit_recall"] == pytest.approx(0.5)
    assert a["mean_tokens"] == pytest.approx(20)
    assert a["p50_sec"] == pytest.approx(0.1)
    assert a["p95_sec"] == pytest.approx(0.3)


def test_rg_path_normalization():
    assert RgBaseline._normalize_path(r".\src\auth.py") == "src/auth.py"


def test_rg_decodes_utf8_and_replaces_invalid_bytes(db, tmp_path, monkeypatch):
    run = subprocess.run

    def fake_rg(_args, **kwargs):
        return run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'./m.py:3:alpha \\xe2\\x80\\x94 \\x81\\n')",
            ],
            **kwargs,
        )

    monkeypatch.setattr("urag.eval.subprocess.run", fake_rg)
    result = RgBaseline(tmp_path).search("alpha", 5, db)

    assert result.hits
    assert result.hits[0].file == "m.py"
    assert result.hits[0].detail == "3: alpha — �"


class CountingEmbedder(Embedder):
    @property
    def dimension(self):
        return 2

    def __init__(self):
        self.texts = []
        self.fail_after: int | None = None

    def embed_passages(self, texts):
        if self.fail_after is not None and len(self.texts) >= self.fail_after:
            raise RuntimeError("interrupted")
        self.texts.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_chunk_cache_reuses_vectors_and_invalidates_changed_inputs(db, tmp_path):
    cfg = load_config(tmp_path)
    embedder = CountingEmbedder()
    first = ChunkBaseline(cfg, db, embedder)
    assert len(embedder.texts) == len(first.chunks)
    embedder.texts.clear()

    cached = ChunkBaseline(cfg, db, embedder)
    assert embedder.texts == []
    assert cached.vectors == first.vectors
    assert cached.cached_chunks == len(first.chunks)
    assert cached.search("alpha", 1, db).hits == first.search("alpha", 1, db).hits

    (tmp_path / "extra.py").write_bytes(b"def extra(): return 2\n")
    ChunkBaseline(cfg, db, embedder)
    assert embedder.texts == ["def extra(): return 2\n"]
    embedder.texts.clear()

    (tmp_path / "extra.py").write_bytes(b"def extra(): return 3\n")
    ChunkBaseline(cfg, db, embedder)
    assert embedder.texts == ["def extra(): return 3\n"]
    embedder.texts.clear()

    cfg.embedding.http_model = "different-model"
    changed_model = ChunkBaseline(cfg, db, embedder)
    assert changed_model.cached_chunks == 0
    assert len(embedder.texts) == len(changed_model.chunks)


def test_chunk_cache_repairs_invalid_vectors(db, tmp_path):
    cfg = load_config(tmp_path)
    embedder = CountingEmbedder()
    first = ChunkBaseline(cfg, db, embedder)
    with sqlite3.connect(cfg.urag_dir / "eval-chunks.sqlite3") as cache:
        cache.execute("UPDATE embeddings SET vector = ?", ('["invalid"]',))
    embedder.texts.clear()

    repaired = ChunkBaseline(cfg, db, embedder)
    assert repaired.cached_chunks == 0
    assert repaired.vectors == first.vectors
    assert embedder.texts


def test_eval_chunk_progress_keeps_json_stdout_clean(db, tmp_path, capsys):
    cfg = load_config(tmp_path)
    run_eval(cfg, db, CountingEmbedder(), systems="chunk", autogen=1, json_out=True)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["chunk_cached_chunks"] == 0
    assert "chunk baseline:" in captured.err


def test_chunk_cache_resumes_completed_batches(db, tmp_path, monkeypatch):
    cfg = load_config(tmp_path)
    (tmp_path / "extra.py").write_text("def extra(): return 2\n", encoding="utf-8")
    monkeypatch.setattr(ChunkBaseline, "BATCH_SIZE", 1)
    embedder = CountingEmbedder()
    embedder.fail_after = 1

    with pytest.raises(RuntimeError, match="interrupted"):
        ChunkBaseline(cfg, db, embedder)

    completed = list(embedder.texts)
    embedder.texts.clear()
    embedder.fail_after = None
    resumed = ChunkBaseline(cfg, db, embedder)
    assert resumed.cached_chunks == 1
    assert len(embedder.texts) == 1
    assert completed[0] not in embedder.texts


def test_eval_alias_scan_uses_imported_symbol():
    assert scan_import_aliases("from core.http import fetch as http_fetch", "python") == {
        "http_fetch": "core.http.fetch"
    }


def test_oracle_rejects_path_traversal(db, tmp_path):
    outside = tmp_path.parent / "eval-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    question = Question(query="q", gold_file="../eval-secret.txt")

    result = OracleBaseline(tmp_path).search(question, db)

    assert result.hits == []


def test_load_questions_reports_line_number(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"query": "valid"}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"questions\.jsonl:2:"):
        load_questions(path)


def test_load_questions_rejects_invalid_label(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"query": "q", "label": null}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="label must be a non-empty string"):
        load_questions(path)


def test_judge_failure_isolated(db, monkeypatch):
    from urag import eval as eval_module

    monkeypatch.setattr(
        eval_module,
        "_llm_chat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    logs = []
    cfg = load_config(db.db_path.parent.parent)
    run = SystemRun("test", [Hit("m.py", None, 1)], 0.0, 1)

    scores = judge_results(
        [Question(query="q")],
        {"test": {0: run}},
        db,
        cfg,
        "http://judge",
        "model",
        "key",
        progress=logs.append,
    )

    assert scores == {}
    assert logs == ["  [test] q0: judge request failed"]


def test_chunk_unit_at(db):
    from urag.eval import ChunkBaseline

    methods = ChunkBaseline._unit_at
    u, _, _ = db.unit_by_id(1)
    assert methods(db, "m.py", u.byte_start) == u.id
