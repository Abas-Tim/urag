"""Tests for the eval harness (autogen, gold resolution, metrics, chunk mapping)."""

from pathlib import Path
import tempfile

import pytest

from urag.config import load_config
from urag.db import Database
from urag.eval import (
    Hit,
    Question,
    SystemRun,
    aggregate,
    autogen_questions,
    _metrics,
    _unit_tokens,
    resolve_question,
)
from urag.extractors.python_ext import PythonExtractor
from urag.models import SourceFile


@pytest.fixture
def db(tmp_path: Path):
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    src = '''"""Mod."""

def alpha(x: int) -> int:
    """Alpha helper."""
    return x + 1

class Beta:
    def go(self):
        return alpha(1)
'''
    units = PythonExtractor().extract(src, "m.py")
    f = SourceFile(path="m.py", kind="source", language="python", size=len(src), mtime=1)
    fid = db.upsert_file(f)
    db.replace_units(fid, units)
    edges = []
    for c in PythonExtractor().collect_calls(src):
        for u in units:
            if u.id is not None and u.byte_start <= c.byte_start <= u.byte_end and u.unit_type in ("function", "method"):
                edges.append((u.id, c.callee, c.callee_full, c.line))
    db.replace_call_edges(fid, edges)
    yield db
    db.close()


def test_autogen_and_resolve(db):
    qs = autogen_questions(db, 2)
    assert qs, "expected autogen questions"
    for q in qs:
        rq = resolve_question(db, q)
        assert rq.gold_unit_ids, q
        if rq.label == "definition":
            assert rq.gold_file == "m.py"


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
        {"unit_recall": 1.0, "file_recall": 1.0, "mrr": 1.0, "tokens": 10, "seconds": 0.1, "n_hits": 1},
        {"unit_recall": 0.0, "file_recall": 1.0, "mrr": 0.0, "tokens": 30, "seconds": 0.3, "n_hits": 1},
    ]
    a = aggregate(rows)
    assert a["unit_recall"] == pytest.approx(0.5)
    assert a["mean_tokens"] == pytest.approx(20)
    assert a["p95_sec"] == pytest.approx(0.3)


def test_chunk_unit_at(db):
    from urag.eval import ChunkBaseline

    methods = ChunkBaseline._unit_at
    u, _, _ = db.unit_by_id(1)
    assert methods(db, "m.py", u.byte_start) is not None
