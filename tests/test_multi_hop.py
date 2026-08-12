"""Tests for multi-hop traversal (callers-of-callers)."""

from pathlib import Path

import pytest

from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.indexer import Indexer
from urag.retrieve import Retriever

SRC = """def a():
    b()


def b():
    c()


def c():
    b()


def d():
    c()


def e():
    pass
"""


@pytest.fixture
def db(tmp_path: Path):
    cfg = load_config(tmp_path)
    (tmp_path / "m.py").write_text(SRC, encoding="utf-8")
    db = Database(cfg.db_path, cfg.embedding.dimension)
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        yield db
    finally:
        db.close()


def _id(db, name):
    row = db.conn.execute(
        "SELECT u.id FROM units u WHERE u.name = ? ORDER BY u.start_line LIMIT 1",
        (name,),
    ).fetchone()
    assert row, name
    return row["id"]


def test_direct_callers_unchanged(db):
    hits = db.callers("c", limit=10)
    assert {h["unit"].name for h in hits} == {"b", "d"}


def test_transitive_two_hops(db):
    hits = db.transitive_callers("c", max_depth=2)
    by_name = {h["unit"].name: h["hop"] for h in hits}
    # c -> b -> c means c is its own 2-hop caller; a -> b -> c puts a at hop 2
    assert by_name == {"b": 1, "d": 1, "a": 2, "c": 2}


def test_transitive_depth_one(db):
    hits = db.transitive_callers("c", max_depth=1)
    assert {h["unit"].name for h in hits} == {"b", "d"}
    assert all(h["hop"] == 1 for h in hits)


def test_cycle_terminates(db):
    hits = db.transitive_callers("c", max_depth=5)
    by_name = {h["unit"].name: h["hop"] for h in hits}
    assert by_name["b"] == 1
    assert by_name["d"] == 1
    assert by_name["a"] == 2
    assert by_name["c"] == 2
    assert len(hits) == 4  # b, d, a, c(self) — no infinite loop


def test_no_callers(db):
    assert db.transitive_callers("e") == []
    assert db.transitive_callers("") == []
    assert db.transitive_callers("missing_symbol") == []


def test_limit(db):
    hits = db.transitive_callers("c", max_depth=3, limit=2)
    assert len(hits) == 2


def test_search_transitive(db, tmp_path):
    cfg = load_config(tmp_path)
    retriever = Retriever(cfg, db, NoopEmbedder())
    result = retriever.search_transitive("c", depth=2)
    by_name = {r.unit.name: r.hop for r in result.results}
    assert by_name == {"b": 1, "d": 1, "a": 2, "c": 2}
    assert result.mode == "calls"
    d = result.to_dict()
    hops = {r["name"]: r["hop"] for r in d["results"]}
    assert hops["a"] == 2
    direct = retriever.search_callers("c")
    assert all(r.hop == 0 for r in direct.results)


def test_search_transitive_without_embedding(tmp_path, db):
    cfg = load_config(tmp_path)
    retriever = Retriever(cfg, db, NoopEmbedder())
    result = retriever.search_transitive("c", depth=2, limit=2)
    assert len(result.results) == 2
