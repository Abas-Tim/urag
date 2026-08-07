from pathlib import Path

from urag.classify import classify
from urag.config import load_config
from urag.db import Database
from urag.embed import Embedder, NoopEmbedder
from urag.indexer import Indexer
from urag.retrieve import Retriever, fit_evidence


class _StaticEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 2

    def embed_passages(self, texts):
        return [[1.0, 0.0] if "auth.py" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "auth" in text else [0.0, 1.0]


def test_dense_and_hybrid_search_use_indexed_vectors(tmp_path: Path):
    (tmp_path / "auth.py").write_text("def validate():\n    return True\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def unrelated():\n    return True\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.embedding.dimension = 2
    embedder = _StaticEmbedder()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, embedder).index_all()
    try:
        dense = Retriever(cfg, db, embedder).search("auth", mode="dense")
        hybrid = Retriever(cfg, db, embedder).search("auth", mode="hybrid")
        assert dense.results[0].file_path == "auth.py"
        assert hybrid.results[0].file_path == "auth.py"
    finally:
        db.close()


def test_non_git_changes_are_marked_stale(tmp_path: Path):
    path = tmp_path / "module.py"
    path.write_text("def value():\n    return 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        path.write_text("def value():\n    return 2\n", encoding="utf-8")
        result = Retriever(cfg, db, NoopEmbedder()).search("value", mode="lexical")
        assert result.results[0].stale is True
    finally:
        db.close()


def test_snake_case_queries_are_symbol_queries():
    assert classify("parse_token") == "symbol"


def test_evidence_fits_character_budget():
    span = "\n".join("x" * 40 for _ in range(20))
    fitted = fit_evidence(span, 20)
    assert len(fitted.splitlines()[0]) <= 40
    assert "full span via urag get" in fitted
