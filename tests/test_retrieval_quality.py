from pathlib import Path

from urag.classify import classify
from urag.config import load_config
from urag.db import Database
from urag.embed import Embedder, NoopEmbedder
from urag.indexer import Indexer
from urag.models import Unit
from urag.retrieve import Retriever, fit_evidence


class _StaticEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 2

    def embed_passages(self, texts):
        return [[1.0, 0.0] if "auth.py" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "auth" in text else [0.0, 1.0]


class _RankingDb:
    def __init__(self, lexical, dense):
        self.lexical = lexical
        self.dense = dense

    def lexical_search(self, *args, **kwargs):
        return self.lexical

    def dense_search(self, *args, **kwargs):
        return self.dense

    def resolve_units(self, name, limit=30, language=None):
        return [
            (unit, path, "")
            for unit, path, _score in self.lexical
            if unit.name == name or unit.qualname == name
        ][:limit]


def _unit(unit_id: int, kind: str, name: str) -> Unit:
    return Unit(
        file_id=unit_id,
        kind=kind,
        unit_type="function" if kind == "symbol" else "doc_chunk",
        name=name,
        qualname=name,
        id=unit_id,
    )


def test_hybrid_promotes_exact_symbol_matches_over_dense_noise(tmp_path: Path):
    doc = _unit(1, "chunk", "notes")
    target = _unit(2, "symbol", "parse")
    noise = _unit(3, "symbol", "unrelated")
    db = _RankingDb(
        [(doc, "README.md", 1.0), (target, "parser.py", 2.0)],
        [(noise, "other.py", 0.1)],
    )
    cfg = load_config(tmp_path)
    retriever = Retriever(cfg, db, _StaticEmbedder())
    retriever._stale_map = lambda paths: {path: False for path in paths}
    retriever._enrich = lambda results, stale=None: None

    result = retriever.search("where is parse defined", mode="hybrid", top_k=1)
    conceptual = retriever.search("how does parse work", mode="hybrid", top_k=1)

    assert result.results[0].unit.name == "parse"
    assert conceptual.results[0].unit.name == "notes"


def test_definition_query_returns_only_exact_symbols(tmp_path: Path):
    (tmp_path / "module.py").write_text(
        "def parse_token():\n    return True\n\ndef unrelated():\n    return False\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        result = Retriever(cfg, db, NoopEmbedder()).search(
            "where is parse_token defined", mode="hybrid", top_k=5
        )
        assert result.mode == "definitions"
        assert [item.unit.name for item in result.results] == ["parse_token"]
    finally:
        db.close()


def test_resolve_units_prioritizes_qualified_exact_match(tmp_path: Path):
    (tmp_path / "module.py").write_text(
        "class pkg:\n"
        "    class Service:\n"
        "        pass\n\n"
        "class other:\n"
        "    class pkg:\n"
        "        class Service:\n"
        "            pass\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        results = db.resolve_units("pkg.Service", limit=1, language="python")
        assert results[0][0].qualname == "pkg.Service"
    finally:
        db.close()


def test_dense_and_hybrid_search_use_indexed_vectors(tmp_path: Path):
    (tmp_path / "auth.py").write_text(
        "def validate():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "other.py").write_text(
        "def unrelated():\n    return True\n", encoding="utf-8"
    )
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


def test_dense_search_hydrates_results_without_per_row_lookup(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "auth.py").write_text(
        "def validate():\n    return True\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path)
    cfg.embedding.dimension = 2
    embedder = _StaticEmbedder()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, embedder).index_all()
    try:
        monkeypatch.setattr(
            db, "unit_by_id", lambda _unit_id: (_ for _ in ()).throw(AssertionError)
        )
        result = db.dense_search([1.0, 0.0], limit=1)
        assert result[0][1] == "auth.py"
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
