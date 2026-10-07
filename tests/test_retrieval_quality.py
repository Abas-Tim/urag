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


def test_dense_search_hydrates_results_without_per_row_lookup(tmp_path: Path, monkeypatch):
    (tmp_path / "auth.py").write_text("def validate():\n    return True\n", encoding="utf-8")
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
        assert result.results[0].stale_basis == "sha256"
    finally:
        db.close()


def test_snake_case_queries_are_symbol_queries():
    assert classify("parse_token") == "symbol"


def test_evidence_fits_character_budget():
    span = "\n".join("x" * 40 for _ in range(20))
    fitted = fit_evidence(span, 20)
    assert len(fitted.splitlines()[0]) <= 40
    assert "full span via urag get" in fitted


def test_definition_common_words_are_not_exact_identifiers():
    from urag.retrieve import _exact_symbol_ids

    assert _exact_symbol_ids("where is the user count defined", []) == set()
    assert _exact_symbol_ids("where is value defined", []) == set()


def test_multiword_queries_are_not_misclassified_as_symbols():
    assert classify("P7 immediate handoff next implementation task DiffmapPsychoImage") == "local"
    assert classify("GPU buffer staging") == "local"
    assert classify("run ParseToken") == "local"
    assert classify("ParseToken") == "symbol"
    assert classify("HTTP") == "symbol"
    assert classify("TokenValidator.validate") == "symbol"


def test_multiword_query_with_leading_short_token_finds_doc_heading(tmp_path: Path):
    (tmp_path / "plan.md").write_text(
        "# Plan\n\n## Immediate handoff\n\nnext implementation task for DiffmapPsychoImage\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        result = Retriever(cfg, db, NoopEmbedder()).search(
            "P7 immediate handoff next implementation task DiffmapPsychoImage",
            mode="hybrid",
            top_k=5,
        )
        assert result.results
        assert any("Immediate handoff" in item.unit.name for item in result.results)
    finally:
        db.close()


def test_exact_symbol_miss_falls_back_to_broader_lexical(tmp_path: Path):
    (tmp_path / "notes.md").write_text(
        "# Notes\n\n## Handoff\n\nP7 status notes.\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        result = Retriever(cfg, db, NoopEmbedder()).search("P7", mode="hybrid", top_k=5)
        assert result.results
        assert "exact symbol" in result.fallback
    finally:
        db.close()


def test_hybrid_reports_lexical_only_when_dense_unavailable(tmp_path: Path):
    (tmp_path / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        result = Retriever(cfg, db, NoopEmbedder()).search("value", mode="hybrid", top_k=3)
        payload = result.to_dict()
        assert payload["mode"] == "lexical"
        assert payload["mode_requested"] == "hybrid"
        assert payload["dense_ready"] is False
        assert "dense unavailable" in payload["fallback"]
    finally:
        db.close()


def test_search_reports_indexing_commit_and_freshness_basis(tmp_path: Path):
    (tmp_path / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        result = Retriever(cfg, db, NoopEmbedder()).search("value", mode="lexical")
        payload = result.to_dict()
        assert payload["head"] == ""
        entry = payload["results"][0]
        assert entry["stale"] is False
        assert entry["stale_basis"] == "sha256"
        assert entry["indexed_commit"] == entry["commit"]
    finally:
        db.close()


def test_lexical_ranking_drops_weak_config_noise_for_long_queries(tmp_path: Path):
    (tmp_path / "mcp.json").write_text(
        '{"mcpServers": {"urag": {"command": "urag", "args": ["mcp"]}}}',
        encoding="utf-8",
    )
    (tmp_path / "probe.py").write_text(
        "def run_probe():\n"
        '    """Extract perceptual Malta probe bands to CSV output."""\n'
        "    return 1\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        hits = db.lexical_search(
            "perceptual Malta probe command input JSON binary cached bands output CSV manifest",
            limit=10,
        )
        assert hits
        assert hits[0][0].name == "run_probe"
        assert all(unit.unit_type != "config_key" for unit, _path, _score in hits)
    finally:
        db.close()


def test_lexical_ranking_keeps_config_keys_when_they_are_the_answer(tmp_path: Path):
    (tmp_path / "mcp.json").write_text(
        '{"mcpServers": {"urag": {"command": "urag", "args": ["mcp"]}}}',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    try:
        hits = db.lexical_search("mcp urag command args", limit=5)
        assert any(unit.unit_type == "config_key" for unit, _path, _score in hits)
    finally:
        db.close()
