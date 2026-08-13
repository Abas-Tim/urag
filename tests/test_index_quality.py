from pathlib import Path

from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.indexer import Indexer


def _index(root: Path) -> tuple:
    cfg = load_config(root)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    return cfg, db


def test_fresh_index_has_valid_fts_and_exact_path_search(tmp_path: Path):
    (tmp_path / "auth.py").write_text(
        "def validate_token(token):\n    return token != ''\n", encoding="utf-8"
    )
    cfg, db = _index(tmp_path)
    try:
        assert db.conn.execute("SELECT count(*) FROM fts_units").fetchone()[0] == 1
        assert (
            db.lexical_search("validate_token", exact=True)[0][0].name
            == "validate_token"
        )
        assert db.lexical_search("auth.py", exact=True)[0][1] == "auth.py"
    finally:
        db.close()


def test_deleted_files_remove_vectors(tmp_path: Path):
    path = tmp_path / "auth.py"
    path.write_text("def validate_token(token):\n    return True\n", encoding="utf-8")
    cfg, db = _index(tmp_path)
    try:
        unit_id = db.conn.execute("SELECT id FROM units LIMIT 1").fetchone()[0]
        db.store_embeddings(
            [(unit_id, "python", "symbol", [0.0] * cfg.embedding.dimension)]
        )
        assert db.stats().embedded == 1
        db.delete_files(["auth.py"])
        assert db.stats().embedded == 0
        assert db.counts() == (0, 0)
    finally:
        db.close()


def test_same_size_same_mtime_content_change_is_reindexed(tmp_path: Path):
    path = tmp_path / "module.py"
    path.write_text("def value():\n    return 1\n", encoding="utf-8")
    cfg, db = _index(tmp_path)
    try:
        old_stat = path.stat()
        path.write_text("def value():\n    return 2\n", encoding="utf-8")
        import os

        os.utime(path, (old_stat.st_atime, old_stat.st_mtime))
        stats = Indexer(cfg, db, NoopEmbedder()).index_all()
        assert stats["changed"] == 1
        unit_id = db.conn.execute(
            "SELECT id FROM units WHERE name = 'value'"
        ).fetchone()[0]
        assert "return 2" in db.load_evidence(unit_id)["span"]
    finally:
        db.close()


def test_unit_id_is_stable_when_symbol_body_changes(tmp_path: Path):
    path = tmp_path / "module.py"
    path.write_text("def value():\n    return 1\n", encoding="utf-8")
    cfg, db = _index(tmp_path)
    try:
        first_id = db.conn.execute(
            "SELECT id FROM units WHERE name = 'value'"
        ).fetchone()[0]
        path.write_text("def value():\n    return 2\n", encoding="utf-8")
        Indexer(cfg, db, NoopEmbedder()).index_all()
        second_id = db.conn.execute(
            "SELECT id FROM units WHERE name = 'value'"
        ).fetchone()[0]
        assert first_id == second_id
    finally:
        db.close()


def test_parent_relationships_and_resolved_call_targets(tmp_path: Path):
    (tmp_path / "auth.py").write_text(
        "class Validator:\n    def method(self):\n        return True\n\ndef validate_token(token):\n    return True\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "def login(token):\n    return validate_token(token)\n", encoding="utf-8"
    )
    cfg, db = _index(tmp_path)
    try:
        caller = db.callers("validate_token")
        assert [row["unit"].name for row in caller] == ["login"]
        assert (
            db.conn.execute(
                "SELECT count(*) FROM call_edges WHERE callee_unit_id IS NOT NULL"
            ).fetchone()[0]
            == 1
        )
        class_id = db.conn.execute(
            "SELECT id FROM units WHERE name = 'Validator'"
        ).fetchone()[0]
        parent_id = db.conn.execute(
            "SELECT parent_id FROM units WHERE name = 'method'"
        ).fetchone()[0]
        assert parent_id == class_id
    finally:
        db.close()


def test_incremental_target_ambiguity_re_resolves_existing_edges(tmp_path: Path):
    (tmp_path / "target.py").write_text(
        "def target():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "app.py").write_text(
        "def caller():\n    return target()\n", encoding="utf-8"
    )
    cfg, db = _index(tmp_path)
    try:
        assert (
            db.conn.execute("SELECT callee_unit_id FROM call_edges").fetchone()[0]
            is not None
        )
        (tmp_path / "other.py").write_text(
            "def target():\n    return False\n", encoding="utf-8"
        )

        Indexer(cfg, db, NoopEmbedder()).index_all()

        assert (
            db.conn.execute("SELECT callee_unit_id FROM call_edges").fetchone()[0]
            is None
        )
    finally:
        db.close()


def test_reopening_index_clears_dangling_call_targets(tmp_path: Path):
    (tmp_path / "target.py").write_text(
        "def target():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "app.py").write_text(
        "def caller():\n    return target()\n", encoding="utf-8"
    )
    cfg, db = _index(tmp_path)
    edge = db.conn.execute("SELECT callee_unit_id FROM call_edges").fetchone()
    assert edge[0] is not None
    db.conn.commit()
    db.conn.execute("PRAGMA foreign_keys = OFF")
    db.conn.execute("UPDATE call_edges SET callee_unit_id = ?", (edge[0] + 1000000,))
    db.conn.commit()
    db.close()

    reopened = Database(cfg.db_path, cfg.embedding.dimension)
    try:
        assert (
            reopened.conn.execute("SELECT callee_unit_id FROM call_edges").fetchone()[0]
            is None
        )
    finally:
        reopened.close()


def test_alias_bindings_are_removed_on_reindex(tmp_path: Path):
    path = tmp_path / "module.py"
    path.write_text(
        "import os.path as op\n\ndef check(path):\n    return op.exists(path)\n",
        encoding="utf-8",
    )
    cfg, db = _index(tmp_path)
    try:
        assert db.conn.execute("SELECT count(*) FROM import_aliases").fetchone()[0] == 1
        path.write_text("def check(path):\n    return exists(path)\n", encoding="utf-8")
        Indexer(cfg, db, NoopEmbedder()).index_all()
        assert db.conn.execute("SELECT count(*) FROM import_aliases").fetchone()[0] == 0
    finally:
        db.close()
