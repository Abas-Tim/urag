import json

from typer.testing import CliRunner

import urag.cli as cli
from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.indexer import Indexer


def _empty_index(tmp_path):
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    db.close()


def test_get_missing_unit_exits_without_traceback(tmp_path):
    _empty_index(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        ["get", "999999", "--root", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "unit not found" in result.output
    assert "TypeError" not in result.output


def test_callees_accepts_symbol_name(tmp_path):
    (tmp_path / "m.py").write_text(
        "def alpha(x):\n    return x + 1\n\n\ndef beta():\n    return alpha(1)\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    db.close()

    result = CliRunner().invoke(cli.app, ["callees", "beta", "--root", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["qualname"] == "beta"
    assert any(c["callee"] == "alpha" for c in payload["callees"])

    missing = CliRunner().invoke(cli.app, ["callees", "nope", "--root", str(tmp_path)])
    assert missing.exit_code == 1
    assert "no definition found" in missing.output


def test_embedding_warning_is_written_to_stderr(tmp_path, monkeypatch, capsys):
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "local"
    cli._embedder_cache.clear()

    def fail(_cfg):
        raise RuntimeError("test failure")

    monkeypatch.setattr(cli, "create_embedder", fail)
    cli._embedder(cfg)
    captured = capsys.readouterr()

    assert "embedding unavailable" in captured.err
    assert "loading embedding model" in captured.out


def test_status_json_output(tmp_path):
    _empty_index(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    db.close()

    result = CliRunner().invoke(cli.app, ["status", "--root", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["files"] == 1
    assert payload["units"] >= 0
    assert payload["embedding"]["provider"] == "none"


def test_doctor_json_output(tmp_path):
    _empty_index(tmp_path)
    result = CliRunner().invoke(cli.app, ["doctor", "--root", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["embedding"]["provider"] == "none"
    assert "coverage" in payload


def test_doctor_reports_coverage_by_reason(tmp_path):
    _empty_index(tmp_path)
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "data.xyz").write_text("binary-ish\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    db.close()

    result = CliRunner().invoke(cli.app, ["doctor", "--root", str(tmp_path), "--json"])

    assert result.exit_code == 0
    coverage = json.loads(result.output)["coverage"]
    assert coverage["counts"]["indexed"] == 1
    assert coverage["counts"]["unsupported_extension"] == 1
    assert coverage["unsupported_extensions"][".xyz"] == 1
    assert coverage["examples"]["unsupported_extension"] == ["data.xyz"]


def test_read_json_output(tmp_path):
    _empty_index(tmp_path)
    (tmp_path / "notes.md").write_text("# title\n\nbody line\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    db.close()

    result = CliRunner().invoke(cli.app, ["read", "notes.md", "--root", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["path"] == "notes.md"
    assert "body line" in payload["span"]
