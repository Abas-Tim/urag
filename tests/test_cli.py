import json
from contextlib import closing

from typer.testing import CliRunner

import urag.cli as cli
from urag.config import load_config
from urag.db import Database
from urag.embed import Embedder, NoopEmbedder
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
    assert "loading embedding model" in captured.err


def test_init_reports_progress_before_model_loading_and_embedding(tmp_path, monkeypatch, capsys):
    (tmp_path / "m.py").write_text("def alpha(): return 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)

    class FakeEmbedder(Embedder):
        @property
        def dimension(self):
            return cfg.embedding.dimension

        def embed_passages(self, texts):
            assert "embedding " in capsys.readouterr().out
            return [[0.0] * self.dimension for _ in texts]

        def embed_query(self, text):
            return [0.0] * self.dimension

    def load(_cfg):
        output = capsys.readouterr().out
        assert "indexing " in output
        assert "lexical and graph index ready" in output
        assert "local CPU embeddings" in output
        return FakeEmbedder()

    monkeypatch.setattr(cli, "_embedder", load)
    cli.init(root=tmp_path, full=True, no_embed=False)
    assert "embedded " in capsys.readouterr().out


def test_deferred_init_then_embedding_only_does_not_scan(tmp_path, monkeypatch):
    (tmp_path / "m.py").write_text("def alpha(): return 1\n", encoding="utf-8")

    def unexpected_load(_cfg):
        raise AssertionError("deferred init must not load a model")

    monkeypatch.setattr(cli, "_embedder", unexpected_load)
    result = CliRunner().invoke(
        cli.app, ["init", "--full", "--defer-embeddings", "--root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "lexical and graph index ready" in result.output
    assert "dense embeddings deferred" in result.output
    for mode in ("lexical", "hybrid"):
        result = CliRunner().invoke(
            cli.app, ["search", "alpha", "--mode", mode, "--json", "--root", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["results"][0]["name"] == "alpha"
    result = CliRunner().invoke(cli.app, ["resolve", "alpha", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    cfg = load_config(tmp_path)
    with closing(Database(cfg.db_path, cfg.embedding.dimension)) as db:
        assert db.lexical_search("alpha", exact=True)
        assert db.stats().embedded == 0

    class FakeEmbedder(Embedder):
        @property
        def dimension(self):
            return cfg.embedding.dimension

        def embed_passages(self, texts):
            return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * (self.dimension - 1)

    monkeypatch.setattr(cli, "_embedder", lambda _cfg: FakeEmbedder())

    def unexpected_scan(_self):
        raise AssertionError("embedding-only phase must not scan source files")

    monkeypatch.setattr(Indexer, "discover", unexpected_scan)
    result = CliRunner().invoke(cli.app, ["index", "--embeddings-only", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    with closing(Database(cfg.db_path, cfg.embedding.dimension)) as db:
        assert db.stats().embedded == 1

    result = CliRunner().invoke(cli.app, ["index", "--embeddings-only", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "embedded 0 pending units" in result.output


def test_index_rejects_conflicting_embedding_phases(tmp_path):
    result = CliRunner().invoke(
        cli.app,
        ["index", "--embeddings-only", "--defer-embeddings", "--root", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


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
