from typer.testing import CliRunner

import urag.cli as cli
from urag.config import load_config
from urag.db import Database


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
