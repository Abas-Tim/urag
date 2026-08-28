import re

import pytest
from typer.testing import CliRunner

import urag.cli as cli
from urag.config import load_config
from urag.db import Database
from urag.embed import LocalEmbedder, model_cache_subdir, purge_model_cache


def _init(tmp_path):
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    db.close()
    return cfg


def test_default_model_is_bge_base_768(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.embedding.model == "BAAI/bge-base-en-v1.5"
    assert cfg.embedding.dimension == 768


def test_embed_shows_current_config(tmp_path):
    _init(tmp_path)
    result = CliRunner().invoke(cli.app, ["embed", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "BAAI/bge-base-en-v1.5" in result.output
    assert "768" in result.output


def test_embed_switch_clears_embeddings_and_updates_config(tmp_path, monkeypatch):
    cfg = _init(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    db.store_embeddings([(1, "python", "symbol", [0.0] * cfg.embedding.dimension)])
    db.set_meta("embedding_fingerprint", cfg.embedding.fingerprint())
    db.close()

    monkeypatch.setattr(cli, "_detect_local_dimension", lambda m: 384)
    monkeypatch.setattr(cli, "purge_model_cache", lambda m: False)
    result = CliRunner().invoke(
        cli.app,
        ["embed", "--root", str(tmp_path), "--model", "BAAI/bge-small-en-v1.5"],
    )

    assert result.exit_code == 0
    cfg2 = load_config(tmp_path)
    assert cfg2.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg2.embedding.dimension == 384
    db = Database(cfg.db_path, cfg2.embedding.dimension, migrate=True)
    assert db.stats().embedded == 0
    assert db.get_meta("embedding_fingerprint") == ""
    db.close()


def test_embed_dimension_mismatch_fails(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setattr(cli, "_detect_local_dimension", lambda m: 768)
    result = CliRunner().invoke(
        cli.app,
        [
            "embed",
            "--root",
            str(tmp_path),
            "--model",
            "BAAI/bge-small-en-v1.5",
            "--dimension",
            "384",
        ],
    )
    assert result.exit_code != 0
    assert "768" in result.output


def test_embed_unknown_model_requires_dimension(tmp_path, monkeypatch):
    _init(tmp_path)
    monkeypatch.setattr(cli, "_detect_local_dimension", lambda m: None)
    result = CliRunner().invoke(
        cli.app,
        ["embed", "--root", str(tmp_path), "--model", "org/custom-model"],
    )
    assert result.exit_code != 0
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    assert "--dimension" in output


def test_embed_switch_purges_old_model_cache(tmp_path, monkeypatch):
    _init(tmp_path)
    purged = []
    monkeypatch.setattr(cli, "_detect_local_dimension", lambda m: 384)
    monkeypatch.setattr(cli, "purge_model_cache", lambda m: purged.append(m) or True)

    CliRunner().invoke(
        cli.app,
        [
            "embed",
            "--root",
            str(tmp_path),
            "--provider",
            "local",
            "--model",
            "BAAI/bge-small-en-v1.5",
        ],
    )
    assert purged == ["BAAI/bge-base-en-v1.5"]

    purged.clear()
    CliRunner().invoke(
        cli.app,
        [
            "embed",
            "--root",
            str(tmp_path),
            "--model",
            "BAAI/bge-large-en-v1.5",
            "--keep-cache",
        ],
    )
    assert purged == []


def test_embed_switch_works_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_detect_local_dimension", lambda m: 384)
    result = CliRunner().invoke(
        cli.app,
        [
            "embed",
            "--root",
            str(tmp_path),
            "--provider",
            "local",
            "--model",
            "BAAI/bge-small-en-v1.5",
        ],
    )
    assert result.exit_code == 0
    cfg = load_config(tmp_path)
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.dimension == 384
    assert not cfg.db_path.exists()


def test_embed_invalid_provider_fails(tmp_path):
    _init(tmp_path)
    result = CliRunner().invoke(cli.app, ["embed", "--root", str(tmp_path), "--provider", "cloud"])
    assert result.exit_code != 0
    assert "provider must be" in result.output


def test_local_embedder_rejects_dimension_mismatch(tmp_path):
    cfg = load_config(tmp_path)
    cfg.embedding.dimension = 384
    with pytest.raises(RuntimeError, match="768"):
        LocalEmbedder(cfg.embedding)


def test_purge_model_cache_removes_model_dir(tmp_path):
    target = tmp_path / model_cache_subdir("BAAI/bge-small-en-v1.5")
    target.mkdir(parents=True)
    (target / "model.onnx").write_bytes(b"x")
    assert purge_model_cache("BAAI/bge-small-en-v1.5", cache_dir=tmp_path)
    assert not target.exists()
    assert not purge_model_cache("BAAI/bge-small-en-v1.5", cache_dir=tmp_path)
