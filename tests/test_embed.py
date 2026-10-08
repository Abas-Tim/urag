import json
import re
from contextlib import closing

import pytest
from typer.testing import CliRunner

import urag.cli as cli
from urag.config import EmbeddingConfig, load_config
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


def test_local_passage_batches_restore_input_order(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        @staticmethod
        def get_embedding_size(model):
            return 2

        def __init__(self, **kwargs):
            assert kwargs["threads"] == 3

        def embed(self, texts, *, batch_size):
            calls.append((texts, batch_size))
            return [[float(len(text)), float(ord(text[0]))] for text in texts]

    monkeypatch.setattr("fastembed.TextEmbedding", FakeModel)
    cfg = EmbeddingConfig(dimension=2, threads=3, batch_size=2)
    emb = LocalEmbedder(cfg, cache_dir=tmp_path)
    assert emb.embed_passages(["longest", "a", "medium"]) == [
        [7.0, 108.0],
        [1.0, 97.0],
        [6.0, 109.0],
    ]
    assert calls == [(["a", "medium", "longest"], 2)]
    assert emb.embed_passages([]) == []
    assert len(calls) == 1


def test_execution_settings_persist_without_invalidating_vectors(tmp_path):
    cfg = _init(tmp_path)
    fingerprint = cfg.embedding.fingerprint()
    with closing(Database(cfg.db_path, cfg.embedding.dimension)) as db:
        db.store_embeddings([(1, "python", "symbol", [0.0] * cfg.embedding.dimension)])
    result = CliRunner().invoke(
        cli.app, ["embed", "--root", str(tmp_path), "--threads", "4", "--batch-size", "16"]
    )
    assert result.exit_code == 0, result.output
    loaded = load_config(tmp_path)
    assert loaded.embedding.threads == 4
    assert loaded.embedding.batch_size == 16
    assert loaded.embedding.fingerprint() == fingerprint
    with closing(Database(cfg.db_path, cfg.embedding.dimension)) as db:
        assert db.stats().embedded == 1


@pytest.mark.parametrize("name", ["batch_size", "threads"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "auto"])
def test_rejects_invalid_execution_settings(tmp_path, name, value):
    cfg = load_config(tmp_path)
    cfg.urag_dir.mkdir(parents=True)
    cfg.config_path.write_text(f"[embedding]\n{name} = {json.dumps(value)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"embedding.{name}"):
        load_config(tmp_path)


def test_purge_model_cache_removes_model_dir(tmp_path):
    target = tmp_path / model_cache_subdir("BAAI/bge-small-en-v1.5")
    target.mkdir(parents=True)
    (target / "model.onnx").write_bytes(b"x")
    assert purge_model_cache("BAAI/bge-small-en-v1.5", cache_dir=tmp_path)
    assert not target.exists()
    assert not purge_model_cache("BAAI/bge-small-en-v1.5", cache_dir=tmp_path)
