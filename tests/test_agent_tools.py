"""Tests for the new agent-facing tools (MCP + Retriever navigation)."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.indexer import Indexer
from urag.mcp_server import create_server, _evidence_budget
from urag.retrieve import Retriever

APP_PY = '''from auth import validate_token


class TokenValidator:
    """Validates access tokens."""

    def validate(self, token):
        return validate_token(token)

    def is_expired(self, ttl):
        return ttl <= 0


def handle_request(token):
    v = TokenValidator()
    return v.validate(token)
'''

AUTH_PY = '''def validate_token(token):
    """Return whether a token is non-empty."""
    return token != ""
'''


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "auth.py").write_text(AUTH_PY, encoding="utf-8")
    (tmp_path / "app.py").write_text(APP_PY, encoding="utf-8")
    (tmp_path / "config.json").write_text(
        '{\n  "server": {"host": "0.0.0.0", "port": 8080}\n}\n', encoding="utf-8"
    )
    (tmp_path / ".env").write_text("LOG_LEVEL=debug\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    Indexer(cfg, db, NoopEmbedder()).index_all()
    yield cfg, db
    db.close()


def _unit_id(db, name):
    row = db.conn.execute(
        "SELECT u.id FROM units u WHERE u.name = ? ORDER BY u.start_line LIMIT 1",
        (name,),
    ).fetchone()
    assert row, f"unit {name} not indexed"
    return row["id"]


def _call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    return json.loads(result.content[0].text)


def test_evidence_budget_distribution():
    assert _evidence_budget(1500, 1) == 1500
    assert _evidence_budget(1500, 5) == 300
    assert _evidence_budget(100, 5) == 200


def test_config_and_env_files_indexed(project):
    cfg, db = project
    langs = {r["language"] for r in db.file_list()}
    assert "json" in langs
    assert "env" in langs
    config_units = db.units_by_file_path("config.json")
    assert {u.qualname for u in config_units} >= {"server.host", "server.port"}


def test_resolve_exact_symbol(project):
    cfg, db = project
    result = Retriever(cfg, db, NoopEmbedder()).resolve("validate_token")
    assert result.results, "resolve should find validate_token"
    assert result.results[0].unit.qualname == "validate_token"
    assert result.results[0].file_path == "auth.py"


def test_callees_of_method(project):
    cfg, db = project
    validate_id = _unit_id(db, "validate")
    result = Retriever(cfg, db, NoopEmbedder()).callees(validate_id)
    assert result is not None
    callees = {c["callee"] for c in result["callees"]}
    assert "validate_token" in callees


def test_dependents_of_module(project):
    cfg, db = project
    result = Retriever(cfg, db, NoopEmbedder()).dependents("auth")
    paths = {r["path"] for r in result["results"]}
    assert "app.py" in paths


def test_children_of_class(project):
    cfg, db = project
    class_id = _unit_id(db, "TokenValidator")
    result = Retriever(cfg, db, NoopEmbedder()).children(class_id)
    names = {r.unit.name for r in result.results}
    assert {"validate", "is_expired"} <= names


def test_siblings_of_method(project):
    cfg, db = project
    validate_id = _unit_id(db, "validate")
    result = Retriever(cfg, db, NoopEmbedder()).children(
        validate_id, include_siblings=True
    )
    names = {r.unit.name for r in result.results}
    assert "is_expired" in names


def test_list_files_and_symbols(project):
    cfg, db = project
    retriever = Retriever(cfg, db, NoopEmbedder())
    files = retriever.list_files()
    assert files["count"] >= 4
    paths = {f["path"] for f in files["files"]}
    assert {"auth.py", "app.py", "config.json"} <= paths

    symbols = retriever.list_symbols("app.py")
    names = {r.unit.name for r in symbols.results}
    assert {"TokenValidator", "validate", "handle_request"} <= names


def test_read_file_range(project):
    cfg, db = project
    result = Retriever(cfg, db, NoopEmbedder()).read_file("auth.py", start=1, end=1)
    assert result["start_line"] == 1
    assert "def validate_token" in result["span"]


def test_read_file_rejects_escape(project):
    cfg, db = project
    result = Retriever(cfg, db, NoopEmbedder()).read_file("../secret.py")
    assert "error" in result


def test_get_many(project):
    cfg, db = project
    ids = [_unit_id(db, "validate_token"), _unit_id(db, "handle_request")]
    evs = Retriever(cfg, db, NoopEmbedder()).get_many(ids)
    assert len(evs) == 2
    assert all("span" in ev for ev in evs)


def test_mcp_new_tools(project):
    cfg, db = project
    server = create_server(cfg.project_root)

    files = _call(server, "list_files", {})
    assert files["count"] >= 4

    resolved = _call(server, "resolve", {"name": "validate_token"})
    assert resolved["count"] >= 1
    assert resolved["results"][0]["file"] == "auth.py"

    validate_id = _unit_id(db, "validate")
    callees = _call(server, "callees", {"unit_id": validate_id})
    assert "validate_token" in {c["callee"] for c in callees["callees"]}

    deps = _call(server, "dependents", {"target": "auth"})
    assert "app.py" in {r["path"] for r in deps["results"]}

    symbols = _call(server, "list_symbols", {"file": "app.py"})
    assert {"TokenValidator", "handle_request"} <= {
        r["name"] for r in symbols["results"]
    }

    read = _call(server, "read_file", {"path": "auth.py", "start": 1, "end": 1})
    assert "def validate_token" in read["span"]

    fetched = _call(server, "fetch_unit", {"unit_id": validate_id})
    assert fetched["name"] == "validate"
    assert "span" in fetched

    batch = _call(
        server,
        "fetch_units",
        {"unit_ids": [_unit_id(db, "validate_token"), _unit_id(db, "handle_request")]},
    )
    assert batch["count"] == 2
    assert all("name" in r and "span" in r for r in batch["results"])


def test_mcp_init_project_lexical_only(tmp_path):
    (tmp_path / "a.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    server = create_server(tmp_path)
    result = _call(server, "init_project", {"embed": False})
    assert result["initialized"] is True
    assert result["units"] >= 1
    assert result["embedded"] == 0


def test_mcp_status_includes_git_and_next_step(tmp_path):
    server = create_server(tmp_path)
    status = _call(server, "status", {})
    assert status["error"] == "index missing; call init_project"
    assert "init_project" in status["next"]


def test_mcp_unit_resource(project):
    cfg, db = project
    server = create_server(cfg.project_root)
    row = db.conn.execute(
        "SELECT u.id FROM units u WHERE u.name = 'validate_token' "
        "AND u.unit_type != 'import' LIMIT 1"
    ).fetchone()
    uid = row["id"]
    result = asyncio.run(server.read_resource(f"urag://unit/{uid}"))
    contents = result if isinstance(result, list) else result.contents
    text = contents[0].content
    assert "def validate_token" in text


def test_recent_changes_in_git_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=str(tmp_path), check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(tmp_path), check=True)
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=str(tmp_path), check=True)
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")

    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    server = create_server(tmp_path)
    result = _call(server, "recent_changes", {"limit": 5})
    assert result["branch"] == "master" or result["branch"] == "main"
    assert "b.py" in result["working"]["untracked"]
    assert result["commits"][0]["subject"] == "initial"
