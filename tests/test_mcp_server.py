import asyncio
import json

from urag.config import default_config
from urag.mcp_server import create_server


def _call(server, name, arguments):
    result = asyncio.run(server.call_tool(name, arguments))
    return json.loads(result.content[0].text)


def test_mcp_reports_missing_index_without_creating_database(tmp_path):
    server = create_server(tmp_path)

    status = _call(server, "status", {})
    search = _call(server, "search", {"query": "anything"})
    callers = _call(server, "callers", {"name": "anything"})
    fetch = _call(server, "fetch_unit", {"unit_id": 1})
    index = _call(server, "index_now", {})

    for response in (status, search, callers, fetch, index):
        assert response["error"] == "index missing; call init_project"
    assert not (tmp_path / ".urag" / "index.db").exists()


def test_mcp_reports_corrupt_index(tmp_path):
    (tmp_path / ".urag").mkdir()
    (tmp_path / ".urag" / "index.db").write_bytes(b"not a sqlite database")
    server = create_server(tmp_path)

    response = _call(server, "status", {})

    assert response["error"].startswith("index unavailable:")


def test_mcp_init_project_adds_gitignore_entry(tmp_path):
    cfg = default_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    server = create_server(tmp_path)

    result = _call(server, "init_project", {})

    assert result["initialized"] is True
    assert (
        ".urag/" in (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    )
