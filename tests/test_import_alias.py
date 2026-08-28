"""Tests for import-alias resolution in call-graph lookups."""

from pathlib import Path

import pytest

from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.extractors.native_ext import (
    CExtractor,
    CSharpExtractor,
    GoExtractor,
    JavaExtractor,
    RustExtractor,
)
from urag.extractors.python_ext import PythonExtractor
from urag.extractors.ts_ext import TsExtractor
from urag.indexer import Indexer
from urag.retrieve import Retriever


def _aliases(ext, src):
    return {a: t for a, t in ext.collect_import_aliases(src)}


def test_python_aliases():
    src = "import os.path as opath\nfrom core.http import fetch as http_fetch\nfrom core.auth import validate\nimport plain\n"
    got = _aliases(PythonExtractor(), src)
    assert got == {
        "opath": "os.path",
        "http_fetch": "core.http.fetch",
        "validate": "core.auth.validate",
    }
    assert "plain" not in got


def test_ts_aliases():
    src = (
        'import * as net from "sdk/net";\n'
        'import { Client as CoreClient } from "sdk/core";\n'
        'import { Logger } from "sdk/log";\n'
        'import Default from "sdk/def";\n'
    )
    got = _aliases(TsExtractor("typescript"), src)
    assert got == {
        "net": "sdk.net",
        "CoreClient": "sdk.core.Client",
        "Logger": "sdk.log.Logger",
        "Default": "sdk.def.default",
    }


def test_go_rust_csharp_aliases():
    assert _aliases(GoExtractor(), 'import hw "myproj/helpers"') == {
        "hw": "myproj.helpers"
    }
    assert _aliases(RustExtractor(), "use core::auth::validate as check;") == {
        "check": "core::auth::validate"
    }
    assert _aliases(RustExtractor(), "pub use crate::foo::{Bar as Baz, Qux};") == {
        "Baz": "crate::foo::Bar"
    }
    assert _aliases(CSharpExtractor(), "using Log = Common.Logging;") == {
        "Log": "Common.Logging"
    }
    assert _aliases(CExtractor("c"), "#include <stdio.h>") == {}
    assert _aliases(
        JavaExtractor(), "import com.x.Y;\nimport static com.x.Util.foo;"
    ) == {
        "Y": "com.x.Y",
        "foo": "com.x.Util.foo",
    }


# ---------------------------------------------------------------------------
# end-to-end resolution on an indexed file set
# ---------------------------------------------------------------------------

SRC_APP = """import os.path as opath

def is_file(path):
    return opath.exists(path)
"""

SRC_APP2 = """from core.http import fetch

def get_user():
    return fetch("/user")
"""

SRC_SHADOW = """from core.http import fetch

def fetch(url):
    return url

def local_call():
    return fetch("/local")
"""


@pytest.fixture
def db(tmp_path: Path):
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    for path, source in (
        ("app.py", SRC_APP),
        ("api.py", SRC_APP2),
        ("shadow.py", SRC_SHADOW),
    ):
        (tmp_path / path).write_text(source, encoding="utf-8")
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        yield db
    finally:
        db.close()


def _name(db, unit_id):
    u, _, _ = db.unit_by_id(unit_id)
    return u.name


def test_alias_resolved(db):
    hits = db.callers("os.path.exists")
    assert [_name(db, h["unit"].id) for h in hits] == ["is_file"]
    assert hits[0]["resolved_target"] == "os.path"
    assert hits[0]["callee_full"] == "opath.exists"


def test_bare_from_import_resolved(db):
    hits = db.callers("core.http.fetch")
    assert [_name(db, h["unit"].id) for h in hits] == ["get_user"]


def test_local_shadowing_wins(db):
    hits = db.callers("core.http.fetch")
    names = {_name(db, h["unit"].id) for h in hits}
    assert "local_call" not in names
    assert "fetch" not in names


def test_last_segment_still_works(db):
    hits = db.callers("exists")
    assert [_name(db, h["unit"].id) for h in hits] == ["is_file"]


def test_fully_qualified_direct_chain(db):
    # non-aliased direct chain still matches
    hits = db.callers("os.path.exists")
    assert len(hits) == 1
    hits2 = db.callers("core.http.fetch")
    assert len(hits2) == 1


def test_no_false_positive_on_bare_call(db):
    # query for the shadowed module path must not surface the local function
    hits = db.callers("other.mod.fetch")
    assert hits == []


def test_search_callers_carries_resolution(db, tmp_path):
    cfg = load_config(tmp_path)
    r = Retriever(cfg, db, NoopEmbedder())
    result = r.search_callers("os.path.exists")
    assert result.results[0].resolved_target == "os.path"
    d = result.to_dict()
    assert d["results"][0]["resolved_to"] == "os.path"


def test_indexer_populates_aliases(tmp_path):
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    src = "import os.path as opath\n\ndef f():\n    return opath.exists('x')\n"
    (tmp_path / "m.py").write_text(src, encoding="utf-8")
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        row = db.conn.execute("SELECT alias, target FROM import_aliases").fetchone()
        assert dict(row) == {"alias": "opath", "target": "os.path"}
    finally:
        db.close()
