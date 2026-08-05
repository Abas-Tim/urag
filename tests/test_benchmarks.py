"""Tests for the call-graph benchmark suites (transitive + import-alias)."""

from pathlib import Path
import shutil

import pytest

from urag.config import load_config
from urag.db import Database
from urag.eval import (
    Hit,
    Question,
    SystemRun,
    _metrics,
    aggregate,
    autogen_alias_questions,
    autogen_transitive_questions,
    scan_import_aliases,
    transitive_caller_ids,
)
from urag.indexer import Indexer
from urag.embed import NoopEmbedder

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks" / "fixtures" / "callgraph_fixture"


@pytest.fixture(scope="module")
def fdb(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    for src in FIXTURE.rglob("*"):
        if src.is_file():
            dest = root / src.relative_to(FIXTURE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    cfg = load_config(root)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
    except RuntimeError:
        pass  # embeddings skipped; call-graph tests don't need them
    yield db
    db.close()


def test_scan_python_aliases():
    src = "import os.path as opath\nfrom core.http import fetch as http_fetch\nimport core.auth as auth\n"
    assert scan_import_aliases(src, "python") == {
        "opath": "os.path",
        "http_fetch": "core.http.fetch",
        "auth": "core.auth",
    }


def test_scan_ts_aliases():
    src = 'import { Client as CoreClient } from "sdk/core";\nimport * as net from "sdk/net";\n'
    assert scan_import_aliases(src, "typescript") == {
        "CoreClient": "sdk.core.Client",
        "net": "sdk.net",
    }


def test_scan_go_csharp_aliases():
    assert scan_import_aliases('import hw "myproj/helpers"', "go") == {"hw": "myproj.helpers"}
    assert scan_import_aliases("using Log = Common.Logging;", "csharp") == {"Log": "Common.Logging"}


def _unit_id(db, name):
    row = db.conn.execute(
        "SELECT u.id FROM units u WHERE u.name = ? ORDER BY u.start_line LIMIT 1", (name,)
    ).fetchone()
    assert row, f"unit {name} not indexed"
    return row["id"]


def test_transitive_ids_depth(fdb):
    hops = transitive_caller_ids(fdb, "stop", max_depth=3)
    assert hops[_unit_id(fdb, "stage3")] == 1
    assert hops[_unit_id(fdb, "stage2")] == 2
    assert hops[_unit_id(fdb, "stage1")] == 3
    assert hops[_unit_id(fdb, "other_entry")] == 3
    assert _unit_id(fdb, "entry") not in hops  # hop 4, beyond max_depth


def test_autogen_transitive(fdb):
    qs = autogen_transitive_questions(fdb, 20)
    stop = [q for q in qs if q.target == "stop"]
    assert stop
    q = stop[0]
    assert q.label == "transitive"
    assert q.gold_hops[_unit_id(fdb, "stage3")] == 1
    assert q.gold_hops[_unit_id(fdb, "stage1")] == 3
    assert q.gold_hops[_unit_id(fdb, "other_entry")] == 3
    assert q.query == "who transitively calls stop"
    assert q.depth == 3


def test_autogen_alias_python(fdb):
    qs = autogen_alias_questions(fdb, FIXTURE, 50)
    targets = {q.target for q in qs}
    assert "os.path.exists" in targets
    assert "os.path.isdir" in targets
    assert "core.http.fetch" in targets
    assert "core.auth.validate" in targets
    exists = [q for q in qs if q.target == "os.path.exists"]
    assert exists[0].gold_unit_ids == [_unit_id(fdb, "is_file")]
    fetch = [q for q in qs if q.target == "core.http.fetch"]
    owners = {gid for q in fetch for gid in q.gold_unit_ids}
    assert owners == {_unit_id(fdb, "get_user"), _unit_id(fdb, "get_item")}
    # fully-qualified target also matches the non-aliased last-segment caller
    validate = [q for q in qs if q.target == "core.auth.validate"]
    assert validate[0].gold_unit_ids == [_unit_id(fdb, "login")]


def test_autogen_alias_other_langs(fdb):
    qs = autogen_alias_questions(fdb, FIXTURE, 50)
    targets = {q.target for q in qs}
    assert "sdk.net.connect" in targets
    assert "sdk.core.makeClient" in targets
    assert "Common.Logging.Info" in targets
    assert "myproj.helpers.Start" in targets


def test_metrics_precision_and_indirect():
    q = Question(
        query="q",
        gold_unit_ids=[7, 8, 9],
        label="transitive",
        target="x",
        gold_hops={7: 1, 8: 2, 9: 3},
    )
    sr = SystemRun(
        "t",
        hits=[Hit("a.py", 7, 10), Hit("b.py", 8, 10), Hit("c.py", 1, 10)],
        seconds=0.01,
        tokens=30,
    )
    m = _metrics(sr, q, top_k=5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["indirect_recall"] == pytest.approx(0.5)  # one of {8,9} found
    a = aggregate([m])
    assert a["precision"] == pytest.approx(2 / 3)
    assert a["indirect_recall"] == pytest.approx(0.5)


def test_metrics_indirect_none_for_non_transitive():
    q = Question(query="q", gold_unit_ids=[7], label="call", target="x")
    m = _metrics(SystemRun("t", [Hit("a.py", 7, 5)], 0.0, 5), q, top_k=5)
    assert m["indirect_recall"] is None
    assert aggregate([m])["indirect_recall"] == 0.0


def test_question_round_trip():
    q = Question(query="q", gold_unit_ids=[1], label="transitive", target="stop", depth=3, gold_hops={1: 2})
    d = q.to_dict()
    assert d["target"] == "stop"
    assert d["depth"] == 3
    assert d["gold_hops"] == {"1": 2}
