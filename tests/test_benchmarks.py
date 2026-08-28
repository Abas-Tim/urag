"""Tests for the call-graph benchmark suites (transitive + import-alias)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.eval import (
    Hit,
    OracleBaseline,
    Question,
    ReadBaseline,
    RgBaseline,
    SystemRun,
    _metrics,
    aggregate,
    autogen_alias_questions,
    autogen_questions,
    autogen_transitive_questions,
    load_questions,
    reresolve_questions,
    resolve_question,
    scan_import_aliases,
    transitive_caller_ids,
)
from urag.indexer import Indexer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import html_report  # noqa: E402
import run_bench  # noqa: E402

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fixtures"
    / "callgraph_fixture"
)


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
        yield db
    finally:
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
    assert scan_import_aliases('import hw "myproj/helpers"', "go") == {
        "hw": "myproj.helpers"
    }
    assert scan_import_aliases("using Log = Common.Logging;", "csharp") == {
        "Log": "Common.Logging"
    }


def _unit_id(db, name):
    row = db.conn.execute(
        "SELECT u.id FROM units u WHERE u.name = ? ORDER BY u.start_line LIMIT 1",
        (name,),
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
    assert len(fetch) == 1
    assert set(fetch[0].gold_unit_ids) == {
        _unit_id(fdb, "get_user"),
        _unit_id(fdb, "get_item"),
    }
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


def test_question_round_trip(tmp_path):
    q = Question(
        query="q",
        gold_unit_ids=[1],
        gold_file="m.py",
        label="transitive",
        target="stop",
        depth=3,
        gold_hops={1: 2},
    )
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(q.to_dict()) + "\n", encoding="utf-8")

    assert load_questions(path) == [q]


def test_question_round_trip_legacy_schema(tmp_path):
    legacy = {
        "query": "q",
        "gold_unit_ids": [1],
        "gold_file": "m.py",
        "label": "definition",
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    q = load_questions(path)[0]
    assert q.gold_files == ["m.py"]
    assert q.gold_file == "m.py"


def test_resolve_gold_files(fdb):
    q = resolve_question(
        fdb,
        Question(
            query="q", gold_unit_ids=[_unit_id(fdb, "entry"), _unit_id(fdb, "get_user")]
        ),
    )
    assert q.gold_files == ["app/api.py", "core/chain.py"]
    assert q.gold_file == "app/api.py"
    q2 = resolve_question(fdb, Question(query="q", gold_file="core/chain.py"))
    assert q2.gold_files == ["core/chain.py"]
    assert q2.gold_unit_ids


def test_oracle_covers_all_gold_files(fdb):
    ids = [_unit_id(fdb, "entry"), _unit_id(fdb, "get_user")]
    q = Question(query="q", gold_unit_ids=ids, gold_file="core/chain.py")
    run = OracleBaseline(FIXTURE).search(q, fdb)
    assert {h.file for h in run.hits} == {"app/api.py", "core/chain.py"}
    assert {h.unit_id for h in run.hits} == set(ids)


def test_metrics_file_recall_fractional():
    q = Question(query="q", gold_files=["a.py", "b.py"])
    m = _metrics(SystemRun("t", [Hit("a.py", 1, 5)], 0.01, 5), q, top_k=5)
    assert m["file_recall"] == pytest.approx(0.5)
    assert m["mrr"] == pytest.approx(1.0)


def test_metrics_read_baseline_unit_ids():
    q = Question(query="q", gold_unit_ids=[7, 8])
    sr = SystemRun("read", [Hit("f.py", 7, 100, unit_ids=[7, 8, 9])], 0.0, 100)
    m = _metrics(sr, q, top_k=5)
    assert m["unit_recall"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx(1.0)


def test_autogen_questions(fdb):
    qs = autogen_questions(fdb, 3)
    assert len(qs) == 6
    assert {q.label for q in qs} == {"definition", "call"}
    assert all(q.gold_unit_ids for q in qs)
    assert all(q.query.startswith("where is ") for q in qs if q.label == "definition")
    assert all(q.target for q in qs if q.label == "call")


def test_reresolve_questions(fdb):
    trans = autogen_transitive_questions(fdb, 10)
    rebuilt = reresolve_questions(fdb, trans)
    assert len(rebuilt) == len(trans)
    for a, b in zip(trans, rebuilt):
        assert a.gold_unit_ids == b.gold_unit_ids
        assert a.gold_hops == b.gold_hops

    defs = [q for q in autogen_questions(fdb, 2) if q.label == "definition"]
    red = reresolve_questions(fdb, defs)
    assert [q.gold_unit_ids for q in red] == [q.gold_unit_ids for q in defs]

    calls = [q for q in autogen_questions(fdb, 2) if q.label == "call"]
    rec = reresolve_questions(fdb, calls)
    for a, b in zip(calls, rec):
        assert set(a.gold_unit_ids) == set(b.gold_unit_ids)


def test_rg_and_read_baselines(fdb):
    if not shutil.which("rg"):
        pytest.skip("ripgrep not installed")
    rg = RgBaseline(FIXTURE)
    run = rg.search("stage3", 5, fdb)
    assert run.hits
    assert all(h.tokens >= 1 for h in run.hits)
    assert any("core/chain.py" in h.file for h in run.hits)
    assert any(h.detail for h in run.hits)

    stop = rg.search("who transitively calls stop", 5, fdb)
    assert stop.hits
    assert any("core/chain.py" in h.file for h in stop.hits)

    read = ReadBaseline(FIXTURE)
    rrun = read.search("stage3", 5, fdb)
    assert rrun.hits
    assert all(h.unit_ids for h in rrun.hits)
    assert rrun.tokens > 0


def test_html_report_renders(capsys):
    report = {
        "schema_version": 2,
        "urag_version": "test",
        "top_k": 3,
        "questions": [
            {
                "query": "where is stop defined",
                "label": "definition",
                "gold_file": "core/chain.py",
                "gold_files": ["core/chain.py"],
                "gold_unit_ids": [1],
                "target": "",
                "depth": 1,
                "gold_hops": {},
            }
        ],
        "systems": {
            "urag-hybrid": {
                "n": 1,
                "unit_recall": 1.0,
                "file_recall": 1.0,
                "precision": 1.0,
                "indirect_recall": 0.0,
                "mrr": 1.0,
                "mean_tokens": 30,
                "mean_sec": 0.01,
                "p50_sec": 0.01,
                "p95_sec": 0.01,
            },
            "rg": {
                "n": 1,
                "unit_recall": 0.0,
                "file_recall": 0.0,
                "precision": 0.0,
                "indirect_recall": 0.0,
                "mrr": 0.0,
                "mean_tokens": 40,
                "mean_sec": 0.02,
                "p50_sec": 0.02,
                "p95_sec": 0.02,
            },
            "read": {
                "n": 1,
                "unit_recall": 1.0,
                "file_recall": 1.0,
                "precision": 0.1,
                "indirect_recall": 0.0,
                "mrr": 1.0,
                "mean_tokens": 400,
                "mean_sec": 0.001,
                "p50_sec": 0.001,
                "p95_sec": 0.001,
            },
        },
        "per_query": {
            "urag-hybrid": [
                {
                    "unit_recall": 1.0,
                    "file_recall": 1.0,
                    "precision": 1.0,
                    "indirect_recall": 0.0,
                    "mrr": 1.0,
                    "tokens": 30,
                    "seconds": 0.01,
                    "n_hits": 1,
                }
            ],
            "rg": [
                {
                    "unit_recall": 0.0,
                    "file_recall": 0.0,
                    "precision": 0.0,
                    "indirect_recall": 0.0,
                    "mrr": 0.0,
                    "tokens": 40,
                    "seconds": 0.02,
                    "n_hits": 1,
                }
            ],
            "read": [
                {
                    "unit_recall": 1.0,
                    "file_recall": 1.0,
                    "precision": 0.1,
                    "indirect_recall": 0.0,
                    "mrr": 1.0,
                    "tokens": 400,
                    "seconds": 0.001,
                    "n_hits": 1,
                }
            ],
        },
        "hits": {
            "urag-hybrid": [
                [
                    {
                        "file": "core/chain.py",
                        "unit_id": 1,
                        "unit_ids": [],
                        "tokens": 30,
                        "title": "stop()",
                        "detail": "def stop():\n    pass",
                    }
                ]
            ],
            "rg": [
                [
                    {
                        "file": "core/chain.py",
                        "unit_id": None,
                        "unit_ids": [],
                        "tokens": 40,
                        "title": "",
                        "detail": "3: stop()",
                    }
                ]
            ],
            "read": [
                [
                    {
                        "file": "core/chain.py",
                        "unit_id": 1,
                        "unit_ids": [1, 2, 3],
                        "tokens": 400,
                        "title": "",
                        "detail": "def entry(): ...",
                    }
                ]
            ],
        },
        "index_seconds": 1.2,
    }
    html = html_report.render_report(report, title="t.json")
    assert html.startswith("<!DOCTYPE html>")
    assert "urag hybrid" in html
    assert "grep (opencode Grep)" in html
    assert "read file (opencode Read)" in html
    assert "def stop()" in html
    assert "gold" in html
    assert "Token efficiency" in html


def test_summarize_warns_on_empty_system(capsys):
    run_bench.summarize("t", {"systems": {"broken": {}}})
    assert "no results" in capsys.readouterr().out


def test_git_head_format():
    head = run_bench.git_head(Path(__file__).resolve().parents[1])
    assert "@" in head


def test_benchmark_declined_root_is_not_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    called = []

    monkeypatch.setattr(run_bench, "ROOT", tmp_path)
    monkeypatch.setattr(run_bench, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(run_bench, "git_head", lambda _root: "test@123")
    monkeypatch.setattr(
        run_bench,
        "bench",
        lambda root, *_args: called.append(root) or {"systems": {}, "questions": []},
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--root", str(root), "--no-html"],
    )

    run_bench.main()

    assert called == [run_bench.FIXTURE]


def test_reuse_report_runs_only_recorded_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    report = tmp_path / "before.json"
    report.write_text(
        json.dumps({"root": str(root), "questions": []}), encoding="utf-8"
    )
    called = []

    monkeypatch.setattr(run_bench, "ROOT", tmp_path)
    monkeypatch.setattr(run_bench, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(run_bench, "git_head", lambda _root: "test@123")
    monkeypatch.setattr(
        run_bench,
        "bench",
        lambda selected, *_args: (
            called.append(selected) or {"systems": {}, "questions": []}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bench.py",
            "--reuse-questions",
            str(report),
            "--yes",
            "--no-html",
        ],
    )

    run_bench.main()

    assert called == [root.resolve()]


def test_validate_reuse_questions_rejects_invalid_entries():
    assert run_bench.validate_reuse_questions([{"label": "definition"}])
    assert run_bench.validate_reuse_questions([{"query": "q", "label": None}])
    assert run_bench.validate_reuse_questions([{"query": "q"}]) is None
