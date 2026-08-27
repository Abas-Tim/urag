"""Tests for reference collection, XAML extraction, and dead-code queries."""

from pathlib import Path

import pytest

from urag.config import load_config
from urag.db import Database
from urag.embed import NoopEmbedder
from urag.indexer import Indexer
from urag.retrieve import Retriever
from urag.extractors.native_ext import CSharpExtractor, JavaExtractor
from urag.extractors.python_ext import PythonExtractor
from urag.extractors.ts_ext import TsExtractor
from urag.extractors.xml_ext import XmlExtractor


# ---------------------------------------------------------------------------
# reference extraction
# ---------------------------------------------------------------------------


def test_csharp_references():
    src = """namespace App {
public class MainWindow : Window {
    private BoolToVisConverter _c = new BoolToVisConverter();
    public MainWindow(Service s) { }
    public static MainWindow Create() => null;
    public void Run(Window win) {
        Window x = win as MainWindow;
        var list = new List<Window>();
        typeof(MainWindow);
        if (win is MainWindow m) { }
    }
    [Deprecated]
    public int Id { get; set; }
}
}
"""
    refs = CSharpExtractor().collect_references(src)
    kinds = {(r.target, r.kind) for r in refs}
    assert ("MainWindow", "base") not in kinds
    assert ("Window", "base") in kinds
    assert ("BoolToVisConverter", "type") in kinds
    assert ("BoolToVisConverter", "construct") in kinds
    assert ("Service", "type") in kinds
    assert ("Window", "type") in kinds
    assert ("MainWindow", "cast") in kinds
    assert ("List", "construct") in kinds
    assert ("Window", "generic") in kinds
    assert ("MainWindow", "cast") in kinds
    assert ("Deprecated", "attribute") in kinds
    assert ("int", "type") not in kinds
    assert ("var", "type") not in kinds


def test_csharp_constructor_calls():
    src = """class App {
    void Run() {
        var w = new MainWindow();
    }
}
"""
    calls = CSharpExtractor().collect_calls(src)
    assert {("MainWindow", "MainWindow")} == {(c.callee, c.callee_full) for c in calls}


def test_java_constructor_calls_and_refs():
    src = """public class A {
    void run() {
        B helper = new B();
        helper.process("x");
    }
}
"""
    calls = JavaExtractor().collect_calls(src)
    assert ("B", "B") in {(c.callee, c.callee_full) for c in calls}
    refs = JavaExtractor().collect_references(src)
    kinds = {(r.target, r.kind) for r in refs}
    assert ("B", "construct") in kinds
    assert ("B", "type") in kinds


def test_python_references():
    src = """import os

class Auth(Base, os.PathLike):
    def validate(self, token: Token) -> bool:
        if isinstance(token, Token):
            return True
        t: Token = token
        return self.check(token)
"""
    refs = PythonExtractor().collect_references(src)
    kinds = {(r.target, r.kind) for r in refs}
    assert ("Base", "base") in kinds
    assert ("PathLike", "base") in kinds
    assert ("Token", "type") in kinds
    assert ("Token", "cast") in kinds
    assert ("bool", "type") not in kinds


def test_ts_references():
    src = """import { Token } from './tok'

class Auth extends Base implements Ix {
    private t: Token;
    constructor(svc: Service) { this.svc = new Service(); }
    run(): MainWindow {
        const w = new MainWindow();
        return w;
    }
}
"""
    refs = TsExtractor("typescript").collect_references(src)
    kinds = {(r.target, r.kind) for r in refs}
    assert ("Base", "base") in kinds
    assert ("Ix", "base") in kinds
    assert ("Token", "type") in kinds
    assert ("Service", "type") in kinds
    assert ("Service", "construct") in kinds
    assert ("MainWindow", "construct") in kinds
    calls = TsExtractor("typescript").collect_calls(src)
    assert ("MainWindow", "MainWindow") in {(c.callee, c.callee_full) for c in calls}


# ---------------------------------------------------------------------------
# XAML extraction
# ---------------------------------------------------------------------------

XAML = """<Window xmlns="https://github.com/avaloniaui"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        x:Class="MyApp.MainWindow"
        Loaded="OnLoaded">
  <Window.Resources>
    <BoolToVisibilityConverter x:Key="BoolToVis" />
  </Window.Resources>
  <StackPanel>
    <TextBlock Text="{Binding Name}" />
    <ItemsControl Items="{Binding Items}">
      <ItemsControl.ItemTemplate>
        <DataTemplate DataType="Item">
          <Button Content="{Binding}" Command="{x:Static MyApp.MainWindow.OpenCommand}" />
        </DataTemplate>
      </ItemsControl.ItemTemplate>
    </ItemsControl>
  </StackPanel>
</Window>
"""


def test_xaml_units():
    units = XmlExtractor().extract(XAML, "MainWindow.axaml")
    by_type = {(u.unit_type, u.name) for u in units}
    assert ("file", "MainWindow.axaml") in by_type
    assert ("class", "MainWindow") in by_type
    assert ("event", "OnLoaded") in by_type
    assert ("resource", "BoolToVis") in by_type
    assert ("template", "Item") in by_type


def test_xaml_references():
    refs = XmlExtractor().collect_references(XAML)
    kinds = {(r.target, r.kind) for r in refs}
    assert ("MainWindow", "xaml_type") in kinds
    assert ("OnLoaded", "xaml_event") in kinds
    assert ("BoolToVisibilityConverter", "xaml_type") in kinds
    assert ("Item", "xaml_type") in kinds
    assert ("OpenCommand", "xaml_member") in kinds
    assert ("StackPanel", "xaml_type") not in kinds


def test_xaml_attached_property_reference():
    src = (
        '<Window xmlns:b="urn:behaviors">\n'
        '  <Grid b:DragDropBehavior.EnableDragDrop="True" DragDrop.AllowDrop="True">\n'
        "  </Grid>\n"
        "</Window>\n"
    )
    refs = XmlExtractor().collect_references(src)
    kinds = {(r.target, r.kind) for r in refs}
    assert ("DragDropBehavior", "xaml_member") in kinds
    assert ("DragDrop", "xaml_member") not in kinds


# ---------------------------------------------------------------------------
# end-to-end index: references, callers-via-construction, deadcode
# ---------------------------------------------------------------------------

CS_APP = """namespace App {
public class MainWindow {
    private readonly BoolToVisConverter _c = new BoolToVisConverter();
    public void Show() { OnLoaded(); }
    private void OnLoaded() { }
}
public class LegacyWidget {
    public void Draw() { }
}
}
"""

CS_PROGRAM = """namespace App {
public class Boot {
    public static void Main() {
        var w = new MainWindow();
        w.Show();
    }
}
}
"""

XAML_WINDOW = """<Window xmlns="https://github.com/avaloniaui"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        x:Class="App.MainWindow"
        Loaded="OnLoaded">
  <Window.Resources>
    <BoolToVisConverter x:Key="BoolToVis" />
  </Window.Resources>
</Window>
"""


@pytest.fixture
def refdb(tmp_path: Path):
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    for path, source in (
        ("MainWindow.cs", CS_APP),
        ("Program.cs", CS_PROGRAM),
        ("MainWindow.axaml", XAML_WINDOW),
    ):
        (tmp_path / path).write_text(source, encoding="utf-8")
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        yield db, cfg
    finally:
        db.close()


def _names(db, hits):
    return {db.unit_by_id(h["unit"].id)[0].name for h in hits}


def test_callers_finds_constructor(refdb):
    db, cfg = refdb
    hits = db.callers("MainWindow")
    names = _names(db, hits)
    assert "Main" in names


def test_references_finds_construction_and_xaml(refdb):
    db, cfg = refdb
    hits = db.references("MainWindow")
    by_file = {h["path"] for h in hits}
    assert "Program.cs" in by_file
    assert "MainWindow.axaml" in by_file


def test_references_finds_xaml_resource_usage(refdb):
    db, cfg = refdb
    hits = db.references("BoolToVisConverter")
    by_file = {h["path"] for h in hits}
    assert "MainWindow.cs" in by_file
    assert "MainWindow.axaml" in by_file


def test_deadcode_excludes_referenced_symbols(refdb):
    db, cfg = refdb
    rows = db.unreferenced_symbols(limit=100)
    names = {r["unit"].name for r in rows}
    assert "MainWindow" not in names
    assert "OnLoaded" not in names
    assert "BoolToVisConverter" not in names
    assert "LegacyWidget" in names
    assert "Draw" in names


def test_retriever_search_references(refdb):
    db, cfg = refdb
    r = Retriever(cfg, db, NoopEmbedder())
    result = r.search_references("MainWindow")
    assert result.mode == "references"
    files = {res.file_path for res in result.results}
    assert "Program.cs" in files
    assert "MainWindow.axaml" in files


def test_impact_query_routes_reference_intent(refdb):
    db, cfg = refdb
    r = Retriever(cfg, db, NoopEmbedder())
    result = r.search("what references MainWindow")
    files = {res.file_path for res in result.results}
    assert "Program.cs" in files
    assert "MainWindow.axaml" in files


def test_transitive_references(refdb):
    db, cfg = refdb
    # BoolToVisConverter is referenced from MainWindow.cs (field) and axaml;
    # MainWindow is referenced from Program.cs + axaml. Depth-2 from the
    # converter should reach Program.cs via MainWindow.
    rows = db.transitive_references("BoolToVisConverter", max_depth=2)
    files = {r["path"] for r in rows}
    assert "MainWindow.cs" in files


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


def test_mcp_references_and_dead_symbols(refdb):
    import asyncio
    import json

    from urag.mcp_server import create_server

    db, cfg = refdb
    server = create_server(cfg.project_root)

    def call(name, arguments):
        result = asyncio.run(server.call_tool(name, arguments))
        return json.loads(result.content[0].text)

    refs = call("references", {"name": "MainWindow"})
    files = {r["file"] for r in refs["results"]}
    assert "Program.cs" in files
    assert "MainWindow.axaml" in files
    assert any(r.get("ref_kind") == "construct" for r in refs["results"])

    dead = call("dead_symbols", {"limit": 100})
    names = {r["name"] for r in dead["results"]}
    assert "LegacyWidget" in names
    assert "MainWindow" not in names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_read_positional_range(tmp_path):
    import urag.cli as cli
    from typer.testing import CliRunner

    cfg = load_config(tmp_path)
    cfg.embedding.provider = "none"
    cfg.save()
    db = Database(cfg.db_path, cfg.embedding.dimension)
    db.close()
    (tmp_path / "doc.md").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli.app, ["read", "doc.md", "2", "3", "--root", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "two\nthree" in result.output
    assert "four" not in result.output


def test_old_index_migrates_ref_edges(tmp_path):
    """Indexes built before ref_edges get reference edges on the next
    `index` run without any file changes."""
    cfg = load_config(tmp_path)
    db = Database(cfg.db_path, cfg.embedding.dimension)
    (tmp_path / "mvc.cs").write_text(
        "class A { void Run() { var b = new B(); } }\nclass B {}\n",
        encoding="utf-8",
    )
    Indexer(cfg, db, NoopEmbedder()).index_all()
    db.conn.execute("DELETE FROM ref_edges")
    db.delete_meta("ref_edges_v1")
    db.conn.commit()
    db.close()

    db = Database(cfg.db_path, cfg.embedding.dimension)
    try:
        Indexer(cfg, db, NoopEmbedder()).index_all()
        rows = db.conn.execute("SELECT ref FROM ref_edges").fetchall()
        refs = {r["ref"] for r in rows}
        assert "B" in refs
        assert db.get_meta("refs_pending") == ""
    finally:
        db.close()
