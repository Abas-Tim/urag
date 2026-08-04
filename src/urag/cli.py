"""urag CLI: init, index, watch, search, get, status, doctor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import (
    UURAG_DIR,
    Config,
    discover_project_root,
    load_config,
    default_model_cache_dir,
)
from .db import Database
from .embed import Embedder, NoopEmbedder, create_embedder
from .indexer import Indexer
from .retrieve import Retriever
from .watcher import run_watch

app = typer.Typer(help="urag: structure-aware, token-efficient RAG for projects", no_args_is_help=True)
console = Console()

_embedder_cache: dict[str, Embedder] = {}


def _embedder(cfg: Config) -> Embedder:
    key = f"{cfg.embedding.provider}:{cfg.embedding.model}"
    if key in _embedder_cache:
        return _embedder_cache[key]
    try:
        emb = create_embedder(cfg.embedding)
    except Exception as exc:
        console.print(f"[yellow]warning: embedding unavailable ({exc}); using lexical-only mode[/yellow]")
        emb = NoopEmbedder()
    _embedder_cache[key] = emb
    return emb


def _engine(root: Path | None = None) -> tuple[Config, Database]:
    root = root or discover_project_root()
    cfg = load_config(root)
    if not cfg.db_path.exists():
        raise typer.BadParameter(
            f"no index found in {cfg.project_root}. Run: urag init"
        )
    return cfg, Database(cfg.db_path, cfg.embedding.dimension)


@app.command()
def init(
    root: Path = typer.Option(".", help="project root to index"),
    full: bool = typer.Option(False, "--full", help="also run a full index"),
    no_embed: bool = typer.Option(False, "--no-embed", help="skip model download / embedding"),
):
    """Set up .urag/ config and initial index for a project."""
    root = root.resolve()
    cfg = load_config(root)
    if not cfg.urag_dir.exists():
        cfg.urag_dir.mkdir(parents=True, exist_ok=True)
    gi = root / ".gitignore"
    if gi.exists():
        content = gi.read_text(encoding="utf-8", errors="replace")
        if UURAG_DIR not in content.splitlines():
            gi.write_text(content.rstrip() + f"\n{UURAG_DIR}/\n", encoding="utf-8")
            console.print(f"[green]added .urag/ to {gi}[/green]")
    else:
        gi.write_text(f"{UURAG_DIR}/\n", encoding="utf-8")
        console.print(f"[green]created {gi} with .urag/ entry[/green]")
    db = Database(cfg.db_path, cfg.embedding.dimension)
    console.print(f"[green]initialized {cfg.urag_dir}[/green]")
    console.print(f"config: {cfg.config_path}")
    if full:
        indexer = Indexer(cfg, db, _embedder(cfg), progress=lambda m: console.print(m))
        indexer.index_all()
    db.close()


@app.command()
def index(
    root: Path = typer.Option(".", help="project root"),
    no_embed: bool = typer.Option(False, "--no-embed", help="skip embedding new units"),
):
    """Incrementally index changed files (full pass on first run)."""
    cfg, db = _engine(root)
    embedder = NoopEmbedder() if no_embed else _embedder(cfg)
    indexer = Indexer(cfg, db, embedder, progress=lambda m: console.print(m))
    indexer.index_all()
    db.close()


@app.command()
def watch(
    root: Path = typer.Option(".", help="project root"),
    rescan_minutes: float = typer.Option(30, "--rescan", help="periodic full rescan interval (0 = off)"),
):
    """Continuously re-index on file changes."""
    cfg, db = _engine(root)
    indexer = Indexer(cfg, db, _embedder(cfg), progress=lambda m: console.print(m))
    run_watch(cfg, indexer, rescan_minutes=rescan_minutes)
    db.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="natural-language or symbol query"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid | lexical | dense"),
    language: Optional[str] = typer.Option(None, "--language", help="filter by language"),
    evidence: bool = typer.Option(False, "--evidence", help="include L2 source spans"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Search the project index."""
    cfg, db = _engine(root)
    try:
        from .git_aware import Git

        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).search(
            query, top_k=top_k, mode=mode, language=language
        )
        if json_out:
            payload = result.to_dict()
            if evidence:
                from .retrieve import fit_evidence

                for r in payload["results"]:
                    ev = db.load_evidence(r["id"])
                    r["evidence"] = fit_evidence(ev["span"], result.budget_tokens) if ev and "span" in ev else None
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        console.print(f"[dim]class={result.query_class} budget={result.budget_tokens} tok · {result.mode}[/dim]")
        if not result.results:
            console.print("[yellow]no results[/yellow]")
            return
        for r in result.results:
            u = r.unit
            loc = f"{r.file_path}:{u.start_line}-{u.end_line}"
            head = f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{loc}[/dim]"
            if r.stale:
                head += " [red][stale][/red]"
            console.print(head)
            if u.signature:
                console.print(f"  [cyan]{u.signature}[/cyan]")
            if u.summary:
                console.print(f"  {u.summary}")
            badges = []
            if r.lexical_rank:
                badges.append(f"lexical#{r.lexical_rank}")
            if r.dense_rank:
                badges.append(f"dense#{r.dense_rank}")
            if r.commit:
                badges.append(r.commit[:8])
            if badges:
                console.print(f"  [dim]{' '.join(badges)} · score {r.score:.3f}[/dim]")
            if r.caller_of:
                console.print(f"  [green]calls {r.caller_of} at line {r.call_line}[/green]")
            if evidence:
                from .retrieve import fit_evidence

                ev = db.load_evidence(u.id) if u.id else None
                if ev and "span" in ev:
                    console.print(f"  [dim]--- evidence (lines {ev['lines'][0]}-{ev['lines'][1]}, budget {result.budget_tokens} tok) ---[/dim]")
                    console.print(fit_evidence(ev["span"], result.budget_tokens))
    finally:
        db.close()


@app.command()
def callers(
    name: str = typer.Argument(..., help="symbol name to find callers of"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Who calls a symbol? Exact call-graph lookup."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).search_callers(name, limit=top_k or 20)
        if json_out:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        if not result.results:
            console.print(f"[yellow]no callers found for {name}[/yellow]")
            return
        for r in result.results:
            u = r.unit
            console.print(
                f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{r.file_path}:{u.start_line}-{u.end_line}[/dim]"
            )
            console.print(f"  [green]calls {r.caller_of} at line {r.call_line}[/green]")
    finally:
        db.close()


@app.command()
def classify(
    query: str = typer.Argument(..., help="query to classify"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Show the adaptive query classification (budget tier) for a query."""
    from .classify import BUDGETS, classify

    qc = classify(query)
    if json_out:
        print(json.dumps({"query": query, "class": qc, **BUDGETS[qc]}))
        return
    console.print(f"[bold]{query}[/bold] -> [green]{qc}[/green] "
                  f"(top_k={BUDGETS[qc]['top_k']}, budget={BUDGETS[qc]['tokens']} tokens)")


@app.command()
def get(
    unit_id: int = typer.Argument(..., help="unit id from search results"),
    root: Path = typer.Option(".", help="project root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Fetch the exact source span (L2 evidence) for a unit."""
    cfg, db = _engine(root)
    from .git_aware import Git

    ev = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).get(unit_id)
    db.close()
    if ev is None:
        console.print("[yellow]unit not found[/yellow]", file=sys.stderr)
        raise typer.Exit(1)
    if json_out:
        print(json.dumps(ev, ensure_ascii=False, indent=2))
        return
    head = f"[bold]{ev['file']}[/bold] lines {ev['lines'][0]}-{ev['lines'][1]}"
    if ev.get("commit"):
        head += f" [dim]@{ev['commit'][:8]}[/dim]"
    if ev.get("stale"):
        head += " [red][stale][/red]"
    console.print(head)
    console.print(ev["span"])


@app.command()
def mcp(
    root: Optional[Path] = typer.Option(None, "--root", help="project root (default: auto-discover)"),
):
    """Run the MCP server over stdio for agent harnesses."""
    from .mcp_server import serve

    serve(root)


@app.command()
def status(root: Path = typer.Option(".", help="project root")):
    """Show index stats."""
    cfg, db = _engine(root)
    s = db.stats()
    db.close()
    t = Table(title=f"urag index — {cfg.project_root}")
    t.add_column("metric")
    t.add_column("value")
    t.add_row("files", str(s.files))
    t.add_row("units", str(s.units))
    t.add_row("embedded", str(s.embedded))
    t.add_row("db size", f"{s.size_bytes / 1024:.0f} KiB")
    t.add_row("last indexed", s.last_indexed)
    t.add_row("embedding provider", cfg.embedding.provider)
    t.add_row("embedding model", cfg.embedding.model)
    for lang, count in sorted(s.by_language.items()):
        t.add_row(f"  {lang}", str(count))
    console.print(t)


@app.command()
def doctor(root: Path = typer.Option(".", help="project root")):
    """Check the installation and index health."""
    cfg, db = _engine(root)
    ok = True
    db.close()
    console.print(f"[bold]project:[/bold] {cfg.project_root}")
    console.print(f"[bold]index:[/bold] {cfg.db_path} [green]OK[/green]")
    if cfg.embedding.provider == "local":
        cache = default_model_cache_dir()
        console.print(f"[bold]model cache:[/bold] {cache}")
        try:
            emb = create_embedder(cfg.embedding)
            emb.embed_query("probe")
            console.print(f"[bold]embedding:[/bold] {cfg.embedding.model} [green]OK[/green] ({emb.dimension}d)")
        except Exception as exc:
            ok = False
            console.print(f"[bold]embedding:[/bold] [red]FAILED — {exc}[/red]")
    else:
        console.print(f"[bold]embedding:[/bold] {cfg.embedding.provider}")
    if not ok:
        raise typer.Exit(1)


def main() -> None:
    app()
