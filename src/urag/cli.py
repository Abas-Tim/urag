"""urag CLI: init, embed, index, watch, search, get, status, doctor."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Legacy Windows consoles default to cp1252; project docs can contain any
# unicode. Force UTF-8 before rich captures stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .config import (
    UURAG_DIR,
    Config,
    default_model_cache_dir,
    discover_project_root,
    ensure_gitignore,
    load_config,
)
from .db import Database
from .embed import Embedder, NoopEmbedder, create_embedder, purge_model_cache
from .indexer import Indexer
from .retrieve import Retriever
from .watcher import run_watch

app = typer.Typer(
    help="urag: structure-aware, token-efficient RAG for projects", no_args_is_help=True
)
console = Console()
error_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def _root_callback(
    version: bool = typer.Option(
        False, "--version", help="show version and exit", show_default=False
    ),
) -> None:
    if version:
        from . import __version__

        console.print(f"urag {__version__}")
        raise typer.Exit()


_embedder_cache: dict[str, Embedder] = {}


def _flush_progress(msg: str) -> None:
    """Print an index progress line and flush it immediately so agent
    harnesses see output while long runs are still going."""
    console.print(msg)
    with contextlib.suppress(Exception):
        sys.stdout.flush()


_XML_FAMILY_EXTS = (".xaml", ".axaml", ".xml", ".csproj", ".props", ".targets")


def _find_xml_family_files(cfg: Config, cap: int = 1000) -> int:
    """Count XML-family files under the project root, pruning excluded and
    hidden directories. Cheap existence check for the doctor hint."""
    skip = {
        ".git",
        ".hg",
        ".svn",
        ".urag",
        "node_modules",
        "target",
        "dist",
        "build",
        "obj",
        "bin",
        ".venv",
        "venv",
        "__pycache__",
    }
    found = 0
    for _dirpath, dirnames, filenames in os.walk(cfg.project_root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for name in filenames:
            if name.lower().endswith(_XML_FAMILY_EXTS):
                found += 1
                if found >= cap:
                    return found
    return found


def _embedder(cfg: Config) -> Embedder:
    key = f"{cfg.embedding.provider}:{cfg.embedding.model}:{cfg.embedding.dimension}"
    if key in _embedder_cache:
        return _embedder_cache[key]
    if cfg.embedding.provider == "local":
        _flush_progress(
            f"[dim]loading embedding model {cfg.embedding.model} "
            "(first run downloads it, may take a while)...[/dim]"
        )
    try:
        emb = create_embedder(cfg.embedding)
    except Exception as exc:
        error_console.print(
            f"[yellow]warning: embedding unavailable ({exc}); using lexical-only mode[/yellow]"
        )
        emb = NoopEmbedder()
    _embedder_cache[key] = emb
    return emb


def _engine(root: Path | None = None, migrate: bool = False) -> tuple[Config, Database]:
    root = root or discover_project_root()
    cfg = load_config(root)
    if not cfg.db_path.exists():
        raise typer.BadParameter(f"no index found in {cfg.project_root}. Run: urag init")
    try:
        return cfg, Database(cfg.db_path, cfg.embedding.dimension, migrate=migrate)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def init(
    root: Path = typer.Option(".", help="project root to index"),
    full: bool = typer.Option(False, "--full", help="also run a full index"),
    no_embed: bool = typer.Option(False, "--no-embed", help="skip model download / embedding"),
):
    """Set up .urag/ config and initial index for a project."""
    root = root.resolve()
    cfg = load_config(root, create=True)
    if not cfg.urag_dir.exists():
        cfg.urag_dir.mkdir(parents=True, exist_ok=True)
    gi = root / ".gitignore"
    gitignore_existed = gi.exists()
    if ensure_gitignore(root):
        if gitignore_existed:
            console.print(f"[green]added {UURAG_DIR}/ to {gi}[/green]")
        else:
            console.print(f"[green]created {gi} with {UURAG_DIR}/ entry[/green]")
    db = Database(cfg.db_path, cfg.embedding.dimension, migrate=True)
    console.print(f"[green]initialized {cfg.urag_dir}[/green]")
    console.print(f"config: {cfg.config_path}")
    if full:
        embedder = NoopEmbedder() if no_embed else _embedder(cfg)
        indexer = Indexer(cfg, db, embedder, progress=_flush_progress)
        _flush_progress(f"[dim]indexing {root}...[/dim]")
        indexer.index_all()
        console.print(
            "[dim]a first full index can take minutes (local CPU embeddings). "
            "Interrupted? Just run `urag index` again — it resumes where it "
            "stopped.[/dim]"
        )
    db.close()


def _detect_local_dimension(model: str) -> int | None:
    try:
        from fastembed import TextEmbedding

        return TextEmbedding.get_embedding_size(model)
    except Exception:
        return None


@app.command()
def embed(
    root: Path = typer.Option(".", help="project root"),
    model: Optional[str] = typer.Option(None, "--model", help="embedding model (local provider)"),
    provider: Optional[str] = typer.Option(None, "--provider", help="local | http | none"),
    dimension: Optional[int] = typer.Option(
        None, "--dimension", help="vector dimension (auto-detected for local models)"
    ),
    reindex: bool = typer.Option(False, "--reindex", help="re-embed all units after switching"),
    keep_cache: bool = typer.Option(
        False, "--keep-cache", help="keep the old model's files in the local cache"
    ),
):
    """Show or change the embedding model. Switching clears old embeddings."""
    root = root.resolve()
    cfg = load_config(root)
    emb = cfg.embedding

    if model is None and provider is None and dimension is None:
        console.print(f"[bold]embedding config[/bold] ({cfg.config_path})")
        console.print(f"  provider:  {emb.provider}")
        console.print(f"  model:     {emb.model}")
        console.print(f"  dimension: {emb.dimension}")
        if emb.provider == "local":
            console.print(f"  cache:     {default_model_cache_dir()}")
        if cfg.db_path.exists():
            try:
                db = Database(cfg.db_path, emb.dimension)
            except RuntimeError as exc:
                console.print(f"  [yellow]index: {exc}[/yellow]")
            else:
                s = db.stats()
                db.close()
                console.print(f"  embedded:  {s.embedded}/{s.units} units")
        return

    old_provider, old_model, old_dim = emb.provider, emb.model, emb.dimension
    new_provider = provider or old_provider
    if new_provider not in ("local", "http", "none"):
        raise typer.BadParameter(f"provider must be local, http, or none (got {new_provider!r})")
    new_model = model or old_model

    if new_provider == "local":
        detected = _detect_local_dimension(new_model)
        if detected is None:
            if dimension is None:
                raise typer.BadParameter(
                    f"could not detect the vector size of {new_model!r}; pass --dimension"
                )
            new_dim = dimension
        else:
            if dimension is not None and dimension != detected:
                raise typer.BadParameter(
                    f"{new_model!r} produces {detected}-dimensional vectors, not {dimension}"
                )
            new_dim = detected
    elif new_provider == "http":
        new_dim = dimension or old_dim
        if new_dim <= 0:
            raise typer.BadParameter("http provider needs --dimension > 0")
    else:
        new_dim = old_dim

    changed = new_provider != old_provider or new_model != old_model or new_dim != old_dim

    if changed and cfg.db_path.exists():
        db = Database(cfg.db_path, old_dim if old_dim > 0 else new_dim)
        db.clear_embeddings()
        db.delete_meta("embedding_fingerprint")
        db.close()
        console.print("[yellow]cleared old embeddings[/yellow]")

    if (
        old_provider == "local"
        and (new_provider != "local" or new_model != old_model)
        and not keep_cache
    ):
        if new_provider == "local":
            try:
                create_embedder(
                    replace(
                        cfg.embedding,
                        provider=new_provider,
                        model=new_model,
                        dimension=new_dim,
                    )
                )
            except Exception as exc:
                raise typer.BadParameter(f"new model {new_model!r} failed to load: {exc}") from exc
        if purge_model_cache(old_model):
            console.print(f"[yellow]removed {old_model} from the local model cache[/yellow]")

    emb.provider = new_provider
    emb.model = new_model
    emb.dimension = new_dim
    cfg.save()

    if not changed:
        console.print("[green]embedding config unchanged[/green]")
    else:
        console.print(f"[green]switched to {new_provider}:{new_model} ({new_dim}d)[/green]")
        if not reindex and new_provider != "none" and cfg.db_path.exists():
            console.print("[yellow]run `urag index` to re-embed units with the new model[/yellow]")

    if reindex and new_provider != "none" and cfg.db_path.exists():
        db = Database(cfg.db_path, new_dim, migrate=True)
        indexer = Indexer(cfg, db, _embedder(cfg), progress=lambda m: console.print(m))
        indexer.index_all()
        db.close()


@app.command()
def index(
    root: Path = typer.Option(".", help="project root"),
    no_embed: bool = typer.Option(False, "--no-embed", help="skip embedding new units"),
):
    """Incrementally index changed files (full pass on first run)."""
    cfg, db = _engine(root, migrate=True)
    embedder = NoopEmbedder() if no_embed else _embedder(cfg)
    indexer = Indexer(cfg, db, embedder, progress=_flush_progress)
    _flush_progress(f"[dim]indexing {cfg.project_root}...[/dim]")
    indexer.index_all()
    db.close()


@app.command()
def watch(
    root: Path = typer.Option(".", help="project root"),
    rescan_minutes: float = typer.Option(
        30, "--rescan", help="periodic full rescan interval (0 = off)"
    ),
):
    """Continuously re-index on file changes."""
    cfg, db = _engine(root, migrate=True)
    indexer = Indexer(cfg, db, _embedder(cfg), progress=_flush_progress)
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
                    r["evidence"] = (
                        fit_evidence(ev["span"], result.budget_tokens)
                        if ev and "span" in ev
                        else None
                    )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        console.print(
            f"[dim]class={result.query_class} budget={result.budget_tokens} tok · {result.mode}[/dim]"
        )
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
                    console.print(
                        f"  [dim]--- evidence (lines {ev['lines'][0]}-{ev['lines'][1]}, budget {result.budget_tokens} tok) ---[/dim]"
                    )
                    console.print(fit_evidence(ev["span"], result.budget_tokens))
    finally:
        db.close()


@app.command("eval")
def eval_cmd(
    root: Path = typer.Option(".", help="project root"),
    questions: Optional[Path] = typer.Option(
        None, "--questions", help="JSONL of {query, gold_file?, gold_unit_ids?}"
    ),
    autogen: Optional[int] = typer.Option(
        None,
        "--autogen",
        help="auto-generate N definition + N call questions with provable gold",
    ),
    transitive: Optional[int] = typer.Option(
        None,
        "--transitive",
        help="add N multi-hop (callers-of-callers) questions with provable gold",
    ),
    alias: Optional[int] = typer.Option(
        None,
        "--alias",
        help="add N import-alias resolution questions with provable gold",
    ),
    reference: Optional[int] = typer.Option(
        None,
        "--reference",
        help="add N reference questions with provable gold",
    ),
    top_k: int = typer.Option(5, "--top-k", help="recall@k cutoff"),
    systems: Optional[str] = typer.Option(
        None,
        "--systems",
        help="comma list: urag-auto,urag-hybrid,urag-lexical,rg,chunk",
    ),
    judge_url: Optional[str] = typer.Option(
        None, "--judge-url", help="OpenAI-compatible chat endpoint for the judge tier"
    ),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="judge model name"),
    judge_key: Optional[str] = typer.Option(
        None, "--judge-key", help="judge API key (prefer URAG_JUDGE_KEY)"
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable report"),
    report: Optional[Path] = typer.Option(
        None, "--report", help="write full report JSON to this file"
    ),
    reresolve: bool = typer.Option(
        False,
        "--reresolve",
        help="re-derive gold from the current index for --questions files "
        "taken from an older report (cross-branch comparison)",
    ),
):
    """Compare urag retrieval vs grep / chunk-RAG / whole-file baselines."""
    from .eval import run_eval

    cfg, db = _engine(root)
    try:
        run_eval(
            cfg,
            db,
            _embedder(cfg),
            questions=questions,
            autogen=autogen,
            transitive=transitive,
            alias=alias,
            reference=reference,
            top_k=top_k,
            systems=systems,
            judge_url=judge_url,
            judge_model=judge_model,
            judge_key=judge_key,
            json_out=json_out,
            report=report,
            reresolve=reresolve,
            console=console,
        )
    finally:
        db.close()


@app.command()
def callers(
    name: str = typer.Argument(..., help="symbol name to find callers of"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    depth: int = typer.Option(
        1,
        "--depth",
        min=1,
        help="hop depth; 1 = direct callers, >1 = callers-of-callers",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Who calls a symbol? Exact call-graph lookup (--depth for multi-hop)."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        retriever = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root))
        if depth > 1:
            result = retriever.search_transitive(name, depth=depth, limit=top_k or 20)
        else:
            result = retriever.search_callers(name, limit=top_k or 20)
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
            if r.hop > 1:
                console.print(f"  [cyan]hop {r.hop}[/cyan]")
            if r.resolved_target:
                console.print(f"  [cyan]via alias -> {r.resolved_target}[/cyan]")
    finally:
        db.close()


@app.command()
def references(
    name: str = typer.Argument(..., help="symbol name to find references of"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    depth: int = typer.Option(
        1,
        "--depth",
        min=1,
        help="hop depth; 1 = direct referencers, >1 = referencers-of-referencers",
    ),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Who references a symbol? Type mentions, constructions, bases, XAML."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        retriever = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root))
        if depth > 1:
            result = retriever.search_transitive_references(name, depth=depth, limit=top_k or 30)
        else:
            result = retriever.search_references(name, limit=top_k or 30)
        if json_out:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        if not result.results:
            console.print(f"[yellow]no references found for {name}[/yellow]")
            return
        for r in result.results:
            u = r.unit
            console.print(
                f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{r.file_path}:{u.start_line}-{u.end_line}[/dim]"
            )
            console.print(
                f"  [green]references {r.caller_of} at line {r.call_line} ({r.ref_kind})[/green]"
            )
            if r.hop > 1:
                console.print(f"  [cyan]hop {r.hop}[/cyan]")
            if r.resolved_target:
                console.print(f"  [cyan]via alias -> {r.resolved_target}[/cyan]")
    finally:
        db.close()


@app.command()
def deadcode(
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    language: Optional[str] = typer.Option(None, "--language", help="filter by language"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """List candidate dead symbols (no incoming calls or references)."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).unreferenced(
            limit=top_k or 50, language=language
        )
        if json_out:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        if not result.results:
            console.print("[green]no unreferenced symbols found[/green]")
            return
        console.print("[dim]heuristic candidates only — verify with git grep before removing[/dim]")
        for r in result.results:
            u = r.unit
            console.print(
                f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{r.file_path}:{u.start_line}-{u.end_line}[/dim]"
            )
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
    console.print(
        f"[bold]{query}[/bold] -> [green]{qc}[/green] "
        f"(top_k={BUDGETS[qc]['top_k']}, budget={BUDGETS[qc]['tokens']} tokens)"
    )


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
        error_console.print("[yellow]unit not found[/yellow]")
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
    root: Optional[Path] = typer.Option(
        None, "--root", help="project root (default: auto-discover)"
    ),
):
    """Run the MCP server over stdio for agent harnesses."""
    from .mcp_server import serve

    serve(root)


@app.command()
def resolve(
    name: str = typer.Argument(..., help="symbol name or qualified name"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Find an exact symbol definition by name."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).resolve(
            name, limit=top_k or 10
        )
        if json_out:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        if not result.results:
            console.print(f"[yellow]no definition found for {name}[/yellow]")
            return
        for r in result.results:
            u = r.unit
            head = f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{r.file_path}:{u.start_line}-{u.end_line}[/dim]"
            if r.stale:
                head += " [red][stale][/red]"
            console.print(head)
            if u.signature:
                console.print(f"  [cyan]{u.signature}[/cyan]")
            if u.summary:
                console.print(f"  {u.summary}")
    finally:
        db.close()


@app.command()
def callees(
    unit_id: int = typer.Argument(..., help="unit id from search results"),
    root: Path = typer.Option(".", help="project root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """List what a unit calls (its call sites)."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).callees(unit_id)
        if result is None:
            error_console.print("[yellow]unit not found[/yellow]")
            raise typer.Exit(1)
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        console.print(
            f"[bold]{result['qualname'] or result['name']}[/bold] [dim]{result['file']}[/dim]"
        )
        if not result["callees"]:
            console.print("[yellow]no call sites found[/yellow]")
            return
        for c in result["callees"]:
            console.print(
                f"  calls [green]{c['callee_full'] or c['callee']}[/green] at line {c['line']}"
            )
    finally:
        db.close()


@app.command()
def dependents(
    target: str = typer.Argument(..., help="module or symbol to find importers of"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Find what imports (depends on) a module or symbol."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).dependents(
            target, limit=top_k or 50
        )
        result["count"] = len(result["results"])
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        console.print(f"[bold]{target}[/bold]: {len(result['results'])} dependent(s)")
        for r in result["results"]:
            line = f"  {r['path']}"
            if r.get("alias"):
                line += f" [dim]as {r['alias']}[/dim]"
            console.print(line)
    finally:
        db.close()


@app.command("symbols")
def symbols_cmd(
    file: str = typer.Argument(..., help="project-relative file path"),
    root: Path = typer.Option(".", help="project root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """List every indexed unit in a file."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).list_symbols(file)
        if json_out:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return
        if not result.results:
            console.print(f"[yellow]no units indexed for {file}[/yellow]")
            return
        for r in result.results:
            u = r.unit
            console.print(
                f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{u.start_line}-{u.end_line}[/dim]"
            )
            if u.signature:
                console.print(f"  [cyan]{u.signature}[/cyan]")
    finally:
        db.close()


@app.command("read")
def read_cmd(
    path: str = typer.Argument(..., help="project-relative file path"),
    root: Path = typer.Option(".", help="project root"),
    start: Optional[int] = typer.Argument(
        None, help="first line (1-based); also available as --start"
    ),
    end: Optional[int] = typer.Argument(
        None, help="last line (inclusive); also available as --end"
    ),
    start_opt: Optional[int] = typer.Option(None, "--start", help="first line (1-based)"),
    end_opt: Optional[int] = typer.Option(None, "--end", help="last line (inclusive)"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Read a file (or a line range) from the project."""
    from .git_aware import Git

    cfg, db = _engine(root)
    try:
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).read_file(
            path,
            start=start_opt if start_opt is not None else start,
            end=end_opt if end_opt is not None else end,
        )
        if json_out:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if "error" in result:
                raise typer.Exit(1)
            return
        if "error" in result:
            error_console.print(f"[yellow]{result['error']}[/yellow]")
            raise typer.Exit(1)
        console.print(
            f"[bold]{result['path']}[/bold] [dim]lines {result['start_line']}-{result['end_line']} of {result['total_lines']}[/dim]"
        )
        console.print(result["span"])
    finally:
        db.close()


@app.command("recent")
def recent_cmd(
    root: Path = typer.Option(".", help="project root"),
    limit: int = typer.Option(20, "--limit", help="number of commits"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Show recent git changes (branch, working tree, recent commits)."""
    from .git_aware import Git

    cfg, db = _engine(root)
    db.close()
    git = Git(cfg.project_root)
    result = git.recent_changes(limit=limit)
    if json_out:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold]branch:[/bold] {result['branch']} [dim]@{result['head'][:8] if result['head'] else ''}[/dim]"
    )
    w = result["working"]
    if w["changed"] or w["deleted"] or w["untracked"]:
        console.print("[bold]working changes:[/bold]")
        for p in w["changed"]:
            console.print(f"  [yellow]M[/yellow] {p}")
        for p in w["deleted"]:
            console.print(f"  [red]D[/red] {p}")
        for p in w["untracked"]:
            console.print(f"  [green]?[/green] {p}")
    console.print("[bold]recent commits:[/bold]")
    for c in result["commits"]:
        console.print(
            f"  [cyan]{c['short']}[/cyan] {c['subject']} [dim]({len(c['files'])} files)[/dim]"
        )


@app.command()
def status(
    root: Path = typer.Option(".", help="project root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Show index stats."""
    cfg, db = _engine(root)
    s = db.stats()
    db.close()
    if json_out:
        payload = {
            "root": str(cfg.project_root.resolve()),
            "files": s.files,
            "units": s.units,
            "embedded": s.embedded,
            "db_size_kib": round(s.size_bytes / 1024),
            "last_indexed": s.last_indexed,
            "embedding": {
                "provider": cfg.embedding.provider,
                "model": cfg.embedding.model,
                "dimension": cfg.embedding.dimension,
            },
            "by_language": s.by_language,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
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
def doctor(
    root: Path = typer.Option(".", help="project root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
):
    """Check the installation and index health."""
    cfg, db = _engine(root)
    ok = True
    db.close()
    hints: list[str] = []
    if "xml" not in cfg.index.languages:
        xml_hits = _find_xml_family_files(cfg)
        if xml_hits:
            hint = (
                f"{xml_hits} XML-family file(s) found but 'xml' is not in "
                "index.languages (config predates XAML support); add it to "
                ".urag/urag.toml and re-run `urag index` to close the markup blind spot"
            )
            hints.append(hint)
    embedding: dict = {"provider": cfg.embedding.provider}
    if cfg.embedding.provider == "local":
        cache = default_model_cache_dir()
        embedding["model"] = cfg.embedding.model
        embedding["cache"] = str(cache)
        try:
            emb = create_embedder(cfg.embedding)
            emb.embed_query("probe")
            embedding["status"] = "ok"
            embedding["dimension"] = emb.dimension
        except Exception as exc:
            ok = False
            embedding["status"] = "failed"
            embedding["error"] = str(exc)
    if json_out:
        payload = {
            "ok": ok,
            "root": str(cfg.project_root.resolve()),
            "index": str(cfg.db_path),
            "embedding": embedding,
            "hints": hints,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not ok:
            raise typer.Exit(1)
        return
    console.print(f"[bold]project:[/bold] {cfg.project_root}")
    console.print(f"[bold]index:[/bold] {cfg.db_path} [green]OK[/green]")
    for hint in hints:
        console.print(f"[yellow]hint: {hint}[/yellow]")
    if cfg.embedding.provider == "local":
        console.print(f"[bold]model cache:[/bold] {embedding.get('cache', '')}")
        if embedding.get("status") == "ok":
            console.print(
                f"[bold]embedding:[/bold] {cfg.embedding.model} [green]OK[/green] ({embedding['dimension']}d)"
            )
        else:
            console.print(
                f"[bold]embedding:[/bold] [red]FAILED — {embedding.get('error', '')}[/red]"
            )
    else:
        console.print(f"[bold]embedding:[/bold] {cfg.embedding.provider}")
    if not ok:
        raise typer.Exit(1)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app()
