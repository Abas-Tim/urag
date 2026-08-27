"""urag CLI: init, embed, index, watch, search, get, status, doctor."""

from __future__ import annotations

import json
import os
import sys
import time
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
    discover_project_root,
    ensure_gitignore,
    load_config,
    default_model_cache_dir,
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
    try:
        sys.stdout.flush()
    except Exception:
        pass


_XML_FAMILY_EXTS = (".xaml", ".axaml", ".xml", ".csproj", ".props", ".targets")


def _find_xml_family_files(cfg: Config, cap: int = 1000) -> int:
    """Count XML-family files under the project root, pruning excluded and
    hidden directories. Cheap existence check for the doctor hint."""
    import os

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
    for dirpath, dirnames, filenames in os.walk(cfg.project_root):
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
    no_embed: bool = typer.Option(
        False, "--no-embed", help="skip model download / embedding"
    ),
):
    """Set up .urag/ config and initial index for a project."""
    root = root.resolve()
    cfg = load_config(root)
    if not cfg.urag_dir.exists():
        cfg.urag_dir.mkdir(parents=True, exist_ok=True)
    gi = root / ".gitignore"
    gitignore_existed = gi.exists()
    if ensure_gitignore(root):
        if gitignore_existed:
            console.print(f"[green]added {UURAG_DIR}/ to {gi}[/green]")
        else:
            console.print(f"[green]created {gi} with {UURAG_DIR}/ entry[/green]")
    db = Database(cfg.db_path, cfg.embedding.dimension)
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
    model: Optional[str] = typer.Option(
        None, "--model", help="embedding model (local provider)"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="local | http | none"
    ),
    dimension: Optional[int] = typer.Option(
        None, "--dimension", help="vector dimension (auto-detected for local models)"
    ),
    reindex: bool = typer.Option(
        False, "--reindex", help="re-embed all units after switching"
    ),
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
            db = Database(cfg.db_path, emb.dimension)
            s = db.stats()
            db.close()
            console.print(f"  embedded:  {s.embedded}/{s.units} units")
        return

    old_provider, old_model, old_dim = emb.provider, emb.model, emb.dimension
    new_provider = provider or old_provider
    if new_provider not in ("local", "http", "none"):
        raise typer.BadParameter(
            f"provider must be local, http, or none (got {new_provider!r})"
        )
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
                    f"{new_model!r} produces {detected}-dimensional vectors, "
                    f"not {dimension}"
                )
            new_dim = detected
    elif new_provider == "http":
        new_dim = dimension or old_dim
        if new_dim <= 0:
            raise typer.BadParameter("http provider needs --dimension > 0")
    else:
        new_dim = old_dim

    changed = (
        new_provider != old_provider or new_model != old_model or new_dim != old_dim
    )

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
        if purge_model_cache(old_model):
            console.print(
                f"[yellow]removed {old_model} from the local model cache[/yellow]"
            )

    emb.provider = new_provider
    emb.model = new_model
    emb.dimension = new_dim
    cfg.save()

    if not changed:
        console.print("[green]embedding config unchanged[/green]")
    else:
        console.print(
            f"[green]switched to {new_provider}:{new_model} ({new_dim}d)[/green]"
        )
        if not reindex and new_provider != "none" and cfg.db_path.exists():
            console.print(
                "[yellow]run `urag index` to re-embed units with the new model[/yellow]"
            )

    if reindex and new_provider != "none" and cfg.db_path.exists():
        db = Database(cfg.db_path, new_dim)
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
    cfg, db = _engine(root)
    indexer = Indexer(cfg, db, _embedder(cfg), progress=_flush_progress)
    run_watch(cfg, indexer, rescan_minutes=rescan_minutes)
    db.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="natural-language or symbol query"),
    root: Path = typer.Option(".", help="project root"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    mode: str = typer.Option("hybrid", "--mode", help="hybrid | lexical | dense"),
    language: Optional[str] = typer.Option(
        None, "--language", help="filter by language"
    ),
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
            head = (
                f"[bold]{u.qualname or u.name}[/bold] ({u.unit_type}) [dim]{loc}[/dim]"
            )
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
                console.print(
                    f"  [green]calls {r.caller_of} at line {r.call_line}[/green]"
                )
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
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="judge model name"
    ),
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
    from .eval import (
        EVAL_SCHEMA_VERSION,
        ChunkBaseline,
        Hit,
        OracleBaseline,
        ReadBaseline,
        RgBaseline,
        SystemRun,
        _metrics,
        _tokens,
        _unit_tokens,
        aggregate,
        autogen_alias_questions,
        autogen_questions,
        autogen_reference_questions,
        autogen_transitive_questions,
        judge_results,
        load_questions,
        reresolve_questions,
        resolve_question,
    )
    from .git_aware import Git

    cfg, db = _engine(root)
    embedder = _embedder(cfg)
    try:
        if questions:
            qs = load_questions(questions)
            if reresolve:
                before = len(qs)
                qs = reresolve_questions(db, qs)
                console.print(
                    f"[dim]reresolved {len(qs)}/{before} questions against the current index[/dim]"
                )
        elif autogen:
            qs = autogen_questions(db, autogen)
        else:
            qs = autogen_questions(db, 10)
        if transitive:
            qs += autogen_transitive_questions(db, transitive)
        if alias:
            qs += autogen_alias_questions(db, cfg.project_root, alias)
        if reference:
            qs += autogen_reference_questions(db, reference)
        qs = [resolve_question(db, q) for q in qs]

        chosen = (systems or "urag-auto,urag-hybrid,urag-lexical,rg,chunk").split(",")
        retriever = Retriever(cfg, db, embedder, Git(cfg.project_root))
        rg = RgBaseline(cfg.project_root)
        read = ReadBaseline(cfg.project_root) if "read" in chosen else None
        chunk = ChunkBaseline(cfg, db, embedder) if "chunk" in chosen else None
        oracle = OracleBaseline(cfg.project_root)

        if any(name in chosen for name in ("urag-auto", "urag-hybrid", "urag-lexical")):
            retriever.search("__warmup__", top_k=top_k)

        def urag_run(result) -> SystemRun:
            hits = []
            total = 0
            for x in result.results:
                ev = db.load_evidence(x.unit.id)
                span = (ev or {}).get("span", "")
                toks = _tokens(span) if span else _unit_tokens(x.unit)
                hits.append(
                    Hit(
                        x.file_path,
                        x.unit.id,
                        toks,
                        detail=span,
                        title=(x.unit.signature or x.unit.name),
                    )
                )
                total += toks
            return SystemRun(result.mode, hits, 0.0, total)

        rows: dict[str, list] = {s: [None] * len(qs) for s in chosen}
        runs_by_system: dict[str, dict[int, SystemRun]] = {}
        console.print(
            f"[dim]evaluating {len(qs)} questions (top_k={top_k}) across {len(chosen)} systems[/dim]"
        )

        def safe_run(name: str, fn) -> SystemRun:
            t = time.perf_counter()
            try:
                run = fn()
            except Exception as exc:  # one failing system must not kill the eval
                console.print(f"[yellow]{name}: query failed: {exc}[/yellow]")
                return SystemRun(name, [], 0.0, 0)
            if run.seconds == 0.0:
                run.seconds = time.perf_counter() - t
            return run

        for i, q in enumerate(qs):
            runs: dict[str, SystemRun] = {}
            if "urag-auto" in chosen:
                runs["urag-auto"] = safe_run(
                    "urag-auto",
                    lambda: urag_run(retriever.search(q.query, top_k=top_k)),
                )
            if "urag-hybrid" in chosen:
                runs["urag-hybrid"] = safe_run(
                    "urag-hybrid",
                    lambda: urag_run(
                        retriever.search(q.query, top_k=top_k, query_class="local")
                    ),
                )
            if "urag-lexical" in chosen:
                runs["urag-lexical"] = safe_run(
                    "urag-lexical",
                    lambda: urag_run(
                        retriever.search(q.query, top_k=top_k, mode="lexical")
                    ),
                )
            if "rg" in chosen:
                runs["rg"] = safe_run("rg", lambda: rg.search(q.query, top_k, db))
            if read is not None:
                runs["read"] = safe_run("read", lambda: read.search(q.query, top_k, db))
            if chunk is not None:
                runs["chunk"] = safe_run(
                    "chunk", lambda: chunk.search(q.query, top_k, db)
                )
            if (
                q.target
                and q.label != "reference"
                and ("urag-callers" in chosen or "urag-transitive" in chosen)
            ):
                if "urag-callers" in chosen:
                    runs["urag-callers"] = safe_run(
                        "urag-callers",
                        lambda: urag_run(
                            retriever.search_callers(q.target, limit=top_k)
                        ),
                    )
                if "urag-transitive" in chosen:
                    runs["urag-transitive"] = safe_run(
                        "urag-transitive",
                        lambda: urag_run(
                            retriever.search_transitive(
                                q.target, depth=q.depth or 3, limit=top_k
                            )
                        ),
                    )
            if q.label == "reference" and "urag-references" in chosen:
                runs["urag-references"] = safe_run(
                    "urag-references",
                    lambda: urag_run(
                        retriever.search_references(q.target, limit=top_k)
                    ),
                )
            if "oracle" in chosen:
                runs["oracle"] = safe_run("oracle", lambda: oracle.search(q, db))
            for name, run in runs.items():
                runs_by_system.setdefault(name, {})[i] = run
                rows[name][i] = _metrics(run, q, top_k)

        agg = {name: aggregate(v) for name, v in rows.items()}
        labels = sorted({q.label for q in qs})
        by_label = {
            name: {
                label: aggregate(
                    [row for q, row in zip(qs, system_rows) if q.label == label]
                )
                for label in labels
            }
            for name, system_rows in rows.items()
        }
        if json_out or report:
            from . import __version__

            payload = {
                "schema_version": EVAL_SCHEMA_VERSION,
                "urag_version": __version__,
                "root": str(cfg.project_root.resolve()),
                "top_k": top_k,
                "questions": [q.to_dict() for q in qs],
                "systems": {name: agg.get(name, {}) for name in chosen},
                "by_label": by_label,
                "per_query": rows,
                "hits": {
                    name: [
                        [
                            h.to_dict()
                            for h in (
                                runs_by_system.get(name, {}).get(i)
                                or SystemRun(name, [], 0.0, 0)
                            ).hits
                        ]
                        for i in range(len(qs))
                    ]
                    for name in chosen
                },
            }
            if chunk is not None:
                payload["chunk_load_seconds"] = chunk.load_seconds
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if json_out:
                print(text)
            if report:
                report.write_text(text, encoding="utf-8")
        if not json_out:
            t = Table(
                title=f"urag eval — {cfg.project_root} ({len(qs)} questions, top_k={top_k})"
            )
            t.add_column("system")
            t.add_column("recall@k")
            t.add_column("mrr")
            t.add_column("tokens/run")
            t.add_column("p50(s)")
            t.add_column("p95(s)")
            for name in chosen:
                a = agg.get(name, {})
                t.add_row(
                    name,
                    f"{a.get('unit_recall', 0):.2f}",
                    f"{a.get('mrr', 0):.2f}",
                    f"{a.get('mean_tokens', 0):.0f}",
                    f"{a.get('p50_sec', 0) * 1000:.0f}ms",
                    f"{a.get('p95_sec', 0) * 1000:.0f}ms",
                )
            console.print(t)

        if judge_url:
            console.print("[dim]running LLM judge tier...[/dim]")
            scores = judge_results(
                qs,
                runs_by_system,
                db,
                cfg,
                judge_url,
                judge_model or "gpt-4o-mini",
                judge_key or os.environ.get("URAG_JUDGE_KEY", ""),
                progress=lambda m: console.print(m),
            )
            console.print("[bold]answer quality (correct 0-10):[/bold]")
            for name in chosen:
                console.print(f"  {name}: {scores.get(name, float('nan')):.2f}")
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
            result = retriever.search_transitive_references(
                name, depth=depth, limit=top_k or 30
            )
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
    language: Optional[str] = typer.Option(
        None, "--language", help="filter by language"
    ),
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
        console.print(
            "[dim]heuristic candidates only — verify with git grep before removing[/dim]"
        )
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
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).callees(
            unit_id
        )
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
        result = Retriever(cfg, db, _embedder(cfg), Git(cfg.project_root)).list_symbols(
            file
        )
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
    start_opt: Optional[int] = typer.Option(
        None, "--start", help="first line (1-based)"
    ),
    end_opt: Optional[int] = typer.Option(None, "--end", help="last line (inclusive)"),
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
    if "xml" not in cfg.index.languages:
        xml_hits = _find_xml_family_files(cfg)
        if xml_hits:
            console.print(
                f"[yellow]hint: {xml_hits} XML-family file(s) found but 'xml' is not in "
                "index.languages (config predates XAML support); add it to "
                ".urag/urag.toml and re-run `urag index` to close the markup blind spot[/yellow]"
            )
    if cfg.embedding.provider == "local":
        cache = default_model_cache_dir()
        console.print(f"[bold]model cache:[/bold] {cache}")
        try:
            emb = create_embedder(cfg.embedding)
            emb.embed_query("probe")
            console.print(
                f"[bold]embedding:[/bold] {cfg.embedding.model} [green]OK[/green] ({emb.dimension}d)"
            )
        except Exception as exc:
            ok = False
            console.print(f"[bold]embedding:[/bold] [red]FAILED — {exc}[/red]")
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
