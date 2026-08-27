"""MCP server exposing urag as agent tools (model-agnostic).

Tools return compact evidence packets: search first with a small top_k,
then fetch exact source spans only for the units you plan to use.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import __version__
from .config import Config, discover_project_root, ensure_gitignore, load_config
from .db import Database
from .embed import Embedder, NoopEmbedder, create_embedder
from .git_aware import Git
from .indexer import Indexer
from .retrieve import Retriever, fit_evidence

_embedder_lock = threading.Lock()
_embedder_cache: dict[str, Embedder] = {}

INSTRUCTIONS = """You are connected to urag, a structure-aware project index.

Token-conscious workflow:
1. If `status` reports no index, call `init_project` (or `init_project`
   with embed=false for a fast lexical-only index) before searching.
2. `search` with top_k=3-5 first. Results are compact records (signature,
   summary, file:line). Prefer `mode=hybrid`; use `lexical` for exact
   symbol/identifier lookups, `dense` for conceptual questions.
3. Use `fetch_unit` (or `fetch_units` for several ids) only for the 1-3 most
   relevant hits to get exact source spans. Never request whole files.
4. Browse files and symbols without search: `list_files`, `list_symbols`,
   `read_file`, `resolve` (exact definition), `children` (methods of a class).
5. Ask impact questions precisely: `callers` (who calls X), `references`
   (who uses/constructs/mentions X, including XAML markup), `callees` (what X
   calls), `dependents` (what imports X), and `recent_changes` (git state).
6. For dead-code hunts use `references` + `callers` per candidate, and
   `dead_symbols` for a candidate list — then verify with grep before
   removing anything. `index_now` re-syncs after files change; `status`
   shows freshness.
Filter by `language` when you know the stack (python, typescript, javascript).
"""


def _embedder(cfg: Config) -> Embedder:
    key = f"{cfg.embedding.provider}:{cfg.embedding.model}"
    with _embedder_lock:
        if key not in _embedder_cache:
            try:
                _embedder_cache[key] = create_embedder(cfg.embedding)
            except Exception:
                _embedder_cache[key] = NoopEmbedder()
        return _embedder_cache[key]


class IndexUnavailableError(RuntimeError):
    pass


def _open(cfg: Config, create: bool = False) -> Database:
    if not create and not cfg.db_path.is_file():
        raise IndexUnavailableError("index missing; call init_project")
    try:
        return Database(cfg.db_path, cfg.embedding.dimension)
    except (sqlite3.DatabaseError, RuntimeError) as exc:
        raise IndexUnavailableError(f"index unavailable: {exc}") from exc


@contextmanager
def _database(cfg: Config, create: bool = False):
    db = _open(cfg, create=create)
    try:
        yield db
    except sqlite3.DatabaseError as exc:
        raise IndexUnavailableError(f"index unavailable: {exc}") from exc
    finally:
        db.close()


def _error_response(exc: IndexUnavailableError) -> str:
    return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _packet(r, include_evidence: bool, db: Database, budget: int) -> dict:
    u = r.unit
    packet = {
        "id": u.id,
        "name": u.name,
        "qualname": u.qualname,
        "type": u.unit_type,
        "kind": u.kind,
        "signature": u.signature,
        "summary": u.summary,
        "concepts": u.concepts,
        "relationships": u.relationships,
        "parent_id": u.parent_id,
        "file": r.file_path,
        "lines": [u.start_line, u.end_line],
        "score": round(r.score, 4),
        "ranks": {"lexical": r.lexical_rank, "dense": r.dense_rank},
        "commit": r.commit,
        "stale": r.stale,
    }
    if r.caller_of:
        packet["calls"] = r.caller_of
        packet["call_line"] = r.call_line
    if r.ref_kind:
        packet["ref_kind"] = r.ref_kind
    if r.hop > 0:
        packet["hop"] = r.hop
    if r.resolved_target:
        packet["resolved_to"] = r.resolved_target
    if include_evidence and u.id is not None:
        ev = db.load_evidence(u.id)
        if ev and "span" in ev:
            packet["evidence"] = fit_evidence(ev["span"], budget)
    return packet


def _evidence_budget(total: int, count: int) -> int:
    """Split a total token budget across `count` result packets."""
    if count <= 1:
        return total
    return max(200, total // count)


def _unit_meta(db: Database, unit_id: int) -> dict | None:
    got = db.unit_by_id(unit_id)
    if not got:
        return None
    u, path, commit = got
    return {
        "unit_id": unit_id,
        "name": u.name,
        "qualname": u.qualname,
        "type": u.unit_type,
        "signature": u.signature,
        "summary": u.summary,
        "file": path,
        "lines": [u.start_line, u.end_line],
        "commit": commit,
    }


def create_server(root: Path | None = None) -> MCPServer:
    project_root = root.resolve() if root is not None else discover_project_root()
    cfg = load_config(project_root)
    git = Git(cfg.project_root)

    server = MCPServer(
        name="urag",
        title="urag project index",
        description="Structure-aware, token-efficient RAG for software projects",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @server.tool(
        name="search",
        title="Search the project index",
        description=(
            "Search indexed symbols and docs. Returns compact records with "
            "signature, summary, file:line, commit provenance and ranks. "
            "Queries are auto-classified into budget tiers (symbol/local/"
            "debugging/impact); pass query_class to override. Use lexical "
            "mode for exact identifiers, hybrid for everything else."
        ),
    )
    def search(
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        language: str | None = None,
        include_evidence: bool = False,
        query_class: str | None = None,
    ) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).search(
                    query,
                    top_k=top_k,
                    mode=mode or "hybrid",
                    language=language,
                    query_class=query_class,
                )
                per_budget = _evidence_budget(result.budget_tokens, len(result.results))
                packets = [
                    _packet(r, include_evidence, db, per_budget) for r in result.results
                ]
                return json.dumps(
                    {
                        "query": query,
                        "mode": result.mode,
                        "class": result.query_class,
                        "budget_tokens": result.budget_tokens,
                        "count": len(packets),
                        "results": packets,
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="fetch_unit",
        title="Fetch exact source span for a unit",
        description=(
            "Load the exact source lines (L2 evidence) for a unit id returned "
            "by search. Prefer this over reading whole files. Includes the "
            "commit the unit was indexed at and a stale flag."
        ),
    )
    def fetch_unit(unit_id: int) -> str:
        try:
            with _database(cfg) as db:
                retriever = Retriever(cfg, db, _embedder(cfg), git)
                ev = retriever.get(unit_id)
                if not ev:
                    return json.dumps({"error": "unit not found"})
                meta = _unit_meta(db, unit_id) or {}
                meta.pop("unit_id", None)
                return json.dumps({**meta, **ev}, ensure_ascii=False)
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="fetch_units",
        title="Fetch exact source spans for several units",
        description=(
            "Batch version of fetch_unit: load exact source spans for a list "
            "of unit ids in one round-trip. Each entry includes the unit "
            "metadata, file, lines, span, commit, and a stale flag. Pass "
            "max_tokens to trim each span to a budget."
        ),
    )
    def fetch_units(unit_ids: list[int], max_tokens: int | None = None) -> str:
        try:
            with _database(cfg) as db:
                retriever = Retriever(cfg, db, _embedder(cfg), git)
                evs = retriever.get_many(unit_ids, max_tokens=max_tokens)
                enriched = []
                for ev in evs:
                    meta = _unit_meta(db, ev["unit_id"]) or {}
                    meta.pop("unit_id", None)
                    enriched.append({**meta, **ev})
                return json.dumps(
                    {"count": len(enriched), "results": enriched},
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="callers",
        title="Find who calls a symbol",
        description=(
            "Exact call-graph lookup: returns the units that call `name`, "
            "with the call-site line. The precise way to answer 'what calls "
            "X' / 'what breaks if X changes'. Pass depth > 1 to also walk "
            "callers-of-callers (multi-hop traversal); each result then "
            "carries its `hop` (1 = direct caller)."
        ),
    )
    def callers(name: str, limit: int = 20, depth: int = 1) -> str:
        try:
            with _database(cfg) as db:
                retriever = Retriever(cfg, db, _embedder(cfg), git)
                if depth > 1:
                    result = retriever.search_transitive(name, depth=depth, limit=limit)
                else:
                    result = retriever.search_callers(name, limit=limit)
                packets = [
                    _packet(r, False, db, result.budget_tokens) for r in result.results
                ]
                return json.dumps(
                    {
                        "query": name,
                        "mode": "calls",
                        "depth": depth,
                        "count": len(packets),
                        "results": packets,
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="references",
        title="Find who references a symbol",
        description=(
            "Usage-site lookup: returns the units that reference `name` — "
            "type mentions, object constructions (new X()), base classes, "
            "generic arguments, casts, attributes, and XAML bindings "
            "(x:Class, DataType, {x:Static}, {StaticResource}, event "
            "handlers). The precise way to answer 'who uses X' / 'is X "
            "dead'. Each result carries the reference site line and its "
            "kind. Pass depth > 1 to walk referencers-of-referencers."
        ),
    )
    def references(name: str, limit: int = 30, depth: int = 1) -> str:
        try:
            with _database(cfg) as db:
                retriever = Retriever(cfg, db, _embedder(cfg), git)
                if depth > 1:
                    result = retriever.search_transitive_references(
                        name, depth=depth, limit=limit
                    )
                else:
                    result = retriever.search_references(name, limit=limit)
                packets = [
                    _packet(r, False, db, result.budget_tokens) for r in result.results
                ]
                return json.dumps(
                    {
                        "query": name,
                        "mode": "references",
                        "depth": depth,
                        "count": len(packets),
                        "results": packets,
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="dead_symbols",
        title="List candidate dead symbols",
        description=(
            "Heuristic dead-code candidates: symbol units (classes, methods, "
            "functions, ...) with no incoming call edges and no incoming "
            "reference edges. Excludes imports, config keys, and files under "
            "test/spec paths. This is NOT proof of death — dynamic dispatch, "
            "reflection, entry points, and unsupported file types can produce "
            "false positives. Always verify candidates with the grep tool "
            "before removing code."
        ),
    )
    def dead_symbols(limit: int = 50, language: str | None = None) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).unreferenced(
                    limit=limit, language=language
                )
                packets = [
                    _packet(r, False, db, result.budget_tokens) for r in result.results
                ]
                return json.dumps(
                    {
                        "mode": "deadcode",
                        "note": "heuristic candidates only — verify before removing",
                        "count": len(packets),
                        "results": packets,
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="resolve",
        title="Find a symbol definition by name",
        description=(
            "Fast exact-definition lookup by name or qualified name. Returns "
            "the units whose name/qualname matches exactly (classes, "
            "functions, methods, etc.). Prefer this over search for a known "
            "symbol; use search for fuzzy or conceptual questions."
        ),
    )
    def resolve(name: str, limit: int = 10) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).resolve(
                    name, limit=limit
                )
                packets = [
                    _packet(r, False, db, result.budget_tokens) for r in result.results
                ]
                return json.dumps(
                    {
                        "query": name,
                        "mode": "resolve",
                        "count": len(packets),
                        "results": packets,
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="callees",
        title="Find what a unit calls",
        description=(
            "The inverse of callers: given a unit id, return every call site "
            "inside that unit (callee, full chain, line). Use to answer 'what "
            "does X call' / 'what are X's dependencies'."
        ),
    )
    def callees(unit_id: int) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).callees(unit_id)
                return (
                    json.dumps(result, ensure_ascii=False)
                    if result
                    else json.dumps({"error": "unit not found"})
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="dependents",
        title="Find what imports a module or symbol",
        description=(
            "Dependency lookup: return the files that import the given module "
            "or symbol (target), matched exactly or by sub-module prefix "
            "against import bindings and import units. Use for 'what depends "
            "on X' / 'what would break if I move X'."
        ),
    )
    def dependents(target: str, limit: int = 50) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).dependents(
                    target, limit=limit
                )
                result["count"] = len(result["results"])
                return json.dumps(result, ensure_ascii=False)
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="children",
        title="List the members of a unit",
        description=(
            "Structural navigation: list the child units of a unit id (e.g. "
            "the methods of a class). Set include_siblings=true to also "
            "return sibling units of the same parent."
        ),
    )
    def children(unit_id: int, include_siblings: bool = False) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).children(
                    unit_id, include_siblings=include_siblings
                )
                packets = [
                    _packet(r, False, db, result.budget_tokens) for r in result.results
                ]
                return json.dumps(
                    {
                        "unit_id": unit_id,
                        "mode": result.mode,
                        "count": len(packets),
                        "results": packets,
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="list_files",
        title="List indexed files",
        description=(
            "List all indexed files with language, kind, size, commit, and "
            "unit count. Optionally filter by language."
        ),
    )
    def list_files(language: str | None = None) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).list_files(
                    language=language
                )
                return json.dumps(result, ensure_ascii=False)
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="list_symbols",
        title="List symbols in a file",
        description=(
            "List every indexed unit (symbols and doc chunks) in a file, with "
            "signature, summary, and line range. Use to get an overview of a "
            "file without reading it whole."
        ),
    )
    def list_symbols(file: str) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).list_symbols(file)
                packets = [
                    _packet(r, False, db, result.budget_tokens) for r in result.results
                ]
                return json.dumps(
                    {"file": file, "count": len(packets), "results": packets},
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="read_file",
        title="Read a file (optionally a line range)",
        description=(
            "Read a file (or a line range) by project-relative path. Returns "
            "the lines as a trimmed span plus the total line count. Prefer "
            "list_symbols + fetch_unit for symbols; use this for config or "
            "context around a symbol."
        ),
    )
    def read_file(path: str, start: int | None = None, end: int | None = None) -> str:
        try:
            with _database(cfg) as db:
                result = Retriever(cfg, db, _embedder(cfg), git).read_file(
                    path, start=start, end=end
                )
                return json.dumps(result, ensure_ascii=False)
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="recent_changes",
        title="Show recent git changes",
        description=(
            "Best-effort git state: current branch and HEAD, working-tree "
            "changes (changed/deleted/untracked), and recent commits with "
            "their files. Use to understand what changed recently before "
            "relying on evidence."
        ),
    )
    def recent_changes(limit: int = 20) -> str:
        result = git.recent_changes(limit=limit)
        return json.dumps(result, ensure_ascii=False)

    @server.tool(
        name="index_now",
        title="Re-index the project",
        description="Incrementally re-index changed files and embed new units.",
    )
    def index_now() -> str:
        try:
            with _database(cfg) as db:
                indexer = Indexer(cfg, db, _embedder(cfg))
                stats = indexer.index_all()
                s = db.stats()
                return json.dumps(
                    {
                        **stats,
                        "units": s.units,
                        "embedded": s.embedded,
                        "last_indexed": s.last_indexed,
                    }
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.tool(
        name="status",
        title="Project index status",
        description="Index stats: files, units, embeddings, freshness, config.",
    )
    def status() -> str:
        try:
            with _database(cfg) as db:
                s = db.stats()
                return json.dumps(
                    {
                        "root": str(cfg.project_root),
                        "files": s.files,
                        "units": s.units,
                        "embedded": s.embedded,
                        "by_language": s.by_language,
                        "last_indexed": s.last_indexed,
                        "provider": cfg.embedding.provider,
                        "model": cfg.embedding.model,
                        "git": {
                            "branch": git.current_branch(),
                            "head": git.head(refresh=True),
                        },
                    },
                    ensure_ascii=False,
                )
        except IndexUnavailableError as exc:
            return json.dumps(
                {
                    "root": str(cfg.project_root),
                    "error": str(exc),
                    "next": "call init_project (optionally embed=false for a "
                    "fast lexical-only index) to build the index",
                },
                ensure_ascii=False,
            )

    @server.tool(
        name="init_project",
        title="Initialize the project index",
        description=(
            "Set up .urag/ config and run a first full index. Call this when "
            "status reports the index is missing. Pass embed=false for a fast "
            "lexical-only index (no model download); you can run index_now "
            "later with embeddings enabled."
        ),
    )
    def init_project(embed: bool = True) -> str:
        cfg.urag_dir.mkdir(parents=True, exist_ok=True)
        ensure_gitignore(cfg.project_root)
        try:
            with _database(cfg, create=True) as db:
                indexer = Indexer(cfg, db, _embedder(cfg) if embed else NoopEmbedder())
                stats = indexer.index_all()
                s = db.stats()
                return json.dumps(
                    {
                        "initialized": True,
                        "root": str(cfg.project_root),
                        "embedding": embed,
                        **stats,
                        "units": s.units,
                        "embedded": s.embedded,
                    }
                )
        except IndexUnavailableError as exc:
            return _error_response(exc)

    @server.resource("urag://unit/{unit_id}")
    def unit_resource(unit_id: str) -> str:
        try:
            with _database(cfg) as db:
                ev = db.load_evidence(int(unit_id))
                if not ev or "span" not in ev:
                    return json.dumps({"error": "unit not found"}, ensure_ascii=False)
                return ev["span"]
        except (IndexUnavailableError, ValueError):
            return json.dumps({"error": "unit not found"}, ensure_ascii=False)

    return server


def serve(root: Path | None = None) -> None:
    server = create_server(root)
    server.run(transport="stdio")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="urag-mcp")
    parser.add_argument("--root", default=os.environ.get("URAG_ROOT"))
    args = parser.parse_args()
    serve(Path(args.root) if args.root else None)
