"""MCP server exposing urag as agent tools (model-agnostic).

Tools return compact evidence packets: search first with a small top_k,
then fetch exact source spans only for the units you plan to use.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .config import Config, discover_project_root, load_config
from .db import Database
from .embed import Embedder, NoopEmbedder, create_embedder
from .git_aware import Git
from .indexer import Indexer
from .retrieve import Retriever, fit_evidence

_embedder_lock = threading.Lock()
_embedder_cache: dict[str, Embedder] = {}

INSTRUCTIONS = """You are connected to urag, a structure-aware project index.

How to use it efficiently (token-conscious workflow):
1. `search` with top_k=3-5 first. Results are compact records (signature,
   summary, file:line). Prefer `mode=hybrid`; use `lexical` for exact
   symbol/identifier lookups, `dense` for conceptual questions.
2. Use `fetch_unit` only for the 1-3 most relevant hits to get the exact
   source span. Never request whole files.
3. `index_now` re-syncs after files change; `status` shows freshness.
4. If the project is not indexed yet, call `init_project` first.
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


def _open(cfg: Config) -> Database:
    return Database(cfg.db_path, cfg.embedding.dimension)


def _packet(r, include_evidence: bool, db: Database, budget: int) -> dict:
    u = r.unit
    packet = {
        "id": u.id,
        "name": u.name,
        "qualname": u.qualname,
        "type": u.unit_type,
        "signature": u.signature,
        "summary": u.summary,
        "file": r.file_path,
        "lines": [u.start_line, u.end_line],
        "score": round(r.score, 4),
        "ranks": {"lexical": r.lexical_rank, "dense": r.dense_rank},
        "commit": r.commit,
        "stale": r.stale,
    }
    if include_evidence and u.id is not None:
        ev = db.load_evidence(u.id)
        if ev and "span" in ev:
            packet["evidence"] = fit_evidence(ev["span"], budget)
    return packet


def create_server(root: Path | None = None) -> MCPServer:
    cfg = load_config(discover_project_root(root))
    git = Git(cfg.project_root)

    server = MCPServer(
        name="urag",
        title="urag project index",
        description="Structure-aware, token-efficient RAG for software projects",
        version="0.1.0",
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
        db = _open(cfg)
        try:
            result = Retriever(cfg, db, _embedder(cfg), git).search(
                query, top_k=top_k, mode=mode or "hybrid",
                language=language, query_class=query_class,
            )
            packets = [_packet(r, include_evidence, db, result.budget_tokens) for r in result.results]
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
        finally:
            db.close()

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
        db = _open(cfg)
        try:
            ev = Retriever(cfg, db, _embedder(cfg), git).get(unit_id)
            return json.dumps(ev, ensure_ascii=False) if ev else json.dumps({"error": "unit not found"})
        finally:
            db.close()

    @server.tool(
        name="callers",
        title="Find who calls a symbol",
        description=(
            "Exact call-graph lookup: returns the units that call `name`, "
            "with the call-site line. The precise way to answer 'what calls "
            "X' / 'what breaks if X changes'."
        ),
    )
    def callers(name: str, limit: int = 20) -> str:
        db = _open(cfg)
        try:
            result = Retriever(cfg, db, _embedder(cfg), git).search_callers(name, limit=limit)
            packets = [_packet(r, False, db, result.budget_tokens) for r in result.results]
            return json.dumps(
                {"query": name, "mode": "calls", "count": len(packets), "results": packets},
                ensure_ascii=False,
            )
        finally:
            db.close()

    @server.tool(
        name="index_now",
        title="Re-index the project",
        description="Incrementally re-index changed files and embed new units.",
    )
    def index_now() -> str:
        db = _open(cfg)
        try:
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
        finally:
            db.close()

    @server.tool(
        name="status",
        title="Project index status",
        description="Index stats: files, units, embeddings, freshness, config.",
    )
    def status() -> str:
        db = _open(cfg)
        try:
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
                },
                ensure_ascii=False,
            )
        finally:
            db.close()

    @server.tool(
        name="init_project",
        title="Initialize the project index",
        description=(
            "Set up .urag/ config and run a first full index. Call this when "
            "status reports the index is missing."
        ),
    )
    def init_project() -> str:
        cfg.urag_dir.mkdir(parents=True, exist_ok=True)
        db = _open(cfg)
        try:
            indexer = Indexer(cfg, db, _embedder(cfg))
            stats = indexer.index_all()
            s = db.stats()
            return json.dumps(
                {
                    "initialized": True,
                    "root": str(cfg.project_root),
                    **stats,
                    "units": s.units,
                    "embedded": s.embedded,
                }
            )
        finally:
            db.close()

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
