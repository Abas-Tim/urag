"""Retrieval: hybrid lexical + dense search with RRF fusion.

Lexical (FTS5) finds names and exact terms; dense (sqlite-vec) finds
concepts. Results are fused with Reciprocal Rank Fusion and returned as
compact packets; L2 evidence is loaded on demand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .db import Database
from .embed import Embedder
from .git_aware import Git
from .indexer import Indexer
from .models import RetrievedUnit
from .classify import BUDGETS


@dataclass
class SearchResult:
    results: list[RetrievedUnit]
    mode: str
    query: str
    query_class: str = "local"
    budget_tokens: int = 1500

    def to_dict(self, include_evidence: bool = False) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "class": self.query_class,
            "budget_tokens": self.budget_tokens,
            "count": len(self.results),
            "results": [r.to_dict(include_evidence=include_evidence) for r in self.results],
        }


def _rrf_scores(
    lists: list[list[tuple[int, float]]], k: int = 60, weights: list[float] | None = None
) -> dict[int, dict[str, int | float]]:
    """Fuse ranked unit-id lists via RRF. Each list: (unit_id, rank_weight)."""
    fused: dict[int, dict[str, int | float]] = {}
    for list_idx, lst in enumerate(lists):
        weight = weights[list_idx] if weights and list_idx < len(weights) else 1.0
        for rank, (uid, _w) in enumerate(lst):
            entry = fused.setdefault(uid, {"rrf": 0.0, "lexical": None, "dense": None})
            entry["rrf"] = float(entry["rrf"]) + weight / (k + rank + 1)
            if list_idx == 0:
                entry["lexical"] = rank + 1
            elif list_idx == 1:
                entry["dense"] = rank + 1
    return fused


def fit_evidence(span: str, max_tokens: int) -> str:
    """Trim a source span to a token budget (~4 chars/token), keeping the
    head of the unit and a truncation note. Full span stays available."""
    if not span:
        return span
    if max_tokens <= 0:
        return ""
    if len(span) // 4 <= max_tokens:
        return span
    budget_chars = max_tokens * 4
    out: list[str] = []
    used_chars = 0
    for line in span.splitlines():
        cost = len(line) + 1
        if used_chars + cost > budget_chars:
            break
        out.append(line)
        used_chars += cost
    skipped = span.count("\n") + 1 - len(out)
    if skipped > 0:
        note = f"... ({skipped} more lines; full span via urag get <id>)"
        if used_chars + len(note) + 1 > budget_chars:
            note = "... (full span via urag get <id>)"
        if used_chars + len(note) + 1 <= budget_chars:
            out.append(note)
    return "\n".join(out)


class Retriever:
    def __init__(self, cfg: Config, db: Database, embedder: Embedder, git: Git | None = None):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        self.git = git
        self._stale_cache: dict[str, set[str]] = {}

    # ---------- staleness (git provenance) ----------

    def _changed_since(self, commit: str) -> set[str]:
        if not self.git:
            return set()
        if commit not in self._stale_cache:
            self._stale_cache[commit] = self.git.changed_since(commit)
        return self._stale_cache[commit]

    def _stale_map(self, paths: list[str]) -> dict[str, bool]:
        root = getattr(getattr(self, "cfg", None), "project_root", self.db.db_path.parent.parent)
        stale: dict[str, bool] = {}
        for path in paths:
            row = self.db.conn.execute(
                'SELECT sha256, "commit" FROM files WHERE path = ?', (path,)
            ).fetchone()
            if row is None:
                stale[path] = True
                continue
            current = root / Path(path)
            if not current.exists():
                stale[path] = True
                continue
            try:
                digest = hashlib.sha256(current.read_bytes()).hexdigest()
            except OSError:
                stale[path] = True
                continue
            if row["sha256"]:
                stale[path] = digest != row["sha256"]
            else:
                stale[path] = bool(row["commit"] and path in self._changed_since(row["commit"]))
        return stale

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        language: str | None = None,
        query_class: str | None = None,
    ) -> SearchResult:
        from .classify import BUDGETS, MODE_BY_CLASS, classify, top_k_for

        qc = query_class or classify(query)
        rc = self.cfg.retrieval
        budget = BUDGETS[qc]["tokens"]
        if rc.max_evidence_tokens > 0:
            budget = min(budget, rc.max_evidence_tokens)
        default_k = top_k_for(qc)
        if top_k is None and rc.default_top_k > 0:
            default_k = min(default_k, rc.default_top_k)
        k = top_k or default_k
        if qc == "impact":
            target = self._impact_symbol(query)
            if target:
                depth = 3 if any(
                    phrase in query.lower()
                    for phrase in ("what breaks", "blast radius", "downstream", "transitive")
                ) else 1
                hits = (
                    self.db.transitive_callers(target, max_depth=depth, limit=max(k * 3, 10))
                    if depth > 1
                    else self.db.callers(target, limit=max(k * 3, 10))
                )
                if hits:
                    results = [
                        RetrievedUnit(
                            h["unit"], h["path"], score=1.0,
                            caller_of=h["callee_full"] or target, call_line=h["line"],
                            hop=h.get("hop", 0),
                            resolved_target=h.get("resolved_target", ""),
                        )
                        for h in hits
                    ]
                    results = self._limit_results(results, k)
                    self._enrich(results)
                    return SearchResult(results, "calls", query, qc, budget)
        if mode == "hybrid":
            mode = MODE_BY_CLASS.get(qc, mode)
        mode = mode or MODE_BY_CLASS.get(qc, "hybrid")
        if mode == "lexical":
            hits = self.db.lexical_search(query, rc.lexical_candidates, language, exact=qc == "symbol")
            results = [RetrievedUnit(u, path, score=s) for u, path, s in hits]
        elif mode == "dense":
            try:
                qvec = self.embedder.embed_query(query)
            except RuntimeError:
                return SearchResult([], "dense", query, qc, budget)
            hits = self.db.dense_search(qvec, rc.dense_candidates, language)
            results = [RetrievedUnit(u, path, score=-s) for u, path, s in hits]
        else:
            lists: list[list[tuple[int, float]]] = []
            lexical = self.db.lexical_search(query, rc.lexical_candidates, language)
            lists.append([(u.id or 0, s) for u, _p, s in lexical])
            try:
                qvec = self.embedder.embed_query(query)
                dense = self.db.dense_search(qvec, rc.dense_candidates, language)
                lists.append([(u.id or 0, s) for u, _p, s in dense])
            except RuntimeError:
                dense = []
            fused = _rrf_scores(
                lists, rc.rrf_k, weights=[rc.lexical_weight, rc.dense_weight]
            )
            ranked = sorted(fused.items(), key=lambda kv: kv[1]["rrf"], reverse=True)[: max(k * 4, 20)]
            by_id = {u.id: (u, p) for u, p, _s in lexical}
            for u, p, _s in dense:
                by_id.setdefault(u.id, (u, p))
            results = []
            for uid, scores in ranked:
                if uid in by_id:
                    u, p = by_id[uid]
                    results.append(
                        RetrievedUnit(
                            u,
                            p,
                            score=float(scores["rrf"]),
                            lexical_rank=scores["lexical"],
                            dense_rank=scores["dense"],
                        )
                    )
        results = self._limit_results(results, k)
        stale = self._stale_map([r.file_path for r in results])
        self._enrich(results, stale)
        return SearchResult(results, mode, query, qc, budget)

    def _limit_results(self, results: list[RetrievedUnit], limit: int) -> list[RetrievedUnit]:
        per_file: dict[str, int] = {}
        seen: set[tuple[str, str, str]] = set()
        selected: list[RetrievedUnit] = []
        cap = max(1, self.cfg.retrieval.max_results_per_file)
        for result in results:
            key = (result.file_path, result.unit.qualname, result.unit.unit_type)
            if key in seen or per_file.get(result.file_path, 0) >= cap:
                continue
            seen.add(key)
            per_file[result.file_path] = per_file.get(result.file_path, 0) + 1
            selected.append(result)
            if len(selected) >= limit:
                break
        return selected

    def _enrich(self, results: list[RetrievedUnit], stale: dict[str, bool] | None = None) -> None:
        """Attach commit + staleness to results."""
        stale = stale or self._stale_map([r.file_path for r in results])
        for r in results:
            row = self.db.conn.execute(
                'SELECT "commit" FROM files WHERE path = ?', (r.file_path,)
            ).fetchone()
            r.commit = row["commit"] if row else ""
            r.stale = stale.get(r.file_path, False)

    @staticmethod
    def _impact_symbol(query: str) -> str | None:
        """Extract the target symbol from an impact query like 'what calls X'."""
        import re

        keywords = ("calls", "uses", "breaks", "imports", "invokes", "depends", "references", "touches")
        stop = {
            "what", "who", "this", "the", "a", "an", "of", "for", "on", "in", "if", "when",
            "where", "which", "that", "why", "does", "is", "are", "it", "its", "they", "to", "i",
            "method", "function", "class", "interface", "code", "files", "file",
        }
        verbs = {
            "change", "changes", "changing", "changed", "modify", "edit", "editing",
            "remove", "removing", "removed", "add", "adding", "break", "breaking",
            "alter", "updating", "update", "call", "use", "import", "run", "invoke",
        }
        words = query.split()
        for i, w in enumerate(words):
            base = w.strip("?.,()[]")
            if base.lower() in keywords:
                for j in range(i + 1, len(words)):
                    m = re.match(r"[A-Za-z_][A-Za-z0-9_.$:]*", words[j].strip("?.,()[]"))
                    tok = m.group(0) if m else ""
                    if tok.lower() not in stop and tok.lower() not in verbs:
                        return tok
        for tok in reversed(re.findall(r"[A-Za-z_][A-Za-z0-9_.$:]*", query)):
            if tok.lower() not in stop and tok.lower() not in verbs:
                return tok
        return None

    def search_callers(self, name: str, limit: int = 20) -> SearchResult:
        """Direct call-graph lookup: who calls `name`."""
        hits = self.db.callers(name, limit=limit)
        results = [
            RetrievedUnit(
                h["unit"], h["path"], score=1.0,
                caller_of=h["callee_full"] or name, call_line=h["line"],
                resolved_target=h.get("resolved_target", ""),
            )
            for h in hits
        ]
        self._enrich(results)
        budget = self._impact_budget()
        return SearchResult(results, "calls", f"callers of {name}", "impact", budget)

    def search_transitive(self, name: str, depth: int = 3, limit: int = 20) -> SearchResult:
        """Multi-hop call-graph lookup: callers-of-callers up to `depth` hops."""
        hits = self.db.transitive_callers(name, max_depth=depth, limit=limit)
        results = [
            RetrievedUnit(
                h["unit"], h["path"], score=1.0,
                caller_of=h["callee_full"] or name, call_line=h["line"],
                hop=h.get("hop", 0),
            )
            for h in hits
        ]
        self._enrich(results)
        budget = self._impact_budget()
        return SearchResult(results, "calls", f"transitive callers of {name} (depth {depth})", "impact", budget)

    def _impact_budget(self) -> int:
        maximum = getattr(getattr(self, "cfg", None), "retrieval", None)
        maximum_tokens = getattr(maximum, "max_evidence_tokens", 0)
        if maximum_tokens > 0:
            return min(BUDGETS["impact"]["tokens"], maximum_tokens)
        return BUDGETS["impact"]["tokens"]

    def get(self, unit_id: int) -> dict | None:
        ev = self.db.load_evidence(unit_id)
        if ev and ev.get("file"):
            ev["stale"] = self._stale_map([ev["file"]]).get(ev["file"], True)
        return ev
