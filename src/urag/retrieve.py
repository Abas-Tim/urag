"""Retrieval: hybrid lexical + dense search with RRF fusion.

Lexical (FTS5) finds names and exact terms; dense (sqlite-vec) finds
concepts. Results are fused with Reciprocal Rank Fusion and returned as
compact packets; L2 evidence is loaded on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    lists: list[list[tuple[int, float]]], k: int = 60
) -> dict[int, dict[str, int | float]]:
    """Fuse ranked unit-id lists via RRF. Each list: (unit_id, rank_weight)."""
    fused: dict[int, dict[str, int | float]] = {}
    for list_idx, lst in enumerate(lists):
        for rank, (uid, _w) in enumerate(lst):
            entry = fused.setdefault(uid, {"rrf": 0.0, "lexical": None, "dense": None})
            entry["rrf"] = float(entry["rrf"]) + 1.0 / (k + rank + 1)
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
    used = 0
    for line in span.splitlines():
        cost = len(line) // 4 + 1
        if used + cost > budget_chars:
            break
        out.append(line)
        used += cost
    skipped = span.count("\n") + 1 - len(out)
    if skipped > 0:
        out.append(f"... ({skipped} more lines; full span via urag get <id>)")
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
        if not self.git:
            return {}
        per_commit: dict[str, set[str]] = {}
        for p in paths:
            row = self.db.conn.execute(
                'SELECT "commit" FROM files WHERE path = ?', (p,)
            ).fetchone()
            commit = row["commit"] if row else ""
            if commit:
                per_commit.setdefault(commit, set()).add(p)
        stale: dict[str, bool] = {}
        for commit, paths_for_commit in per_commit.items():
            changed = self._changed_since(commit)
            for p in paths_for_commit:
                stale[p] = p in changed
        return stale

    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        language: str | None = None,
        query_class: str | None = None,
    ) -> SearchResult:
        from .classify import BUDGETS, classify, top_k_for

        qc = query_class or classify(query)
        budget = BUDGETS[qc]["tokens"]
        k = top_k or top_k_for(qc)
        if qc == "impact":
            target = self._impact_symbol(query)
            if target:
                hits = self.db.callers(target, limit=max(k * 3, 10))
                if hits:
                    results = [
                        RetrievedUnit(
                            h["unit"], h["path"], score=1.0,
                            caller_of=h["callee_full"] or target, call_line=h["line"],
                            resolved_target=h.get("resolved_target", ""),
                        )
                        for h in hits[:k]
                    ]
                    self._enrich(results)
                    return SearchResult(results, "calls", query, qc, budget)
        if qc == "symbol" and mode == "hybrid":
            mode = "lexical"
        rc = self.cfg.retrieval
        mode = mode or "hybrid"
        if mode == "lexical":
            hits = self.db.lexical_search(query, rc.lexical_candidates, language)
            results = [RetrievedUnit(u, path, score=s) for u, path, s in hits[:k]]
        elif mode == "dense":
            try:
                qvec = self.embedder.embed_query(query)
            except RuntimeError:
                return SearchResult([], "dense", query, qc, budget)
            hits = self.db.dense_search(qvec, rc.dense_candidates, language)
            results = [RetrievedUnit(u, path, score=-s) for u, path, s in hits[:k]]
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
            fused = _rrf_scores(lists, rc.rrf_k)
            ranked = sorted(fused.items(), key=lambda kv: kv[1]["rrf"], reverse=True)[: max(k, 20)]
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
            results = results[:k]
        stale = self._stale_map([r.file_path for r in results])
        self._enrich(results, stale)
        return SearchResult(results, mode, query, qc, budget)

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
                    m = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", words[j].strip("?.,()[]"))
                    tok = m.group(0) if m else ""
                    if tok.lower() not in stop and tok.lower() not in verbs:
                        return tok
        for tok in reversed(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query)):
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
        return SearchResult(results, "calls", f"callers of {name}", "impact", BUDGETS["impact"]["tokens"])

    def get(self, unit_id: int) -> dict | None:
        ev = self.db.load_evidence(unit_id)
        if ev and self.git and ev.get("commit"):
            ev["stale"] = ev["file"] in self.git.changed_since(ev["commit"])
        return ev
