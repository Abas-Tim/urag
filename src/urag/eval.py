"""urag eval: assess retrieval quality vs non-urag context baselines.

For each question in a gold set, run urag (hybrid + lexical), an rg/grep
baseline, a naive chunk-embedding baseline, and (optionally) a whole-file
oracle, then compare:
  - recall@k / MRR        (accuracy)
  - tokens/retrieval      (token efficiency, urag's thesis)
  - p50/p95 latency       (speed)
  - stale rate            (provenance)

Provable gold (definition + call questions) can be auto-generated from the
index via --autogen, so you can evaluate without hand labeling. A judge tier
(--judge-url/model/key) adds LLM answer-quality scoring per system.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, language_for_path
from .db import Database
from .embed import Embedder
from .indexer import Indexer
from .models import Unit
from .retrieve import Retriever


EVAL_SCHEMA_VERSION = 2


@dataclass
class Question:
    query: str
    gold_unit_ids: list[int] = field(default_factory=list)
    gold_file: str = ""
    label: str = "symbol"
    target: str = ""
    depth: int = 1
    gold_hops: dict[int, int] = field(default_factory=dict)
    gold_files: list[str] | None = None

    def __post_init__(self) -> None:
        self.gold_files = self.gold_files or (
            [self.gold_file] if self.gold_file else []
        )

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "label": self.label,
            "gold_file": self.gold_file
            or (self.gold_files[0] if self.gold_files else ""),
            "gold_files": self.gold_files,
            "gold_unit_ids": self.gold_unit_ids,
            "target": self.target,
            "depth": self.depth,
            "gold_hops": {str(k): v for k, v in self.gold_hops.items()},
        }


@dataclass
class Hit:
    file: str
    unit_id: int | None
    tokens: int
    detail: str = ""
    unit_ids: list[int] = field(default_factory=list)
    title: str = ""

    def to_dict(self, detail_cap: int = 400) -> dict:
        return {
            "file": self.file,
            "unit_id": self.unit_id,
            "unit_ids": self.unit_ids,
            "tokens": self.tokens,
            "title": self.title[:detail_cap],
            "detail": self.detail[:detail_cap]
            + ("…" if len(self.detail) > detail_cap else ""),
        }


@dataclass
class SystemRun:
    name: str
    hits: list[Hit]
    seconds: float
    tokens: int


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _unit_tokens(u: Unit) -> int:
    return _tokens(f"{u.signature} {u.summary} {u.concepts}")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class RgBaseline:
    """opencode `grep` emulation: rank files by matching-line counts and show
    the matching lines, like the agent's grep tool output."""

    STOPWORDS = {
        "who",
        "what",
        "where",
        "when",
        "which",
        "why",
        "how",
        "is",
        "are",
        "was",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "for",
        "do",
        "does",
        "did",
        "call",
        "calls",
        "called",
        "calling",
        "define",
        "defined",
        "definition",
        "defines",
        "transitively",
        "and",
        "or",
        "it",
        "that",
    }

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = path.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def search(self, query: str, top_k: int, db: Database) -> SystemRun:
        import re

        raw = re.split(r"[^A-Za-z0-9_.]+", query)
        terms: list[str] = []
        seen: set[str] = set()
        for t in raw:
            if len(t) <= 1 or t.lower() in self.STOPWORDS:
                continue
            candidates = [
                p
                for p in re.split(r"[._]", t)
                if len(p) > 1 and p.lower() not in self.STOPWORDS
            ] + [t]
            for cand in candidates:
                if cand not in seen:
                    seen.add(cand)
                    terms.append(cand)
            if len(terms) >= 4:
                break
        terms = terms[:4]
        t0 = time.perf_counter()
        if not terms:
            return SystemRun("rg", [], 0.0, 0)
        counts: dict[str, int] = {}
        lines: dict[str, list[str]] = {}
        for term in terms:
            try:
                proc = subprocess.run(
                    ["rg", "--no-messages", "-i", "-n", "-F", term, "."],
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            for line in proc.stdout.splitlines():
                if line.count(":") < 2:
                    continue
                path, lineno, text = line.rsplit(":", 2)
                path = self._normalize_path(path)
                counts[path] = counts.get(path, 0) + 1
                bucket = lines.setdefault(path, [])
                if len(bucket) < 40:
                    entry = f"{lineno}: {text}"
                    if entry not in bucket:
                        bucket.append(entry)
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: top_k * 2]
        hits: list[Hit] = []
        for path, _cnt in ranked:
            rows = db.conn.execute(
                "SELECT u.id FROM units u JOIN files f ON f.id = u.file_id WHERE f.path = ? ORDER BY u.start_line LIMIT 3",
                (path,),
            ).fetchall()
            ids = [r["id"] for r in rows]
            detail = "\n".join(lines[path])
            matched_tokens = _tokens(detail)
            for uid in ids:
                hits.append(Hit(path, uid, matched_tokens, detail=detail))
            if not ids:
                hits.append(Hit(path, None, matched_tokens, detail=detail))
            if len(hits) >= top_k * 5:
                break
        return SystemRun(
            "rg", hits, time.perf_counter() - t0, sum(h.tokens for h in hits)
        )


class ReadBaseline:
    """opencode `read` emulation: the standard agent loop is grep to find
    candidate files, then read the top files whole. Context = full file
    contents, so token cost is what the agent actually pays."""

    MAX_FILES = 10

    def __init__(self, root: Path):
        self.root = root
        self.rg = RgBaseline(root)

    def search(self, query: str, top_k: int, db: Database) -> SystemRun:
        t0 = time.perf_counter()
        grepped = self.rg.search(query, top_k, db)
        files: list[str] = []
        for h in grepped.hits:
            if h.file not in files:
                files.append(h.file)
            if len(files) >= self.MAX_FILES:
                break
        if not files:
            return SystemRun("read", [], time.perf_counter() - t0, 0)
        hits: list[Hit] = []
        total = 0
        for path in files[:top_k]:
            p = self.root / path
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rows = db.conn.execute(
                "SELECT u.id, u.unit_type FROM units u JOIN files f ON f.id = u.file_id WHERE f.path = ? AND u.unit_type != 'import'",
                (path,),
            ).fetchall()
            unit_ids = [r["id"] for r in rows]
            toks = _tokens(text)
            total += toks
            hits.append(
                Hit(
                    path,
                    unit_ids[0] if unit_ids else None,
                    toks,
                    detail="\n".join(text.splitlines()[:40]),
                    unit_ids=unit_ids,
                )
            )
        return SystemRun("read", hits, time.perf_counter() - t0, total)


class ChunkBaseline:
    """Naive structure-free RAG: fixed-size chunks of every file, embedded."""

    CHUNK_CHARS = 600

    def __init__(self, cfg: Config, db: Database, embedder: Embedder):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        chunks: list[
            tuple[str, str, int, int]
        ] = []  # (file, text, byte_start, byte_end)
        for p in Indexer(cfg, db, embedder).discover():
            rel = p.relative_to(cfg.project_root).as_posix()
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            size = len(text)
            for start in range(0, size, self.CHUNK_CHARS):
                seg = text[start : start + self.CHUNK_CHARS]
                chunks.append((rel, seg, start, start + len(seg)))
        self.chunks = chunks
        t0 = time.perf_counter()
        self.vectors = embedder.embed_passages([c[1] for c in chunks])
        self.load_seconds = time.perf_counter() - t0

    def search(self, query: str, top_k: int, db: Database) -> SystemRun:
        t0 = time.perf_counter()
        q = self.embedder.embed_query(query)
        scored = [
            (i, sum(a * b for a, b in zip(q, v))) for i, v in enumerate(self.vectors)
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        hits: list[Hit] = []
        for i, _s in scored[:top_k]:
            file, text, bs, be = self.chunks[i]
            unit_id = self._unit_at(db, file, bs)
            hits.append(Hit(file, unit_id, _tokens(text)))
        return SystemRun(
            "chunk-rag", hits, time.perf_counter() - t0, sum(h.tokens for h in hits)
        )

    @staticmethod
    def _unit_at(db: Database, file: str, byte_start: int) -> int | None:
        row = db.conn.execute(
            """
            SELECT u.id FROM units u JOIN files f ON f.id = u.file_id
            WHERE f.path = ? AND u.byte_start <= ? AND u.byte_end >= ? AND u.unit_type != 'import'
            ORDER BY (u.byte_end - u.byte_start) LIMIT 1
            """,
            (file, byte_start, byte_start),
        ).fetchone()
        return row["id"] if row else None


class OracleBaseline:
    """Whole gold evidence as context (answer-quality upper bound): one span
    per gold unit across ALL gold files (not just the first one)."""

    def __init__(self, root: Path):
        self.root = root

    def _safe_file(self, relative_path: str) -> Path | None:
        root = self.root.resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def search(self, question: Question, db: Database) -> SystemRun:
        hits: list[Hit] = []
        seen: set[tuple[str, int]] = set()
        for gid in question.gold_unit_ids:
            got = db.unit_by_id(gid)
            if not got or (got[1], gid) in seen:
                continue
            seen.add((got[1], gid))
            u, path, _ = got
            text = db.load_evidence(gid) or {}
            span = text.get("span", "")
            toks = _tokens(span) if span else _unit_tokens(u)
            hits.append(Hit(path, gid, toks, detail=span, title=u.signature or u.name))
        if not hits:
            for path in question.gold_files or [question.gold_file]:
                p = self._safe_file(path)
                if p and p.is_file():
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    hits.append(Hit(path, None, _tokens(txt), detail=txt))
        return SystemRun("oracle", hits, 0.0, sum(h.tokens for h in hits))


# ---------------------------------------------------------------------------
# Gold resolution + autogeneration
# ---------------------------------------------------------------------------


def resolve_question(db: Database, q: Question) -> Question:
    if q.gold_unit_ids:
        if not q.gold_files:
            marks = ",".join("?" * len(q.gold_unit_ids))
            rows = db.conn.execute(
                f"SELECT DISTINCT f.path FROM units u JOIN files f ON f.id = u.file_id WHERE u.id IN ({marks}) ORDER BY f.path",
                q.gold_unit_ids,
            ).fetchall()
            q.gold_files = [r["path"] for r in rows]
        if not q.gold_file and q.gold_files:
            q.gold_file = q.gold_files[0]
        return q
    if q.gold_files:
        marks = ",".join("?" * len(q.gold_files))
        rows = db.conn.execute(
            f"SELECT u.id FROM units u JOIN files f ON f.id = u.file_id WHERE f.path IN ({marks}) ORDER BY f.path, u.start_line",
            q.gold_files,
        ).fetchall()
        q.gold_unit_ids = [r["id"] for r in rows]
        if not q.gold_file:
            q.gold_file = q.gold_files[0]
        return q
    if q.gold_file:
        q.gold_files = [q.gold_file]
        rows = db.conn.execute(
            "SELECT u.id FROM units u JOIN files f ON f.id = u.file_id WHERE f.path = ? ORDER BY u.start_line",
            (q.gold_file,),
        ).fetchall()
        q.gold_unit_ids = [r["id"] for r in rows]
        return q
    return q


def _seeded_shuffle(items: list, seed: int = 42) -> list:
    import random

    items = list(items)
    random.Random(seed).shuffle(items)
    return items


def autogen_questions(db: Database, n: int) -> list[Question]:
    """Auto-generate provable-gold questions: definitions + calls."""
    qs: list[Question] = []
    # definition questions: pick n units with meaningful signatures
    rows = db.conn.execute(
        "SELECT id, qualname FROM units WHERE unit_type IN ('function','method','class','struct','interface','enum') AND qualname != '' ORDER BY id",
    ).fetchall()
    for r in _seeded_shuffle(rows)[:n]:
        qs.append(
            Question(
                query=f"where is {r['qualname']} defined",
                gold_unit_ids=[r["id"]],
                gold_file="",
                label="definition",
            )
        )
    # call questions: pick callees that have callers
    calls = db.conn.execute(
        """
        SELECT e.callee, GROUP_CONCAT(DISTINCT e.caller_unit_id) AS callers
        FROM call_edges e GROUP BY e.callee ORDER BY e.callee
        """,
    ).fetchall()
    for r in _seeded_shuffle(calls)[:n]:
        ids = [int(x) for x in r["callers"].split(",")]
        qs.append(
            Question(
                query=f"what calls {r['callee']}",
                gold_unit_ids=ids,
                label="call",
                target=r["callee"],
            )
        )
    return qs


# ---------------------------------------------------------------------------
# Call-graph benchmark questions (transitive + alias), provable gold
# ---------------------------------------------------------------------------


def transitive_caller_ids(
    db: Database, name: str, max_depth: int = 3
) -> dict[int, int]:
    """BFS over call_edges: unit_id -> shortest hop (1 = direct caller).

    Uses the same caller-resolution path as production retrieval
    (``Database.callers``), so the gold matches what the system under test
    can actually find — including alias-resolved and fully-qualified names.
    """
    seen: dict[int, int] = {}
    frontier: list[str] = [name]
    for hop in range(1, max_depth + 1):
        nxt: list[str] = []
        for callee in frontier:
            for row in db.callers(callee, limit=10000):
                uid = row["unit"].id
                if uid in seen:
                    continue
                seen[uid] = hop
                u = row["unit"]
                nxt.append(u.name)
                if u.qualname and u.qualname != u.name:
                    nxt.append(u.qualname)
        frontier = list(dict.fromkeys(nxt))
        if not frontier:
            break
    return seen


def autogen_transitive_questions(
    db: Database, n: int, max_depth: int = 3
) -> list[Question]:
    """Questions whose gold = ALL transitive callers (hops 1..max_depth)."""
    rows = db.conn.execute(
        "SELECT DISTINCT callee FROM call_edges ORDER BY callee"
    ).fetchall()
    qs: list[Question] = []
    for r in _seeded_shuffle(rows):
        callee = r["callee"]
        seen = transitive_caller_ids(db, callee, max_depth=max_depth)
        if not any(h >= 2 for h in seen.values()):
            continue
        qs.append(
            Question(
                query=f"who transitively calls {callee}",
                gold_unit_ids=list(seen.keys()),
                label="transitive",
                target=callee,
                depth=max_depth,
                gold_hops=seen,
            )
        )
        if len(qs) >= n:
            break
    return qs


def scan_import_aliases(source: str, language: str) -> dict[str, str]:
    """alias -> fully-qualified target, from source text (harness-independent)."""
    aliases: dict[str, str] = {}
    if language == "python":
        for line in source.splitlines():
            s = line.strip()
            m = re.match(r"^import\s+([\w.]+)\s+as\s+(\w+)$", s)
            if m:
                aliases[m.group(2)] = m.group(1)
                continue
            m = re.match(r"^from\s+([\w.]+)\s+import\s+(.+)$", s)
            if m:
                for part in m.group(2).split(","):
                    pm = re.match(r"\s*([\w.]+)\s+as\s+(\w+)\s*$", part)
                    if pm:
                        aliases[pm.group(2)] = f"{m.group(1)}.{pm.group(1)}"
    elif language == "typescript":
        for line in source.splitlines():
            s = line.strip()
            m = re.match(r"^import\s+\*\s+as\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", s)
            if m:
                aliases[m.group(1)] = m.group(2).strip("@").replace("/", ".")
                continue
            m = re.match(r"^import\s+(\w+)\s+from\s+['\"]([^'\"]+)['\"]", s)
            if m:
                aliases[m.group(1)] = (
                    f"{m.group(2).strip('@').replace('/', '.')}.default"
                )
                continue
            m = re.match(r"^import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", s)
            if m:
                mod = m.group(2).strip("@").replace("/", ".")
                for part in m.group(1).split(","):
                    pm = re.match(r"\s*(\w+)\s+as\s+(\w+)\s*$", part)
                    if pm:
                        aliases[pm.group(2)] = f"{mod}.{pm.group(1)}"
    elif language == "go":
        for line in source.splitlines():
            m = re.match(r'^\s*import\s+(\w+)\s+"([^"]+)"\s*$', line)
            if m:
                aliases[m.group(1)] = m.group(2).replace("/", ".")
    elif language == "csharp":
        for line in source.splitlines():
            m = re.match(r"^\s*using\s+(\w+)\s*=\s*([\w.]+)\s*;", line)
            if m:
                aliases[m.group(1)] = m.group(2)
    return aliases


def _unit_at_byte(db: Database, file_id: int, byte_start: int) -> int | None:
    """Smallest unit containing byte_start (import units excluded)."""
    row = db.conn.execute(
        """
        SELECT u.id FROM units u
        WHERE u.file_id = ? AND u.byte_start <= ? AND u.byte_end >= ? AND u.unit_type != 'import'
        ORDER BY (u.byte_end - u.byte_start) LIMIT 1
        """,
        (file_id, byte_start, byte_start),
    ).fetchone()
    return row["id"] if row else None


def autogen_alias_questions(db: Database, root: Path, n: int) -> list[Question]:
    """Aliased call sites, gold = the units owning those call sites; one
    question per fully-qualified target chain (e.g. `import os.path as op` +
    `op.exists()` -> "who calls os.path.exists")."""
    from .extractors import get_extractor

    files = db.conn.execute("SELECT id, path, language FROM files").fetchall()
    owners: dict[str, tuple[list[int], str]] = {}
    for f in files:
        if not f["language"] or f["language"] == "markdown":
            continue
        p = root / f["path"]
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        aliases = scan_import_aliases(text, f["language"])
        if not aliases:
            continue
        ext = get_extractor(f["language"])
        if ext is None:
            continue
        for c in ext.collect_calls(text):
            for alias, target in aliases.items():
                prefix = alias + "."
                if c.callee_full == alias:
                    real = target
                elif c.callee_full.startswith(prefix):
                    real = f"{target}.{c.callee_full[len(prefix) :]}"
                else:
                    continue
                owner = _unit_at_byte(db, f["id"], c.byte_start)
                if owner is None:
                    continue
                cur = owners.setdefault(real, ([], f["path"]))
                if owner not in cur[0]:
                    cur[0].append(owner)
    qs: list[Question] = []
    for real, (ids, path) in owners.items():
        qs.append(
            Question(
                query=f"who calls {real}",
                gold_unit_ids=ids,
                gold_file=path,
                label="alias",
                target=real,
            )
        )
        if len(qs) >= n:
            break
    return qs


def load_questions(path: Path) -> list[Question]:
    out: list[Question] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                raise ValueError("expected a JSON object")
            out.append(
                Question(
                    query=d["query"],
                    gold_unit_ids=d.get("gold_unit_ids", []),
                    gold_file=d.get("gold_file", ""),
                    label=d.get("label", "custom"),
                    target=d.get("target", ""),
                    depth=d.get("depth", 1),
                    gold_hops={int(k): v for k, v in d.get("gold_hops", {}).items()},
                    gold_files=d.get("gold_files")
                    or ([d["gold_file"]] if d.get("gold_file") else []),
                )
            )
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ) as exc:
            raise ValueError(f"{path}:{line_number}: invalid question: {exc}") from exc
    return out


def reresolve_questions(db: Database, qs: list[Question]) -> list[Question]:
    """Re-derive gold from the current index for questions taken from an
    older report (cross-branch comparison): unit ids are index-specific, so
    gold must be recomputed from the stored query/target/label. Questions
    whose gold disappears (deleted symbols) are dropped."""
    out: list[Question] = []
    for q in qs:
        rebuilt = Question(
            query=q.query,
            gold_file=q.gold_file,
            label=q.label,
            target=q.target,
            depth=q.depth,
        )
        if q.label == "definition":
            name = q.query.removeprefix("where is ").removesuffix(" defined").strip()
            rows = db.conn.execute(
                "SELECT id FROM units WHERE (qualname = ? OR name = ?) "
                "AND unit_type NOT IN ('import', 'config_key') ORDER BY start_line LIMIT 1",
                (name, name),
            ).fetchall()
            rebuilt.gold_unit_ids = [r["id"] for r in rows]
        elif q.label in ("call", "alias") and q.target:
            rebuilt.gold_unit_ids = sorted(
                {row["unit"].id for row in db.callers(q.target, limit=100000)}
            )
        elif q.label == "transitive" and q.target:
            hops = transitive_caller_ids(db, q.target, max_depth=q.depth or 3)
            rebuilt.gold_unit_ids = list(hops)
            rebuilt.gold_hops = hops
        else:
            rebuilt.gold_unit_ids = list(q.gold_unit_ids)
            rebuilt.gold_files = list(q.gold_files or [])
        if rebuilt.gold_unit_ids or rebuilt.gold_files:
            out.append(rebuilt)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(run: SystemRun, q: Question, top_k: int) -> dict:
    gold_ids = set(q.gold_unit_ids)
    gold_files = set(q.gold_files or ([q.gold_file] if q.gold_file else []))
    hits = run.hits[:top_k]
    hit_files = {h.file for h in hits}
    hit_unit_ids = {h.unit_id for h in hits if h.unit_id is not None}
    for h in hits:
        hit_unit_ids.update(h.unit_ids)

    unit_recall = len(hit_unit_ids & gold_ids) / len(gold_ids) if gold_ids else 0.0
    file_recall = len(hit_files & gold_files) / len(gold_files) if gold_files else 0.0

    relevant = hit_unit_ids & gold_ids if gold_ids else set()
    precision = len(relevant) / len(hit_unit_ids) if hit_unit_ids else 0.0

    indirect_recall: float | None = None
    if q.gold_hops:
        found = {uid for uid in hit_unit_ids if uid in q.gold_hops}
        indirect = {uid for uid, hop in q.gold_hops.items() if hop >= 2}
        if indirect:
            indirect_recall = len(found & indirect) / len(indirect)

    mrr = 0.0
    if gold_ids:
        for i, h in enumerate(hits[:top_k]):
            if (h.unit_id in gold_ids) or (gold_ids & set(h.unit_ids)):
                mrr = 1.0 / (i + 1)
                break
    elif gold_files:
        for i, h in enumerate(hits[:top_k]):
            if h.file in gold_files:
                mrr = 1.0 / (i + 1)
                break
    return {
        "unit_recall": unit_recall,
        "file_recall": file_recall,
        "precision": precision,
        "indirect_recall": indirect_recall,
        "mrr": mrr,
        "tokens": run.tokens,
        "seconds": run.seconds,
        "n_hits": len(hits),
    }


def aggregate(rows: list[dict | None]) -> dict:
    rows = [r for r in rows if r is not None]
    if not rows:
        return {}
    lat = sorted(r["seconds"] for r in rows)
    n = len(lat)

    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "n": n,
        "unit_recall": sum(r["unit_recall"] for r in rows) / n,
        "file_recall": sum(r["file_recall"] for r in rows) / n,
        "precision": _mean("precision"),
        "indirect_recall": _mean("indirect_recall"),
        "mrr": sum(r["mrr"] for r in rows) / n,
        "mean_tokens": sum(r["tokens"] for r in rows) / n,
        "mean_sec": sum(lat) / n,
        "p50_sec": lat[(n - 1) // 2] if n else 0.0,
        "p95_sec": lat[int(math.ceil(0.95 * n)) - 1] if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Judge (optional LLM tier)
# ---------------------------------------------------------------------------


def _llm_chat(url: str, model: str, key: str, messages: list[dict]) -> str:
    import httpx

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            url.rstrip("/") + "/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def judge_results(
    questions: list[Question],
    runs_by_system: dict[str, dict[int, SystemRun]],
    db: Database,
    cfg: Config,
    url: str,
    model: str,
    key: str,
    progress=None,
) -> dict[str, float]:
    """Answer each question per system with the LLM, then grade the answer."""
    from .extractors import get_extractor

    log = progress or (lambda m: None)
    scores: dict[str, list[float]] = {}
    for i, q in enumerate(questions):
        for sys_name, by_q in runs_by_system.items():
            run = by_q.get(i)
            if run is None:
                continue
            context = _system_context(run, q, db, cfg)
            try:
                ans = _llm_chat(
                    url,
                    model,
                    key,
                    [
                        {
                            "role": "system",
                            "content": "You answer questions about a codebase using ONLY the provided context. If the context is insufficient, say so explicitly and mark it insufficient.",
                        },
                        {
                            "role": "user",
                            "content": f"CONTEXT:\n{context}\n\nQUESTION: {q.query}\nAnswer concisely.",
                        },
                    ],
                )
                grade = _llm_chat(
                    url,
                    model,
                    key,
                    [
                        {
                            "role": "system",
                            "content": 'You are an evaluation judge. Given a question, a candidate answer, and the context that was available, respond with ONLY a JSON object {"correct": 0-10, "sufficient": "yes"|"no"}. correct = how well the answer answers the question given the available context (not hallucinating).',
                        },
                        {
                            "role": "user",
                            "content": f"QUESTION: {q.query}\n\nANSWER:\n{ans}\n\nCONTEXT_AVAILABLE:\n{context}",
                        },
                    ],
                )
            except Exception:
                log(f"  [{sys_name}] q{i}: judge request failed")
                continue
            try:
                g = json.loads(grade)
                scores.setdefault(sys_name, []).append(float(g.get("correct", 0)))
            except Exception:
                pass
            log(f"  [{sys_name}] q{i}: {ans[:60]}...")
    return {name: sum(v) / len(v) for name, v in scores.items()}


def _system_context(run: SystemRun, q: Question, db: Database, cfg: Config) -> str:
    parts = []
    for h in run.hits:
        if h.unit_id is not None:
            if h.detail:
                parts.append(f"--- {h.file} (unit {h.unit_id}) ---\n{h.detail}")
                continue
            ev = db.load_evidence(h.unit_id)
            if ev and "span" in ev:
                parts.append(f"--- {h.file} (unit {h.unit_id}) ---\n{ev['span']}")
            else:
                parts.append(f"- {h.file} (unit id {h.unit_id})")
        elif h.detail:
            parts.append(f"--- {h.file} ---\n{h.detail}")
        else:
            parts.append(f"- {h.file}")
    return "\n".join(parts) if parts else "(no context retrieved)"
