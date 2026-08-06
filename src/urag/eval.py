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


@dataclass
class Question:
    query: str
    gold_unit_ids: list[int] = field(default_factory=list)
    gold_file: str = ""
    label: str = "symbol"
    target: str = ""
    depth: int = 1
    gold_hops: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "label": self.label,
            "gold_file": self.gold_file,
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
    """Rank files by number of regex matches for the query terms."""

    def __init__(self, root: Path):
        self.root = root

    def search(self, query: str, top_k: int, db: Database) -> SystemRun:
        import re

        terms = [t for t in re.split(r"[^A-Za-z0-9_.]+", query) if len(t) > 1][:4]
        t0 = time.perf_counter()
        if not terms:
            return SystemRun("rg", [], 0.0, 0)
        # one rg per term, count matches per file
        counts: dict[str, int] = {}
        for term in terms:
            try:
                proc = subprocess.run(
                    ["rg", "--no-messages", "-i", "--count-matches", term, "."],
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            for line in proc.stdout.splitlines():
                if ":" in line:
                    path, cnt = line.rsplit(":", 1)
                    try:
                        counts[path.replace("\\", "/")] = counts.get(path.replace("\\", "/"), 0) + int(cnt)
                    except ValueError:
                        continue
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: top_k * 2]
        # map files to their units for recall
        hits: list[Hit] = []
        for path, cnt in ranked:
            rows = db.conn.execute(
                "SELECT u.id FROM units u JOIN files f ON f.id = u.file_id WHERE f.path = ? ORDER BY u.start_line LIMIT ?",
                (path, 3),
            ).fetchall()
            ids = [r["id"] for r in rows]
            matched_txt = cnt * 40  # ~matched lines worth of context
            for uid in ids:
                hits.append(Hit(path, uid, _tokens(str(matched_txt))))
            if not ids:
                hits.append(Hit(path, None, _tokens(str(matched_txt))))
            if len(hits) >= top_k * 5:
                break
        return SystemRun("rg", hits, time.perf_counter() - t0, sum(h.tokens for h in hits))


class ChunkBaseline:
    """Naive structure-free RAG: fixed-size chunks of every file, embedded."""

    CHUNK_CHARS = 600

    def __init__(self, cfg: Config, db: Database, embedder: Embedder):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        chunks: list[tuple[str, str, int, int]] = []  # (file, text, byte_start, byte_end)
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
            (i, sum(a * b for a, b in zip(q, v)))
            for i, v in enumerate(self.vectors)
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        hits: list[Hit] = []
        for i, _s in scored[:top_k]:
            file, text, bs, be = self.chunks[i]
            unit_id = self._unit_at(db, file, bs)
            hits.append(Hit(file, unit_id, _tokens(text)))
        return SystemRun("chunk-rag", hits, time.perf_counter() - t0, sum(h.tokens for h in hits))

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
    """Whole gold files as context (answer-quality upper bound)."""

    def __init__(self, root: Path):
        self.root = root

    def search(self, question: Question, db: Database) -> SystemRun:
        hits: list[Hit] = []
        total = 0
        for gid in question.gold_unit_ids:
            got = db.unit_by_id(gid)
            if got:
                u, path, _ = got
                if not any(h.file == path for h in hits):
                    text = db.load_evidence(gid) or {}
                    span = text.get("span", "")
                    total += _tokens(span) if span else _unit_tokens(u)
                    hits.append(Hit(path, gid, _tokens(span) if span else _unit_tokens(u)))
        if question.gold_file and not any(h.file == question.gold_file for h in hits):
            p = self.root / question.gold_file
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace")
                total += _tokens(txt)
                hits.append(Hit(question.gold_file, None, _tokens(txt)))
        return SystemRun("oracle", hits, 0.0, total)


# ---------------------------------------------------------------------------
# Gold resolution + autogeneration
# ---------------------------------------------------------------------------


def resolve_question(db: Database, q: Question) -> Question:
    if q.gold_unit_ids:
        if not q.gold_file:
            row = db.conn.execute(
                "SELECT f.path FROM units u JOIN files f ON f.id = u.file_id WHERE u.id = ?",
                (q.gold_unit_ids[0],),
            ).fetchone()
            if row:
                q.gold_file = row["path"]
        return q
    if q.gold_file:
        rows = db.conn.execute(
            "SELECT u.id FROM units u JOIN files f ON f.id = u.file_id WHERE f.path = ?",
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


def _raw_callers(db: Database, names: list[str]) -> list[dict]:
    """Exact-match callers of any of `names` (by last segment or full chain)."""
    names = [n for n in names if n]
    if not names:
        return []
    marks = ",".join("?" * len(names))
    rows = db.conn.execute(
        f"""
        SELECT DISTINCT e.caller_unit_id, e.callee_full, e.line, f.path
        FROM call_edges e JOIN files f ON f.id = e.file_id
        WHERE e.callee IN ({marks}) OR e.callee_full IN ({marks})
        """,
        names + names,
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        got = db.unit_by_id(r["caller_unit_id"])
        if got:
            out.append({"unit": got[0], "path": r["path"], "callee_full": r["callee_full"], "line": r["line"]})
    return out


def transitive_caller_ids(db: Database, name: str, max_depth: int = 3) -> dict[int, int]:
    """BFS over call_edges: unit_id -> shortest hop (1 = direct caller)."""
    seen: dict[int, int] = {}
    frontier: list[str] = [name]
    for hop in range(1, max_depth + 1):
        nxt: list[str] = []
        for callee in frontier:
            for row in _raw_callers(db, [callee]):
                uid = row["unit"].id
                if uid in seen:
                    continue
                seen[uid] = hop
                u = row["unit"]
                nxt.append(u.name)
                if u.qualname and u.qualname != u.name:
                    nxt.append(u.qualname)
        frontier = nxt
        if not frontier:
            break
    return seen


def autogen_transitive_questions(db: Database, n: int, max_depth: int = 3) -> list[Question]:
    """Questions whose gold = ALL transitive callers (hops 1..max_depth)."""
    rows = db.conn.execute("SELECT DISTINCT callee FROM call_edges ORDER BY callee").fetchall()
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
                aliases[m.group(1)] = f"{m.group(2).strip('@').replace('/', '.')}.default"
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
                    real = f"{target}.{c.callee_full[len(prefix):]}"
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
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        out.append(
            Question(
                query=d["query"],
                gold_unit_ids=d.get("gold_unit_ids", []),
                gold_file=d.get("gold_file", ""),
                label=d.get("label", "custom"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(run: SystemRun, q: Question, top_k: int) -> dict:
    gold_ids = set(q.gold_unit_ids)
    gold_file = q.gold_file
    hits = run.hits[:top_k]
    hit_files = {h.file for h in hits}
    hit_unit_ids = {h.unit_id for h in hits if h.unit_id is not None}

    unit_recall = 1.0 if (gold_ids and hit_unit_ids & gold_ids) else 0.0
    file_recall = 1.0 if (gold_file and gold_file in hit_files) else 0.0

    relevant = hit_unit_ids & gold_ids if gold_ids else set()
    precision = len(relevant) / len(hits) if hits else 0.0

    indirect_recall: float | None = None
    if q.gold_hops:
        found = {uid for uid in hit_unit_ids if uid in q.gold_hops}
        indirect = {uid for uid, hop in q.gold_hops.items() if hop >= 2}
        if indirect:
            indirect_recall = len(found & indirect) / len(indirect)

    mrr = 0.0
    if gold_ids:
        for i, h in enumerate(hits[:top_k]):
            if h.unit_id in gold_ids:
                mrr = 1.0 / (i + 1)
                break
    elif gold_file:
        for i, h in enumerate(hits[:top_k]):
            if h.file == gold_file:
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


def aggregate(rows: list[dict]) -> dict:
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
        "p50_sec": lat[int(math.floor(0.50 * n))] if n else 0.0,
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
        resp = client.post(url.rstrip("/") + "/chat/completions", json=payload, headers=headers)
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
) -> dict[str, dict]:
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
            ans = _llm_chat(
                url, model, key,
                [
                    {"role": "system", "content": "You answer questions about a codebase using ONLY the provided context. If the context is insufficient, say so explicitly and mark it insufficient."},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {q.query}\nAnswer concisely."},
                ],
            )
            grade = _llm_chat(
                url, model, key,
                [
                    {"role": "system", "content": "You are an evaluation judge. Given a question, a candidate answer, and the context that was available, respond with ONLY a JSON object {\"correct\": 0-10, \"sufficient\": \"yes\"|\"no\"}. correct = how well the answer answers the question given the available context (not hallucinating)."},
                    {"role": "user", "content": f"QUESTION: {q.query}\n\nANSWER:\n{ans}\n\nCONTEXT_AVAILABLE:\n{context}"},
                ],
            )
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
            ev = db.load_evidence(h.unit_id)
            if ev and "span" in ev:
                parts.append(f"--- {h.file} (unit {h.unit_id}) ---\n{ev['span']}")
            else:
                parts.append(f"- {h.file} (unit id {h.unit_id})")
        else:
            parts.append(f"- {h.file}")
    return "\n".join(parts) if parts else "(no context retrieved)"
