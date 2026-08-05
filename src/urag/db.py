"""SQLite storage: files, units, FTS5 lexical index, sqlite-vec dense index.

Single-file database per project inside .urag/index.db. Fully incremental:
file mtimes decide what needs re-extraction.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

from .models import IndexStats, SourceFile, Unit, RetrievedUnit

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL,
  language TEXT NOT NULL DEFAULT '',
  size INTEGER NOT NULL DEFAULT 0,
  mtime REAL NOT NULL DEFAULT 0,
  sha256 TEXT NOT NULL DEFAULT '',
  "commit" TEXT NOT NULL DEFAULT '',
  indexed_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS units (
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  unit_type TEXT NOT NULL,
  name TEXT NOT NULL,
  qualname TEXT NOT NULL DEFAULT '',
  signature TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  concepts TEXT NOT NULL DEFAULT '',
  relationships TEXT NOT NULL DEFAULT '',
  start_line INTEGER NOT NULL DEFAULT 0,
  end_line INTEGER NOT NULL DEFAULT 0,
  start_col INTEGER NOT NULL DEFAULT 0,
  end_col INTEGER NOT NULL DEFAULT 0,
  byte_start INTEGER NOT NULL DEFAULT 0,
  byte_end INTEGER NOT NULL DEFAULT 0,
  parent_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_units_file ON units(file_id);
CREATE INDEX IF NOT EXISTS idx_units_name ON units(name);
CREATE INDEX IF NOT EXISTS idx_units_qualname ON units(qualname);
CREATE TABLE IF NOT EXISTS call_edges (
  caller_unit_id INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  callee TEXT NOT NULL,
  callee_full TEXT NOT NULL DEFAULT '',
  line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(callee);
CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_unit_id);
"""

FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS units_ai AFTER INSERT ON units BEGIN
  INSERT INTO fts_units(rowid, name, qualname, signature, summary, concepts, unit_type, file_path)
  VALUES (new.id, new.name, new.qualname, new.signature, new.summary, new.concepts, new.unit_type,
          (SELECT path FROM files WHERE id = new.file_id));
END;
CREATE TRIGGER IF NOT EXISTS units_ad AFTER DELETE ON units BEGIN
  INSERT INTO fts_units(fts_units, rowid, name, qualname, signature, summary, concepts, unit_type, file_path)
  VALUES ('delete', old.id, old.name, old.qualname, old.signature, old.summary, old.concepts, old.unit_type,
          (SELECT path FROM files WHERE id = old.file_id));
END;
CREATE TRIGGER IF NOT EXISTS units_au AFTER UPDATE ON units BEGIN
  INSERT INTO fts_units(fts_units, rowid, name, qualname, signature, summary, concepts, unit_type, file_path)
  VALUES ('delete', old.id, old.name, old.qualname, old.signature, old.summary, old.concepts, old.unit_type,
          (SELECT path FROM files WHERE id = old.file_id));
  INSERT INTO fts_units(rowid, name, qualname, signature, summary, concepts, unit_type, file_path)
  VALUES (new.id, new.name, new.qualname, new.signature, new.summary, new.concepts, new.unit_type,
          (SELECT path FROM files WHERE id = new.file_id));
END;
"""


class Database:
    def __init__(self, db_path: Path, dimension: int = 384):
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the watch daemon re-indexes from a
        # debounce thread. The connection is still only ever used by one
        # thread at a time (timer thread in watch mode, main thread in CLI).
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Python 3.12+ removed Connection.enable_load_extension; use the
        # module-level API there. Some CPython builds (e.g. the GitHub macOS
        # runner's framework build) lack extension loading entirely.
        try:
            if hasattr(self.conn, "enable_load_extension"):
                self.conn.enable_load_extension(True)
            else:
                sqlite3.enable_load_extension(self.conn, True)
            sqlite_vec.load(self.conn)
        except (AttributeError, sqlite3.OperationalError) as exc:
            raise RuntimeError(
                "sqlite-vec could not be loaded: this Python's sqlite3 module was "
                "built without loadable-extension support. Use a uv-managed Python "
                "(`uv python install 3.12` + a `.python-version` file) or a build "
                "compiled with --enable-loadable-sqlite-extensions."
            ) from exc
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        c = self.conn
        c.executescript(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(files)")}
        if "commit" not in cols:  # migration for pre-git-aware indexes
            c.execute('ALTER TABLE files ADD COLUMN "commit" TEXT NOT NULL DEFAULT \'\'')
        c.executescript(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_units USING fts5(
              name, qualname, signature, summary, concepts, unit_type, file_path,
              content='units', content_rowid='id',
              tokenize='unicode61 remove_diacritics 2');
        """)
        c.executescript(FTS_TRIGGERS)
        c.executescript(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_units USING vec0(
              unit_id INTEGER PRIMARY KEY,
              language TEXT,
              kind TEXT,
              embedding FLOAT[{self.dimension}]
            );
        """)
        c.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    # ---------- files ----------

    def all_files(self) -> dict[str, SourceFile]:
        rows = self.conn.execute("SELECT * FROM files").fetchall()
        return {r["path"]: SourceFile(**dict(r)) for r in rows}

    def known_paths(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT path FROM files")}

    def upsert_file(self, f: SourceFile) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO files(path, kind, language, size, mtime, sha256, "commit", indexed_at)
            VALUES (:path, :kind, :language, :size, :mtime, :sha256, :commit, :indexed_at)
            ON CONFLICT(path) DO UPDATE SET
              kind = excluded.kind, language = excluded.language,
              size = excluded.size, mtime = excluded.mtime,
              sha256 = excluded.sha256, "commit" = excluded."commit",
              indexed_at = excluded.indexed_at
            RETURNING id
            """,
            {
                "path": f.path,
                "kind": f.kind,
                "language": f.language,
                "size": f.size,
                "mtime": f.mtime,
                "sha256": f.sha256,
                "commit": f.commit,
                "indexed_at": f.indexed_at,
            },
        )
        file_id = cur.fetchone()[0]
        self.conn.commit()
        return file_id

    def delete_files(self, paths: list[str]) -> None:
        if not paths:
            return
        self.conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])
        self.conn.commit()

    # ---------- units ----------

    def replace_units(self, file_id: int, units: list[Unit]) -> None:
        """Replace all units of a file."""
        self.conn.execute("DELETE FROM vec_units WHERE unit_id IN (SELECT id FROM units WHERE file_id = ?)", (file_id,))
        self.conn.execute("DELETE FROM units WHERE file_id = ?", (file_id,))
        for u in units:
            cur = self.conn.execute(
                """
                INSERT INTO units(file_id, kind, unit_type, name, qualname, signature,
                                  summary, concepts, relationships, start_line, end_line,
                                  start_col, end_col, byte_start, byte_end, parent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id, u.kind, u.unit_type, u.name, u.qualname, u.signature,
                    u.summary, u.concepts, u.relationships, u.start_line, u.end_line,
                    u.start_col, u.end_col, u.byte_start, u.byte_end, u.parent_id,
                ),
            )
            u.id = cur.lastrowid
            u.file_id = file_id
        self.conn.commit()

    def units_for_file(self, file_id: int) -> list[Unit]:
        rows = self.conn.execute("SELECT * FROM units WHERE file_id = ?", (file_id,)).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def unit_by_id(self, unit_id: int) -> tuple[Unit, str, str] | None:
        """(unit, path, commit) for a unit id."""
        row = self.conn.execute(
            'SELECT u.*, f.path, f."commit" FROM units u JOIN files f ON f.id = u.file_id WHERE u.id = ?',
            (unit_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_unit(row), row["path"], row["commit"]

    def counts(self) -> tuple[int, int]:
        files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        units = self.conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        return files, units

    def stats(self) -> IndexStats:
        s = IndexStats()
        s.files, s.units = self.counts()
        s.embedded = self.conn.execute("SELECT COUNT(*) FROM vec_units").fetchone()[0]
        s.size_bytes = self.db_path.stat().st_size
        s.last_indexed = self.get_meta("last_indexed_at")
        for r in self.conn.execute("SELECT language, COUNT(*) c FROM files GROUP BY language"):
            s.by_language[r["language"]] = r["c"]
        return s

    # ---------- retrieval ----------

    def lexical_search(self, query: str, limit: int = 30, language: str | None = None) -> list[tuple[Unit, str, float]]:
        q = self._safe_fts(query.strip())
        if not q:
            return []
        rows = self.conn.execute(
            """
            SELECT u.*, f.path, bm25(fts_units) AS score
            FROM fts_units
            JOIN units u ON u.id = fts_units.rowid
            JOIN files f ON f.id = u.file_id
            WHERE fts_units MATCH ? AND (? = '' OR f.language = ?)
            ORDER BY score
            LIMIT ?
            """,
            (q, language or "", language or "", limit),
        ).fetchall()
        return [(self._row_to_unit(r), r["path"], r["score"]) for r in rows]

    @staticmethod
    def _safe_fts(query: str) -> str:
        """Make a user query safe as an FTS5 MATCH expression.

        Bare `:` would be parsed as a column filter (`Runner::run` -> column
        "Runner"), and AND/OR/NOT are operator keywords. Quote every term.
        """
        import re

        terms = [t for t in re.split(r"[^A-Za-z0-9_.#]+", query) if t]
        return " ".join(f'"{t}"' for t in terms)

    def dense_search(
        self, embedding: list[float], limit: int = 30, language: str | None = None
    ) -> list[tuple[Unit, str, float]]:
        vec_json = json.dumps(embedding)
        q = "SELECT unit_id, distance FROM vec_units WHERE embedding MATCH ? "
        params: list = [vec_json]
        if language:
            q += "AND language = ? "
            params.append(language)
        q += f"ORDER BY distance LIMIT {limit}"
        rows = self.conn.execute(q, params).fetchall()
        out: list[tuple[Unit, str, float]] = []
        for r in rows:
            got = self.unit_by_id(r["unit_id"])
            if got:
                u, path, _commit = got
                out.append((u, path, r["distance"]))
        return out

    def unit_embedding(self, unit_id: int) -> list[float] | None:
        row = self.conn.execute(
            "SELECT vec_f32(embedding) AS v FROM vec_units WHERE unit_id = ?", (unit_id,)
        ).fetchone()
        return list(row["v"]) if row else None

    def store_embeddings(self, units: list[tuple[int, str, str, list[float]]]) -> None:
        """Store embeddings: (unit_id, language, kind, vector) pairs."""
        for uid, lang, kind, vec in units:
            self.conn.execute(
                "INSERT INTO vec_units(unit_id, language, kind, embedding) VALUES (?, ?, ?, ?)",
                (uid, lang, kind, json.dumps(vec)),
            )
        self.conn.commit()

    def delete_embedding(self, unit_id: int) -> None:
        self.conn.execute("DELETE FROM vec_units WHERE unit_id = ?", (unit_id,))
        self.conn.commit()

    # ---------- call graph ----------

    def replace_call_edges(self, file_id: int, edges: list[tuple[int, str, str, int]]) -> None:
        """Replace all call edges of a file: (caller_unit_id, callee, full, line)."""
        self.conn.execute(
            "DELETE FROM call_edges WHERE file_id = ?", (file_id,)
        )
        self.conn.executemany(
            "INSERT INTO call_edges(caller_unit_id, file_id, callee, callee_full, line) VALUES (?, ?, ?, ?, ?)",
            [(cid, file_id, callee, full, line) for cid, callee, full, line in edges],
        )
        self.conn.commit()

    def callers(self, name: str, limit: int = 30) -> list[dict]:
        """Units that call `name` (matches last segment or full chain)."""
        name = name.strip()
        if not name:
            return []
        import re

        esc = re.escape(name).replace("%", r"\%").replace("_", r"\_")
        rows = self.conn.execute(
            """
            SELECT u.*, f.path, e.callee_full, e.line
            FROM call_edges e
            JOIN units u ON u.id = e.caller_unit_id
            JOIN files f ON f.id = e.file_id
            WHERE e.callee = ? OR e.callee_full = ? OR e.callee_full LIKE ? ESCAPE '\\'
               OR e.callee_full LIKE ? ESCAPE '\\'
            ORDER BY CASE WHEN f.path LIKE '%test%' THEN 1 ELSE 0 END, e.line
            LIMIT ?
            """,
            (name, name, f"%.{esc}", f"%::{esc}", limit),
        ).fetchall()
        out = []
        for r in rows:
            u = self._row_to_unit(r)
            out.append(
                {
                    "unit": u,
                    "path": r["path"],
                    "callee_full": r["callee_full"],
                    "line": r["line"],
                }
            )
        return out

    def callees(self, unit_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT callee, callee_full, line FROM call_edges WHERE caller_unit_id = ? ORDER BY line",
            (unit_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- evidence (L2) ----------

    def load_evidence(self, unit_id: int) -> dict | None:
        """Load the exact source span for a unit from the file on disk."""
        got = self.unit_by_id(unit_id)
        if not got:
            return None
        u, path, commit = got
        p = Path(path)
        if not p.is_absolute():
            p = self.db_path.parent.parent / p
        if not p.exists():
            return {"unit_id": u.id, "error": f"file missing: {path}", "commit": commit}
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        lines = text.splitlines()
        span = lines[u.start_line - 1 : u.end_line]
        return {
            "unit_id": u.id,
            "file": path,
            "lines": [u.start_line, u.end_line],
            "language": p.suffix.lstrip("."),
            "commit": commit,
            "span": "\n".join(span),
        }

    @staticmethod
    def _row_to_unit(r: sqlite3.Row) -> Unit:
        return Unit(
            id=r["id"],
            file_id=r["file_id"],
            kind=r["kind"],
            unit_type=r["unit_type"],
            name=r["name"],
            qualname=r["qualname"],
            signature=r["signature"],
            summary=r["summary"],
            concepts=r["concepts"],
            relationships=r["relationships"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            start_col=r["start_col"],
            end_col=r["end_col"],
            byte_start=r["byte_start"],
            byte_end=r["byte_end"],
            parent_id=r["parent_id"],
        )

    def close(self) -> None:
        self.conn.close()
