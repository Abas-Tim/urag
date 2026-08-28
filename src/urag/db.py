"""SQLite storage: files, units, FTS5 lexical index, sqlite-vec dense index.

Single-file database per project inside .urag/index.db. Fully incremental:
file mtimes decide what needs re-extraction.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import sqlite_vec

from .db_edges import _CallGraphMixin
from .db_search import _SearchMixin
from .models import IndexStats, SourceFile, Unit

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
  file_path TEXT NOT NULL DEFAULT '',
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
CREATE INDEX IF NOT EXISTS idx_units_unit_type ON units(unit_type);
CREATE TABLE IF NOT EXISTS call_edges (
  caller_unit_id INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  callee TEXT NOT NULL,
  callee_full TEXT NOT NULL DEFAULT '',
  line INTEGER NOT NULL,
  callee_unit_id INTEGER REFERENCES units(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(callee);
CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_unit_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee_full ON call_edges(callee_full);
CREATE TABLE IF NOT EXISTS ref_edges (
  unit_id INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  ref TEXT NOT NULL,
  ref_full TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT '',
  line INTEGER NOT NULL,
  ref_unit_id INTEGER REFERENCES units(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_ref_edges_ref ON ref_edges(ref);
CREATE INDEX IF NOT EXISTS idx_ref_edges_unit ON ref_edges(unit_id);
CREATE INDEX IF NOT EXISTS idx_ref_edges_ref_full ON ref_edges(ref_full);
CREATE TABLE IF NOT EXISTS import_aliases (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  target TEXT NOT NULL,
  PRIMARY KEY (file_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_import_aliases_target ON import_aliases(target);
CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
"""

FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS units_ai AFTER INSERT ON units BEGIN
  INSERT INTO fts_units(rowid, name, qualname, signature, summary, concepts, relationships, unit_type, file_path)
  VALUES (new.id, new.name, new.qualname, new.signature, new.summary, new.concepts, new.relationships, new.unit_type,
          (SELECT path FROM files WHERE id = new.file_id));
END;
CREATE TRIGGER IF NOT EXISTS units_ad AFTER DELETE ON units BEGIN
  INSERT INTO fts_units(fts_units, rowid, name, qualname, signature, summary, concepts, relationships, unit_type, file_path)
  VALUES ('delete', old.id, old.name, old.qualname, old.signature, old.summary, old.concepts, old.relationships, old.unit_type,
          (SELECT path FROM files WHERE id = old.file_id));
END;
CREATE TRIGGER IF NOT EXISTS units_au AFTER UPDATE ON units BEGIN
  INSERT INTO fts_units(fts_units, rowid, name, qualname, signature, summary, concepts, relationships, unit_type, file_path)
  VALUES ('delete', old.id, old.name, old.qualname, old.signature, old.summary, old.concepts, old.relationships, old.unit_type,
          (SELECT path FROM files WHERE id = old.file_id));
  INSERT INTO fts_units(rowid, name, qualname, signature, summary, concepts, relationships, unit_type, file_path)
  VALUES (new.id, new.name, new.qualname, new.signature, new.summary, new.concepts, new.relationships, new.unit_type,
          (SELECT path FROM files WHERE id = new.file_id));
END;
"""


class Database(_SearchMixin, _CallGraphMixin):
    def __init__(self, db_path: Path, dimension: int = 384, migrate: bool = False):
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.migrate = migrate
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
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        c = self.conn
        c.executescript(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(files)")}
        if "commit" not in cols:  # migration for pre-git-aware indexes
            c.execute(
                "ALTER TABLE files ADD COLUMN \"commit\" TEXT NOT NULL DEFAULT ''"
            )
        unit_cols = {r[1] for r in c.execute("PRAGMA table_info(units)")}
        added_unit_file_path = False
        if "file_path" not in unit_cols:
            c.execute("ALTER TABLE units ADD COLUMN file_path TEXT NOT NULL DEFAULT ''")
            c.execute(
                "UPDATE units SET file_path = (SELECT path FROM files WHERE files.id = units.file_id)"
            )
            added_unit_file_path = True
        call_edge_cols = {r[1] for r in c.execute("PRAGMA table_info(call_edges)")}
        if "callee_unit_id" not in call_edge_cols:
            c.execute("ALTER TABLE call_edges ADD COLUMN callee_unit_id INTEGER")
        fts_columns = {
            "name",
            "qualname",
            "signature",
            "summary",
            "concepts",
            "relationships",
            "unit_type",
            "file_path",
        }
        fts_exists = (
            c.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fts_units'"
            ).fetchone()
            is not None
        )
        rebuild_fts = added_unit_file_path or not fts_exists
        if fts_exists:
            current_fts_columns = {
                r[1] for r in c.execute("PRAGMA table_info(fts_units)")
            }
            if current_fts_columns != fts_columns:
                c.execute("DROP TABLE fts_units")
                rebuild_fts = True
        vector_schema = c.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_units'"
        ).fetchone()
        if vector_schema:
            match = re.search(r"FLOAT\[(\d+)\]", vector_schema[0] or "", re.IGNORECASE)
            if match and int(match.group(1)) != self.dimension:
                if not self.migrate:
                    raise RuntimeError(
                        f"embedding dimension mismatch: the index stores "
                        f"{match.group(1)}-dimensional vectors but embedding.dimension "
                        f"is {self.dimension}. Restore the original dimension or run "
                        f"`urag embed --dimension {self.dimension} --reindex` to "
                        f"rebuild embeddings."
                    )
                c.execute("DROP TABLE vec_units")
        c.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_units USING fts5(
              name, qualname, signature, summary, concepts, relationships, unit_type, file_path,
              content='units', content_rowid='id',
              tokenize='unicode61 remove_diacritics 2');
        """)
        c.executescript(
            "DROP TRIGGER IF EXISTS units_ai;"
            "DROP TRIGGER IF EXISTS units_ad;"
            "DROP TRIGGER IF EXISTS units_au;"
        )
        c.executescript(FTS_TRIGGERS)
        c.executescript(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_units USING vec0(
              unit_id INTEGER PRIMARY KEY,
              language TEXT,
              kind TEXT,
              embedding FLOAT[{self.dimension}]
            );
        """)
        if self.migrate:
            c.execute(
                "DELETE FROM vec_units WHERE unit_id NOT IN (SELECT id FROM units)"
            )
            c.execute(
                """
                UPDATE call_edges SET callee_unit_id = NULL
                WHERE callee_unit_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM units WHERE units.id = call_edges.callee_unit_id
                  )
                """
            )
            c.execute(
                """
                UPDATE ref_edges SET ref_unit_id = NULL
                WHERE ref_unit_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM units WHERE units.id = ref_edges.ref_unit_id
                  )
                """
            )
        if rebuild_fts:
            c.execute("INSERT INTO fts_units(fts_units) VALUES ('rebuild')")
        if not self.get_meta("ref_edges_v1"):
            self.set_meta("ref_edges_v1", "1")
            if c.execute("SELECT COUNT(*) FROM units").fetchone()[0]:
                self.set_meta("refs_pending", "1")
        c.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def delete_meta(self, key: str) -> None:
        self.conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        self.conn.commit()

    # ---------- files ----------

    def known_paths(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT path FROM files")}

    def upsert_file(self, f: SourceFile, commit: bool = True) -> int:
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
        if commit:
            self.conn.commit()
        return file_id

    def delete_files(self, paths: list[str]) -> None:
        if not paths:
            return
        for path in paths:
            self.conn.execute(
                """
                UPDATE call_edges SET callee_unit_id = NULL
                WHERE callee_unit_id IN (
                    SELECT id FROM units WHERE file_id = (
                        SELECT id FROM files WHERE path = ?
                    )
                )
                """,
                (path,),
            )
            self.conn.execute(
                """
                UPDATE ref_edges SET ref_unit_id = NULL
                WHERE ref_unit_id IN (
                    SELECT id FROM units WHERE file_id = (
                        SELECT id FROM files WHERE path = ?
                    )
                )
                """,
                (path,),
            )
            self.conn.execute(
                "DELETE FROM vec_units WHERE unit_id IN ("
                "SELECT id FROM units WHERE file_id = (SELECT id FROM files WHERE path = ?)"
                ")",
                (path,),
            )
            self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()

    # ---------- units ----------

    def replace_units(
        self, file_id: int, units: list[Unit], commit: bool = True
    ) -> None:
        """Replace all units of a file."""
        existing = self.conn.execute(
            "SELECT id, kind, unit_type, qualname FROM units WHERE file_id = ? ORDER BY id",
            (file_id,),
        ).fetchall()
        reusable: dict[tuple[str, str, str], list[int]] = {}
        for row in existing:
            key = (row["kind"], row["unit_type"], row["qualname"])
            reusable.setdefault(key, []).append(row["id"])
        self.conn.execute(
            "DELETE FROM vec_units WHERE unit_id IN (SELECT id FROM units WHERE file_id = ?)",
            (file_id,),
        )
        self.conn.execute(
            "UPDATE call_edges SET callee_unit_id = NULL "
            "WHERE callee_unit_id IN (SELECT id FROM units WHERE file_id = ?)",
            (file_id,),
        )
        self.conn.execute(
            "UPDATE ref_edges SET ref_unit_id = NULL "
            "WHERE ref_unit_id IN (SELECT id FROM units WHERE file_id = ?)",
            (file_id,),
        )
        self.conn.execute("DELETE FROM units WHERE file_id = ?", (file_id,))
        planned: list[tuple[Unit, int | None]] = []
        for u in units:
            key = (u.kind, u.unit_type, u.qualname)
            ids = reusable.get(key, [])
            planned.append((u, ids.pop(0) if ids else None))

        for u, stable_id in [item for item in planned if item[1] is not None] + [
            item for item in planned if item[1] is None
        ]:
            values = (
                file_id,
                u.kind,
                u.unit_type,
                u.name,
                u.qualname,
                u.signature,
                u.summary,
                u.concepts,
                u.relationships,
                file_id,
                u.start_line,
                u.end_line,
                u.start_col,
                u.end_col,
                u.byte_start,
                u.byte_end,
                u.parent_id,
            )
            if stable_id is None:
                cur = self.conn.execute(
                    """
                    INSERT INTO units(file_id, kind, unit_type, name, qualname, signature,
                                      summary, concepts, relationships, file_path, start_line, end_line,
                                      start_col, end_col, byte_start, byte_end, parent_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, (SELECT path FROM files WHERE id = ?), ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            else:
                cur = self.conn.execute(
                    """
                    INSERT INTO units(id, file_id, kind, unit_type, name, qualname, signature,
                                      summary, concepts, relationships, file_path, start_line, end_line,
                                      start_col, end_col, byte_start, byte_end, parent_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, (SELECT path FROM files WHERE id = ?), ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (stable_id, *values),
                )
            u.id = cur.lastrowid
            u.file_id = file_id
        for child in units:
            parents = [
                parent
                for parent in units
                if parent.id != child.id
                and parent.byte_start <= child.byte_start
                and parent.byte_end >= child.byte_end
                and (parent.byte_end - parent.byte_start)
                > (child.byte_end - child.byte_start)
            ]
            if parents:
                parent = min(parents, key=lambda item: item.byte_end - item.byte_start)
                child.parent_id = parent.id
                self.conn.execute(
                    "UPDATE units SET parent_id = ? WHERE id = ?", (parent.id, child.id)
                )
        if commit:
            self.conn.commit()

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
        for r in self.conn.execute(
            "SELECT language, COUNT(*) c FROM files GROUP BY language"
        ):
            s.by_language[r["language"]] = r["c"]
        return s


    def store_embeddings(
        self, units: list[tuple[int, str, str, list[float]]], commit: bool = True
    ) -> None:
        """Store embeddings: (unit_id, language, kind, vector) pairs."""
        self.conn.executemany(
            "INSERT INTO vec_units(unit_id, language, kind, embedding) VALUES (?, ?, ?, ?)",
            [(uid, lang, kind, json.dumps(vec)) for uid, lang, kind, vec in units],
        )
        if commit:
            self.conn.commit()

    def clear_embeddings(self) -> None:
        self.conn.execute("DELETE FROM vec_units")
        self.conn.commit()


    # ---------- evidence (L2) ----------

    def load_evidence(self, unit_id: int) -> dict | None:
        """Load the exact source span for a unit from the file on disk."""
        got = self.unit_by_id(unit_id)
        if not got:
            return None
        u, path, commit = got
        file_row = self.conn.execute(
            "SELECT sha256 FROM files WHERE path = ?", (path,)
        ).fetchone()
        indexed_sha256 = file_row["sha256"] if file_row else ""
        p = Path(path)
        if not p.is_absolute():
            p = self.db_path.parent.parent / p
        if not p.exists():
            return {
                "unit_id": u.id,
                "file": path,
                "error": f"file missing: {path}",
                "commit": commit,
                "indexed_sha256": indexed_sha256,
            }
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        lines = text.splitlines()
        first = max(u.start_line, 1)
        span = lines[first - 1 : u.end_line]
        return {
            "unit_id": u.id,
            "file": path,
            "lines": [u.start_line, u.end_line],
            "language": p.suffix.lstrip("."),
            "commit": commit,
            "indexed_sha256": indexed_sha256,
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