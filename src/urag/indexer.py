"""Indexer: file discovery, incremental update, embedding pipeline."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pathspec

from .config import Config, language_for_path
from .db import Database
from .embed import Embedder
from .extractors import get_extractor
from .git_aware import Git
from .models import SourceFile, Unit

BATCH = 64

Progress = Callable[[str], None]


def _noop(msg: str) -> None:
    pass


class Indexer:
    def __init__(self, cfg: Config, db: Database, embedder: Embedder, progress: Progress = _noop):
        self.cfg = cfg
        self.db = db
        self.embedder = embedder
        self.progress = progress
        self._root = cfg.project_root
        self._specs = self._build_specs()
        self.git = Git(self._root)
        self._head = self.git.head()

    # ---------- discovery ----------

    def _build_specs(self) -> list[pathspec.PathSpec]:
        specs: list[pathspec.PathSpec] = []
        if not self.cfg.index.ignore_gitignore:
            gi = self._root / ".gitignore"
            if gi.exists():
                specs.append(pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(encoding="utf-8", errors="replace").splitlines()))
        excludes = list(self.cfg.index.exclude)
        specs.append(pathspec.PathSpec.from_lines("gitwildmatch", [f"{e}/" for e in excludes]))
        return specs

    def _is_excluded(self, rel: str) -> bool:
        if self.cfg.index.include:
            import fnmatch

            return not any(fnmatch.fnmatch(rel, p) for p in self.cfg.index.include)
        return any(s.match_file(rel) for s in self._specs)

    def discover(self) -> list[Path]:
        out: list[Path] = []
        langs = set(self.cfg.index.languages)
        for p in sorted(self._root.rglob("*")):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(self._root).as_posix()
            except ValueError:
                continue
            if self._is_excluded(rel):
                continue
            try:
                result = language_for_path(p)
            except ValueError:
                continue
            if result is None:
                continue
            lang, kind = result
            if lang not in langs:
                continue
            try:
                if p.stat().st_size > self.cfg.index.max_file_bytes:
                    continue
            except OSError:
                continue
            out.append(p)
        return out

    # ---------- indexing ----------

    def index_all(self) -> dict:
        start = time.monotonic()
        files = self.discover()
        known = self.db.known_paths()
        current = {f.relative_to(self._root).as_posix() for f in files}
        deleted = known - current
        if self._head:
            git_changed, git_deleted = self.git.changed_paths()
            if git_deleted:
                deleted |= {p for p in git_deleted if p in known}
        if deleted:
            self.db.delete_files(sorted(deleted))
            self.progress(f"removed {len(deleted)} deleted file(s)")
        changed = [f for f in files if self._needs_reindex(f)]
        if self._head:
            git_changed, _ = self.git.changed_paths()
            changed = [
                f for f in files
                if self._needs_reindex(f) or f.relative_to(self._root).as_posix() in git_changed
            ]
        new = [f for f in files if f.relative_to(self._root).as_posix() not in known]
        for p in changed:
            self._index_file(p)
        if changed or new:
            self._embed_missing()
        if self._head:
            self.db.set_meta("last_commit", self._head)
        self.db.set_meta("last_indexed_at", datetime.now(timezone.utc).isoformat())
        s = self.db.stats()
        self.progress(
            f"indexed {len(files)} files ({len(changed)} updated), {s.units} units, {s.embedded} embedded "
            f"in {time.monotonic() - start:.1f}s"
        )
        return {"files": len(files), "changed": len(changed), "deleted": len(deleted)}

    def index_paths(self, paths: Iterable[Path]) -> dict:
        """Incremental re-index of specific paths (watcher)."""
        changed: list[Path] = []
        deleted: list[str] = []
        for p in paths:
            try:
                rel = p.relative_to(self._root).as_posix()
            except ValueError:
                continue
            if not p.exists():
                deleted.append(rel)
                continue
            try:
                result = language_for_path(p)
            except ValueError:
                continue
            if result is None:
                continue
            lang, kind = result
            if lang not in self.cfg.index.languages:
                continue
            if self._is_excluded(rel):
                continue
            if self._needs_reindex(p):
                changed.append(p)
        if deleted:
            self.db.delete_files(deleted)
        for p in changed:
            self._index_file(p)
        if changed:
            self._embed_missing()
            if self._head:
                self.db.set_meta("last_commit", self._head)
            self.db.set_meta("last_indexed_at", datetime.now(timezone.utc).isoformat())
        return {"changed": len(changed), "deleted": len(deleted)}

    def _needs_reindex(self, p: Path) -> bool:
        try:
            st = p.stat()
        except OSError:
            return False
        row = self.db.conn.execute("SELECT size, mtime FROM files WHERE path = ?", (p.relative_to(self._root).as_posix(),)).fetchone()
        if row is None:
            return True
        return row["size"] != st.st_size or abs(row["mtime"] - st.st_mtime) > 1e-6

    def _index_file(self, p: Path) -> None:
        rel = p.relative_to(self._root).as_posix()
        try:
            st = p.stat()
            data = p.read_bytes()
        except OSError:
            return
        lang, kind = language_for_path(p)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        extractor = get_extractor(lang)
        units = extractor.extract(text, rel) if extractor else []
        f = SourceFile(
            path=rel,
            kind=kind,
            language=lang,
            size=st.st_size,
            mtime=st.st_mtime,
            sha256=hashlib.sha256(data).hexdigest(),
            commit=self._head or "",
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        file_id = self.db.upsert_file(f)
        self.db.replace_units(file_id, units)
        if extractor and units:
            calls = extractor.collect_calls(text)
            edges = self._map_calls(units, calls)
            self.db.replace_call_edges(file_id, edges)
            aliases = extractor.collect_import_aliases(text)
            if aliases:
                self.db.replace_import_aliases(file_id, aliases)
        self.progress(f"  {rel}: {len(units)} units")

    # ---------- embeddings ----------

    def _units_missing_embeddings(self) -> list[Unit]:
        rows = self.db.conn.execute(
            """
            SELECT u.* FROM units u
            WHERE u.id NOT IN (SELECT unit_id FROM vec_units)
              AND (u.summary != '' OR u.concepts != '' OR u.qualname != '' OR u.signature != '')
            ORDER BY u.id
            """
        ).fetchall()
        return [Database._row_to_unit(r) for r in rows]

    def _embed_missing(self) -> int:
        units = self._units_missing_embeddings()
        if not units:
            return 0
        lang_rows = dict(
            self.db.conn.execute("SELECT id, language FROM files").fetchall()
        )
        n = 0
        for i in range(0, len(units), BATCH):
            batch = units[i : i + BATCH]
            texts = [u.retrieval_key for u in batch]
            vecs = self.embedder.embed_passages(texts)
            rows = []
            for u, v in zip(batch, vecs):
                rows.append((u.id, lang_rows.get(u.file_id, ""), u.kind, v))
            self.db.store_embeddings(rows)
            n += len(batch)
            self.progress(f"  embedded {n}/{len(units)}")
        return n

    # ---------- call graph ----------

    def _map_calls(self, units: list[Unit], calls: list) -> list[tuple[int, str, str, int]]:
        """Map call sites to the innermost enclosing symbol unit."""
        funcs = sorted(
            [u for u in units if u.is_symbol and u.unit_type != "import" and u.id is not None],
            key=lambda u: u.byte_start,
        )
        edges: list[tuple[int, str, str, int]] = []
        starts = [u.byte_start for u in funcs]
        import bisect

        for c in calls:
            idx = bisect.bisect_right(starts, c.byte_start) - 1
            if idx >= 0:
                caller = funcs[idx]
                if caller.byte_start <= c.byte_start <= caller.byte_end:
                    edges.append((caller.id, c.callee, c.callee_full, c.line))
        return edges

    # ---------- queries ----------

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.embed_query(query)
