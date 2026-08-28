"""Indexer: file discovery, incremental update, embedding pipeline."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import pathspec

from .config import Config, language_for_path
from .db import Database
from .embed import Embedder
from .extractors import get_extractor
from .git_aware import Git
from .models import SourceFile, Unit

BATCH = 64

_WRITE_LOCK = threading.Lock()

# Bump when extraction semantics change; existing indexes are fully
# re-extracted (embeddings for unchanged retrieval keys are preserved).
EXTRACTOR_VERSION = "2"

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
        self._changed_symbol_names: set[str] = set()
        self._git_dirty: set[str] | None = None
        fingerprint = self.cfg.embedding.fingerprint()
        if self.db.get_meta("embedding_fingerprint") != fingerprint:
            self.db.clear_embeddings()
            self.db.set_meta("embedding_fingerprint", fingerprint)

    # ---------- discovery ----------

    def _build_specs(self) -> list[pathspec.PathSpec]:
        specs: list[pathspec.PathSpec] = []
        if not self.cfg.index.ignore_gitignore:
            gi = self._root / ".gitignore"
            if gi.exists():
                specs.append(
                    pathspec.GitIgnoreSpec.from_lines(
                        gi.read_text(encoding="utf-8", errors="replace").splitlines(),
                    )
                )
        excludes = list(self.cfg.index.exclude)
        specs.append(pathspec.PathSpec.from_lines("gitignore", [f"{e}/" for e in excludes]))
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
            lang, _kind = result
            if lang not in langs and not (lang == "tsx" and "typescript" in langs):
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
        with _WRITE_LOCK:
            return self._index_all()

    def _refresh_git_dirty(self) -> None:
        if self._head and self.git.is_repo():
            changed, _, untracked = self.git.working_paths()
            self._git_dirty = changed | untracked
        else:
            self._git_dirty = None

    def _index_all(self) -> dict:
        start = time.monotonic()
        self._head = self.git.head(refresh=True)
        self._changed_symbol_names.clear()
        self._refresh_git_dirty()
        files = self.discover()
        known = self.db.known_paths()
        current = {f.relative_to(self._root).as_posix() for f in files}
        deleted = known - current
        if self._head:
            _, git_deleted = self.git.changed_paths()
            if git_deleted:
                deleted |= {p for p in git_deleted if p in known}
        if deleted:
            self.db.delete_files(sorted(deleted))
            self.progress(f"removed {len(deleted)} deleted file(s)")
        force = False
        refs_pending = self.db.get_meta("refs_pending") == "1"
        if refs_pending:
            self.progress("upgrading index: extracting reference edges")
            force = True
        extractor_stale = self.db.get_meta("extractor_version") != EXTRACTOR_VERSION
        if extractor_stale:
            self.progress("upgrading index: extractor changed, re-extracting")
            force = True
        changed = files if force else [f for f in files if self._needs_reindex(f)]
        changed_file_ids: set[int] = set()
        for p in changed:
            file_id = self._index_file(p)
            if file_id is not None:
                changed_file_ids.add(file_id)
        self._resolve_call_edges(changed_file_ids)
        self._embed_missing()
        if refs_pending:
            self.db.delete_meta("refs_pending")
        if extractor_stale:
            self.db.set_meta("extractor_version", EXTRACTOR_VERSION)
        if self._head:
            self.db.set_meta("last_commit", self._head)
        self.db.set_meta("last_indexed_at", datetime.now(UTC).isoformat())
        s = self.db.stats()
        self.progress(
            f"indexed {len(files)} files ({len(changed)} updated), {s.units} units, {s.embedded} embedded "
            f"in {time.monotonic() - start:.1f}s"
        )
        return {"files": len(files), "changed": len(changed), "deleted": len(deleted)}

    def index_paths(self, paths: Iterable[Path]) -> dict:
        with _WRITE_LOCK:
            return self._index_paths(paths)

    def _index_paths(self, paths: Iterable[Path]) -> dict:
        """Incremental re-index of specific paths (watcher)."""
        self._head = self.git.head(refresh=True)
        self._changed_symbol_names.clear()
        self._refresh_git_dirty()
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
            lang, _kind = result
            if lang not in self.cfg.index.languages and not (
                lang == "tsx" and "typescript" in self.cfg.index.languages
            ):
                continue
            if self._is_excluded(rel):
                continue
            try:
                if p.stat().st_size > self.cfg.index.max_file_bytes:
                    continue
            except OSError:
                continue
            if self._needs_reindex(p):
                changed.append(p)
        if deleted:
            self.db.delete_files(deleted)
        changed_file_ids: set[int] = set()
        for p in changed:
            file_id = self._index_file(p)
            if file_id is not None:
                changed_file_ids.add(file_id)
        self._resolve_call_edges(changed_file_ids)
        self._embed_missing()
        if self._head:
            self.db.set_meta("last_commit", self._head)
        self.db.set_meta("last_indexed_at", datetime.now(UTC).isoformat())
        return {"changed": len(changed), "deleted": len(deleted)}

    def _needs_reindex(self, p: Path) -> bool:
        try:
            st = p.stat()
        except OSError:
            return False
        rel = p.relative_to(self._root).as_posix()
        row = self.db.conn.execute(
            "SELECT size, mtime, sha256 FROM files WHERE path = ?",
            (rel,),
        ).fetchone()
        if row is None:
            return True
        if row["size"] != st.st_size or abs(row["mtime"] - st.st_mtime) > 1e-6:
            return True
        if self._git_dirty is None or rel in self._git_dirty:
            try:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                return False
            return digest != row["sha256"]
        return False

    def _index_file(self, p: Path) -> int | None:
        rel = p.relative_to(self._root).as_posix()
        try:
            st = p.stat()
            data = p.read_bytes()
        except OSError:
            return None
        result = language_for_path(p)
        if result is None:
            return None
        lang, kind = result
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        extractor = get_extractor(lang)
        units = extractor.extract(text, rel) if extractor else []
        existing = self.db.conn.execute("SELECT id FROM files WHERE path = ?", (rel,)).fetchone()
        if existing:
            old_units = self.db.conn.execute(
                "SELECT name, qualname FROM units WHERE file_id = ? AND kind = 'symbol' "
                "AND unit_type NOT IN ('import', 'config_key')",
                (existing["id"],),
            ).fetchall()
            for unit in old_units:
                self._changed_symbol_names.update(
                    value for value in (unit["name"], unit["qualname"]) if value
                )
        for unit in units:
            if unit.is_symbol and unit.unit_type not in ("import", "config_key"):
                self._changed_symbol_names.update(
                    value for value in (unit.name, unit.qualname) if value
                )
        f = SourceFile(
            path=rel,
            kind=kind,
            language=lang,
            size=st.st_size,
            mtime=st.st_mtime,
            sha256=hashlib.sha256(data).hexdigest(),
            commit=self._head or "",
            indexed_at=datetime.now(UTC).isoformat(),
        )
        try:
            file_id = self.db.upsert_file(f, commit=False)
            self.db.replace_units(file_id, units, commit=False)
            if extractor:
                calls = extractor.collect_calls(text)
                edges = self._map_calls(units, calls)
                self.db.replace_call_edges(file_id, edges, commit=False)
                aliases = extractor.collect_import_aliases(text)
                self.db.replace_import_aliases(file_id, aliases, commit=False)
                refs = extractor.collect_references(text)
                ref_edges = self._map_refs(units, refs)
                self.db.replace_ref_edges(file_id, ref_edges, commit=False)
            self.db.conn.commit()
        except Exception:
            self.db.conn.rollback()
            raise
        self.progress(f"  {rel}: {len(units)} units")
        return file_id

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
        if self.embedder.dimension <= 0:
            return 0
        units = self._units_missing_embeddings()
        if not units:
            return 0
        lang_rows = dict(self.db.conn.execute("SELECT id, language FROM files").fetchall())
        path_rows = dict(self.db.conn.execute("SELECT id, path FROM files").fetchall())
        parent_rows = dict(
            self.db.conn.execute("SELECT id, qualname FROM units WHERE qualname != ''").fetchall()
        )
        n = 0
        t0 = time.monotonic()
        for i in range(0, len(units), BATCH):
            batch = units[i : i + BATCH]
            texts = [
                "\n".join(
                    part
                    for part in (
                        f"file: {path_rows.get(u.file_id, '')}",
                        f"language: {lang_rows.get(u.file_id, '')}",
                        f"parent: {parent_rows.get(u.parent_id, '')}" if u.parent_id else "",
                        u.retrieval_key,
                    )
                    if part
                )
                for u in batch
            ]
            vecs = self.embedder.embed_passages(texts)
            if len(vecs) != len(batch):
                raise RuntimeError(
                    f"embedding provider returned {len(vecs)} vectors for {len(batch)} units"
                )
            rows = []
            for u, v in zip(batch, vecs, strict=True):
                if len(v) != self.embedder.dimension or not all(math.isfinite(x) for x in v):
                    raise RuntimeError(f"invalid embedding for unit {u.id}")
                rows.append((u.id, lang_rows.get(u.file_id, ""), u.kind, v))
            self.db.store_embeddings(rows)
            n += len(batch)
            elapsed = time.monotonic() - t0
            rate = n / elapsed if elapsed > 0 else 0.0
            eta = (len(units) - n) / rate if rate > 0 else 0.0
            self.progress(f"  embedded {n}/{len(units)} ({rate:.1f}/s, eta {eta:.0f}s)")
        return n

    # ---------- call graph ----------

    def _map_calls(self, units: list[Unit], calls: list) -> list[tuple[int, str, str, int]]:
        """Map call sites to the innermost enclosing symbol unit.
        Returns (unit_id, callee, callee_full, line)."""
        funcs = sorted(
            [
                u
                for u in units
                if u.is_symbol and u.unit_type not in ("import", "config_key") and u.id is not None
            ],
            key=lambda u: u.byte_start,
        )
        edges: list[tuple[int, str, str, int]] = []
        starts = [u.byte_start for u in funcs]
        import bisect

        for c in calls:
            idx = bisect.bisect_right(starts, c.byte_start) - 1
            while idx >= 0:
                caller = funcs[idx]
                if caller.id is not None and caller.byte_start <= c.byte_start < caller.byte_end:
                    edges.append((caller.id, c.callee, c.callee_full, c.line))
                    break
                idx -= 1
        return edges

    def _map_refs(self, units: list[Unit], refs: list) -> list[tuple[int, str, str, str, int]]:
        """Map reference sites to the innermost enclosing symbol unit.
        Returns (unit_id, ref, ref_full, kind, line)."""
        funcs = sorted(
            [
                u
                for u in units
                if u.is_symbol and u.unit_type not in ("import", "config_key") and u.id is not None
            ],
            key=lambda u: u.byte_start,
        )
        edges: list[tuple[int, str, str, str, int]] = []
        starts = [u.byte_start for u in funcs]
        import bisect

        for c in refs:
            idx = bisect.bisect_right(starts, c.byte_start) - 1
            while idx >= 0:
                caller = funcs[idx]
                if caller.id is not None and caller.byte_start <= c.byte_start < caller.byte_end:
                    edges.append((caller.id, c.target, c.target_full, c.kind, c.line))
                    break
                idx -= 1
        return edges

    def _resolve_call_edges(self, changed_file_ids: set[int] | None = None) -> None:
        rows = self.db.conn.execute(
            """
            SELECT u.id, u.file_id, u.name, u.qualname, f.path
            FROM units u JOIN files f ON f.id = u.file_id
            WHERE u.kind = 'symbol' AND u.unit_type NOT IN ('import', 'config_key')
            """
        ).fetchall()
        by_file: dict[int, list] = {}
        by_name: dict[str, list] = {}
        by_qualname: dict[str, list] = {}
        for row in rows:
            by_file.setdefault(row["file_id"], []).append(row)
            by_name.setdefault(row["name"], []).append(row)
            by_qualname.setdefault(row["qualname"], []).append(row)
        aliases: dict[int, dict[str, str]] = {}
        for row in self.db.conn.execute("SELECT file_id, alias, target FROM import_aliases"):
            aliases.setdefault(row["file_id"], {})[row["alias"]] = row["target"]

        def module_name(path: str) -> str:
            return path.rsplit(".", 1)[0].replace("/", ".")

        def resolve(file_id: int, callee: str, full: str) -> int | None:
            local = by_file.get(file_id, [])
            candidates = [r for r in local if r["name"] == callee or r["qualname"] == full]
            if len(candidates) == 1:
                return candidates[0]["id"]
            binding = aliases.get(file_id, {})
            target = ""
            for alias, imported in binding.items():
                if full == alias:
                    target = imported
                    break
                if full.startswith(alias + "."):
                    target = imported + full[len(alias) :]
                    break
            if target:
                qualified = by_qualname.get(target, [])
                if len(qualified) == 1:
                    return qualified[0]["id"]
                parts = target.rsplit(".", 1)
                if len(parts) == 2:
                    module, symbol = parts
                    qualified = [
                        r for r in by_name.get(symbol, []) if module_name(r["path"]) == module
                    ]
                    if len(qualified) == 1:
                        return qualified[0]["id"]
            global_matches = by_name.get(callee, [])
            if len(global_matches) == 1:
                return global_matches[0]["id"]
            return None

        edge_query = "SELECT rowid, file_id, callee, callee_full FROM call_edges"
        edge_params: list = []
        if changed_file_ids is not None:
            clauses = ["callee_unit_id IS NULL"]
            if changed_file_ids:
                marks = ",".join("?" for _ in changed_file_ids)
                clauses.append(f"file_id IN ({marks})")
                edge_params.extend(changed_file_ids)
            if self._changed_symbol_names:
                name_marks = ",".join("?" for _ in self._changed_symbol_names)
                clauses.extend((f"callee IN ({name_marks})", f"callee_full IN ({name_marks})"))
                clauses.append(
                    "EXISTS (SELECT 1 FROM import_aliases ia WHERE ia.file_id = call_edges.file_id)"
                )
                names = sorted(self._changed_symbol_names)
                edge_params.extend(names)
                edge_params.extend(names)
            edge_query += " WHERE " + " OR ".join(clauses)
        updates = []
        for edge in self.db.conn.execute(edge_query, edge_params).fetchall():
            target_id = resolve(edge["file_id"], edge["callee"], edge["callee_full"])
            updates.append((target_id, edge["rowid"]))
        self.db.conn.executemany(
            "UPDATE call_edges SET callee_unit_id = ? WHERE rowid = ?", updates
        )
        ref_query = "SELECT rowid, file_id, ref, ref_full FROM ref_edges"
        ref_params: list = []
        if changed_file_ids is not None:
            clauses = ["ref_unit_id IS NULL"]
            if changed_file_ids:
                marks = ",".join("?" for _ in changed_file_ids)
                clauses.append(f"file_id IN ({marks})")
                ref_params.extend(changed_file_ids)
            if self._changed_symbol_names:
                name_marks = ",".join("?" for _ in self._changed_symbol_names)
                clauses.extend((f"ref IN ({name_marks})", f"ref_full IN ({name_marks})"))
                clauses.append(
                    "EXISTS (SELECT 1 FROM import_aliases ia WHERE ia.file_id = ref_edges.file_id)"
                )
                names = sorted(self._changed_symbol_names)
                ref_params.extend(names)
                ref_params.extend(names)
            ref_query += " WHERE " + " OR ".join(clauses)
        ref_updates = []
        for edge in self.db.conn.execute(ref_query, ref_params).fetchall():
            target_id = resolve(edge["file_id"], edge["ref"], edge["ref_full"])
            ref_updates.append((target_id, edge["rowid"]))
        self.db.conn.executemany(
            "UPDATE ref_edges SET ref_unit_id = ? WHERE rowid = ?", ref_updates
        )
        self.db.conn.commit()
        self._changed_symbol_names.clear()

    # ---------- queries ----------

    def embed_query(self, query: str) -> list[float]:
        return self.embedder.embed_query(query)
