"""Call-graph and reference-edge storage and queries.

Split out of db.py so edge logic (callers, references, transitive walks,
dead-code candidates) lives in one place.
"""

from __future__ import annotations

import re


class _CallGraphMixin:
    # ---------- call graph ----------

    def replace_call_edges(
        self, file_id: int, edges: list[tuple[int, str, str, int]], commit: bool = True
    ) -> None:
        """Replace all call edges of a file: (caller_unit_id, callee, full, line)."""
        self.conn.execute("DELETE FROM call_edges WHERE file_id = ?", (file_id,))
        self.conn.executemany(
            "INSERT INTO call_edges(caller_unit_id, file_id, callee, callee_full, line, callee_unit_id) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            [(cid, file_id, callee, full, line) for cid, callee, full, line in edges],
        )
        if commit:
            self.conn.commit()

    def replace_import_aliases(
        self, file_id: int, aliases: list[tuple[str, str]], commit: bool = True
    ) -> None:
        """Replace a file's import bindings: (alias, fully-qualified target)."""
        self.conn.execute("DELETE FROM import_aliases WHERE file_id = ?", (file_id,))
        self.conn.executemany(
            "INSERT INTO import_aliases(file_id, alias, target) VALUES (?, ?, ?)",
            [(file_id, a, t) for a, t in aliases],
        )
        if commit:
            self.conn.commit()

    def replace_ref_edges(
        self,
        file_id: int,
        edges: list[tuple[int, str, str, str, int]],
        commit: bool = True,
    ) -> None:
        """Replace all reference edges of a file:
        (enclosing_unit_id, ref, ref_full, kind, line)."""
        self.conn.execute("DELETE FROM ref_edges WHERE file_id = ?", (file_id,))
        self.conn.executemany(
            "INSERT INTO ref_edges(unit_id, file_id, ref, ref_full, kind, line, ref_unit_id) "
            "VALUES (?, ?, ?, ?, ?, ?, NULL)",
            [
                (uid, file_id, ref, full, kind, line)
                for uid, ref, full, kind, line in edges
            ],
        )
        if commit:
            self.conn.commit()

    def callers(self, name: str, limit: int = 30) -> list[dict]:
        """Units that call `name` (matches last segment or full chain).

        For fully-qualified names, call sites written through import aliases
        are resolved (e.g. `import os.path as op; op.exists()` is found when
        querying `os.path.exists`). Alias bindings that collide with a local
        symbol in the same file are ignored.
        """
        name = name.strip()
        if not name:
            return []
        esc = re.escape(name).replace("%", r"\%").replace("_", r"\_")
        exact_ids = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM units WHERE (name = ? OR qualname = ?) "
                "AND kind = 'symbol' AND unit_type NOT IN ('import', 'config_key')",
                (name, name),
            ).fetchall()
        ]
        resolved_clause = ""
        resolved_params: list = []
        if exact_ids:
            marks = ",".join("?" for _ in exact_ids)
            resolved_clause = f" OR e.callee_unit_id IN ({marks})"
            resolved_params.extend(exact_ids)
        rows = self.conn.execute(
            f"""
            SELECT u.*, f.path, e.callee_full, e.line, '' AS resolved_target
            FROM call_edges e
            JOIN units u ON u.id = e.caller_unit_id
            JOIN files f ON f.id = e.file_id
            WHERE e.callee = ? OR e.callee_full = ? OR e.callee_full LIKE ? ESCAPE '\\'
               OR e.callee_full LIKE ? ESCAPE '\\' {resolved_clause}
            ORDER BY CASE WHEN f.path LIKE '%test%' THEN 1 ELSE 0 END, e.line
            """,
            (name, name, f"%.{esc}", f"%::{esc}", *resolved_params),
        ).fetchall()
        by_unit: dict[int, dict] = {}
        for r in rows:
            by_unit.setdefault(r["id"], self._caller_row(r))
        if any(sep in name for sep in (".", "::")):
            # alias-aware matching: callee chains written through per-file
            # import bindings resolve to the fully-qualified target
            arows = self.conn.execute(
                """
                SELECT DISTINCT e.caller_unit_id, e.callee_full, e.line, f.path,
                       u.*, ia.target AS resolved_target
                FROM call_edges e
                JOIN import_aliases ia ON ia.file_id = e.file_id
                JOIN units u ON u.id = e.caller_unit_id
                JOIN files f ON f.id = e.file_id
                WHERE ((e.callee_full = ia.alias AND ia.target = ?)
                    OR (e.callee_full = ia.alias || '.' || substr(?, length(ia.target) + 2)
                        AND substr(?, 1, length(ia.target) + 1) = ia.target || '.'))
                  AND NOT EXISTS (
                       SELECT 1 FROM units u2 WHERE u2.file_id = ia.file_id
                       AND u2.name = ia.alias AND u2.unit_type != 'import')
                ORDER BY CASE WHEN f.path LIKE '%test%' THEN 1 ELSE 0 END, e.line
                """,
                (name, name, name),
            ).fetchall()
            for r in arows:
                by_unit.setdefault(r["id"], self._caller_row(r))
        return list(by_unit.values())[:limit]

    def _caller_row(self, r: dict) -> dict:
        return {
            "unit": self._row_to_unit(r),
            "path": r["path"],
            "callee_full": r["callee_full"],
            "line": r["line"],
            "resolved_target": r["resolved_target"],
        }

    def callees(self, unit_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT callee, callee_full, line FROM call_edges WHERE caller_unit_id = ? ORDER BY line",
            (unit_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def transitive_callers(
        self, name: str, max_depth: int = 3, limit: int = 30
    ) -> list[dict]:
        """BFS over call_edges: all transitive callers of `name` (callers-of-
        callers, ...). Each row gains `hop` (1 = direct caller). Cycles are
        handled via a visited set; each unit appears at its shortest hop."""
        name = name.strip()
        if not name:
            return []
        visited: dict[int, dict] = {}
        frontier: list[str] = [name]
        for hop in range(1, max_depth + 1):
            nxt: list[str] = []
            for callee in frontier:
                for row in self.callers(callee, limit=10000):
                    uid = row["unit"].id
                    if uid in visited:
                        continue
                    visited[uid] = row | {"hop": hop}
                    u = row["unit"]
                    nxt.append(u.name)
                    if u.qualname and u.qualname != u.name:
                        nxt.append(u.qualname)
            if not nxt:
                break
            frontier = list(dict.fromkeys(nxt))
        return [visited[uid] for uid in visited][:limit]

    def references(self, name: str, limit: int = 30) -> list[dict]:
        """Units that reference `name`: type mentions, constructions, bases,
        annotations, casts, attributes, XAML bindings.

        Alias-aware like callers(): references written through per-file
        import bindings resolve to fully-qualified targets.
        """
        name = name.strip()
        if not name:
            return []
        esc = re.escape(name).replace("%", r"\%").replace("_", r"\_")
        exact_ids = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM units WHERE (name = ? OR qualname = ?) "
                "AND kind = 'symbol' AND unit_type NOT IN ('import', 'config_key')",
                (name, name),
            ).fetchall()
        ]
        resolved_clause = ""
        resolved_params: list = []
        if exact_ids:
            marks = ",".join("?" for _ in exact_ids)
            resolved_clause = f" OR e.ref_unit_id IN ({marks})"
            resolved_params.extend(exact_ids)
        rows = self.conn.execute(
            f"""
            SELECT u.*, f.path, e.ref_full, e.kind AS ref_kind, e.line, '' AS resolved_target
            FROM ref_edges e
            JOIN units u ON u.id = e.unit_id
            JOIN files f ON f.id = e.file_id
            WHERE e.ref = ? OR e.ref_full = ? OR e.ref_full LIKE ? ESCAPE '\\'
               OR e.ref_full LIKE ? ESCAPE '\\' {resolved_clause}
            ORDER BY CASE WHEN f.path LIKE '%test%' THEN 1 ELSE 0 END, e.line
            """,
            (name, name, f"%.{esc}", f"%::{esc}", *resolved_params),
        ).fetchall()
        by_unit: dict[int, dict] = {}
        for r in rows:
            by_unit.setdefault(r["id"], self._ref_row(r))
        if any(sep in name for sep in (".", "::")):
            arows = self.conn.execute(
                """
                SELECT DISTINCT e.unit_id, e.ref_full, e.kind AS ref_kind, e.line, f.path,
                       u.*, ia.target AS resolved_target
                FROM ref_edges e
                JOIN import_aliases ia ON ia.file_id = e.file_id
                JOIN units u ON u.id = e.unit_id
                JOIN files f ON f.id = e.file_id
                WHERE ((e.ref_full = ia.alias AND ia.target = ?)
                    OR (e.ref_full = ia.alias || '.' || substr(?, length(ia.target) + 2)
                        AND substr(?, 1, length(ia.target) + 1) = ia.target || '.'))
                  AND NOT EXISTS (
                       SELECT 1 FROM units u2 WHERE u2.file_id = ia.file_id
                       AND u2.name = ia.alias AND u2.unit_type != 'import')
                ORDER BY CASE WHEN f.path LIKE '%test%' THEN 1 ELSE 0 END, e.line
                """,
                (name, name, name),
            ).fetchall()
            for r in arows:
                by_unit.setdefault(r["id"], self._ref_row(r))
        return list(by_unit.values())[:limit]

    def _ref_row(self, r: dict) -> dict:
        return {
            "unit": self._row_to_unit(r),
            "path": r["path"],
            "ref_full": r["ref_full"],
            "kind": r["ref_kind"],
            "line": r["line"],
            "resolved_target": r["resolved_target"],
        }

    def transitive_references(
        self, name: str, max_depth: int = 3, limit: int = 30
    ) -> list[dict]:
        """BFS over ref_edges: all transitive referencers of `name`. Each row
        gains `hop` (1 = direct referencer). Cycles handled via visited set."""
        name = name.strip()
        if not name:
            return []
        visited: dict[int, dict] = {}
        frontier: list[str] = [name]
        for hop in range(1, max_depth + 1):
            nxt: list[str] = []
            for ref in frontier:
                for row in self.references(ref, limit=10000):
                    uid = row["unit"].id
                    if uid in visited:
                        continue
                    visited[uid] = row | {"hop": hop}
                    u = row["unit"]
                    nxt.append(u.name)
                    if u.qualname and u.qualname != u.name:
                        nxt.append(u.qualname)
            if not nxt:
                break
            frontier = list(dict.fromkeys(nxt))
        return [visited[uid] for uid in visited][:limit]

    def unreferenced_symbols(
        self, limit: int = 50, language: str | None = None
    ) -> list[dict]:
        """Candidate dead symbols: symbol units with no incoming call edges
        and no incoming reference edges. Excludes imports, config keys and
        test files. Heuristic only — dynamic dispatch, reflection, XAML
        bindings of unsupported flavors, and external entry points can
        produce false positives."""
        excluded_types = (
            "import",
            "config_key",
            "file",
            "resource",
            "template",
            "element",
            "event",
        )
        marks = ",".join("?" for _ in excluded_types)
        rows = self.conn.execute(
            f"""
            SELECT u.*, f.path
            FROM units u
            JOIN files f ON f.id = u.file_id
            WHERE u.kind = 'symbol'
              AND u.unit_type NOT IN ({marks})
              AND f.path NOT LIKE '%test%'
              AND f.path NOT LIKE '%spec%'
              AND (? = '' OR f.language = ?)
              AND NOT EXISTS (
                  SELECT 1 FROM call_edges e
                  WHERE e.callee = u.name
                     OR e.callee_full = u.qualname
                     OR e.callee_full = u.name
                     OR e.callee_unit_id = u.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM ref_edges r
                  WHERE r.ref = u.name
                     OR r.ref_full = u.qualname
                     OR r.ref_full = u.name
                     OR r.ref_unit_id = u.id
              )
            ORDER BY f.path, u.start_line
            LIMIT ?
            """,
            (*excluded_types, language or "", language or "", limit),
        ).fetchall()
        return [
            {
                "unit": self._row_to_unit(r),
                "path": r["path"],
                "ref_full": "",
                "kind": "",
                "line": r["start_line"],
                "resolved_target": "",
            }
            for r in rows
        ]

