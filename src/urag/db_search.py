"""Search and navigation queries against the urag index.

Split out of db.py so the storage core stays focused on files/units CRUD.
"""

from __future__ import annotations

import json
import re

from .models import Unit

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CONFIG_HINT = frozenset(
    {
        "config",
        "configs",
        "configuration",
        "configure",
        "configured",
        "setting",
        "settings",
        "env",
        "environment",
        "toml",
        "yaml",
        "yml",
        "json",
        "ini",
        "key",
        "keys",
    }
)
_IMPORT_HINT = frozenset(
    {
        "import",
        "imports",
        "imported",
        "dependency",
        "dependencies",
        "dependents",
        "module",
        "modules",
        "package",
        "packages",
        "require",
        "requires",
        "requirements",
    }
)


def _query_terms(text: str) -> set[str]:
    """Approximate FTS5 unicode61 tokens: casefolded word chars without `_`."""
    return {t for t in _TOKEN_RE.findall(text.casefold()) if len(t) > 1}


class _SearchMixin:
    # ---------- retrieval ----------

    def lexical_search(
        self,
        query: str,
        limit: int = 30,
        language: str | None = None,
        exact: bool = False,
    ) -> list[tuple[Unit, str, float]]:
        q = self._safe_fts(query.strip())
        if not q:
            return []
        exact_filter = ""
        exact_params: tuple = ()
        if exact:
            exact_filter = "AND (u.name = ? OR u.qualname = ? OR f.path = ?)"
            exact_params = (query.strip(), query.strip(), query.strip())
        fetch = limit if exact else max(limit * 4, limit + 10)
        rows = self.conn.execute(
            f"""
            SELECT u.*, f.path,
                   bm25(fts_units, 10.0, 8.0, 4.0, 2.0, 1.0, 1.0, 3.0, 2.0) AS score
            FROM fts_units
            JOIN units u ON u.id = fts_units.rowid
            JOIN files f ON f.id = u.file_id
            WHERE fts_units MATCH ? AND (? = '' OR f.language = ?) {exact_filter}
            ORDER BY score
            LIMIT ?
            """,
            (q, language or "", language or "", *exact_params, fetch),
        ).fetchall()
        hits = [(self._row_to_unit(r), r["path"], r["score"]) for r in rows]
        return self._rank_lexical(query, hits, limit, exact=exact)

    @staticmethod
    def _rank_lexical(
        query: str,
        hits: list[tuple[Unit, str, float]],
        limit: int,
        exact: bool = False,
    ) -> list[tuple[Unit, str, float]]:
        """Reorder lexical candidates by intent and multi-term agreement.

        Units matching more distinct query terms rank first; config-key and
        import units are demoted unless the query asks for them; for longer
        queries, single-term matches are dropped when multi-term matches
        exist. The caller's bm25 order breaks ties."""
        if exact or len(hits) < 2:
            return hits[:limit]
        terms = _query_terms(query)
        if len(terms) < 2:
            return hits[:limit]
        config_intent = bool(terms & _CONFIG_HINT)
        import_intent = bool(terms & _IMPORT_HINT)
        scored: list[tuple[int, int, float, Unit, str]] = []
        for unit, path, score in hits:
            text = "\n".join(
                part
                for part in (
                    unit.name,
                    unit.qualname,
                    unit.signature,
                    unit.summary,
                    unit.concepts,
                    unit.relationships,
                )
                if part
            )
            matched = len(terms & _query_terms(text))
            demoted = (unit.unit_type == "config_key" and not config_intent) or (
                unit.unit_type == "import" and not import_intent
            )
            scored.append((-matched, 1 if demoted else 0, score, unit, path))
        if len(terms) >= 4:
            strong = [entry for entry in scored if entry[0] <= -2]
            if strong:
                scored = strong
        scored.sort(key=lambda entry: entry[:3])
        return [(unit, path, score) for _m, _pen, score, unit, path in scored[:limit]]

    @staticmethod
    def _safe_fts(query: str) -> str:
        """Make a user query safe as an FTS5 MATCH expression.

        Bare `:` would be parsed as a column filter (`Runner::run` -> column
        "Runner"), and AND/OR/NOT are operator keywords. Quote every term.
        """
        terms = [t for t in re.findall(r"[\w.$:#/+-]+", query, flags=re.UNICODE) if t]
        if not terms:
            return ""
        if len(terms) == 1:
            return f'"{terms[0]}"'
        return " OR ".join(f'"{term}"' for term in terms)

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
        if not rows:
            return []
        ids = [row["unit_id"] for row in rows]
        marks = ",".join("?" for _ in ids)
        unit_rows = self.conn.execute(
            f"SELECT u.*, f.path FROM units u JOIN files f ON f.id = u.file_id WHERE u.id IN ({marks})",
            ids,
        ).fetchall()
        by_id: dict[int, tuple[Unit, str]] = {
            row["id"]: (self._row_to_unit(row), row["path"]) for row in unit_rows
        }
        out: list[tuple[Unit, str, float]] = []
        for r in rows:
            got = by_id.get(r["unit_id"])
            if got:
                u, path = got
                out.append((u, path, r["distance"]))
        return out

    # ---------- navigation (agent file/symbol browsing) ----------

    def file_list(self, language: str | None = None) -> list[dict]:
        """All indexed files: (path, language, kind, size, commit, units)."""
        q = """
            SELECT f.path, f.language, f.kind, f.size, f."commit",
                   COUNT(u.id) AS unit_count
            FROM files f LEFT JOIN units u ON u.file_id = f.id
        """
        params: list = []
        if language:
            q += " WHERE f.language = ?"
            params.append(language)
        q += " GROUP BY f.id ORDER BY f.path"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def file_by_path(self, path: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def units_by_file_path(self, path: str) -> list[Unit]:
        rows = self.conn.execute(
            "SELECT u.* FROM units u JOIN files f ON f.id = u.file_id "
            "WHERE f.path = ? ORDER BY u.start_line, u.start_col",
            (path,),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def resolve_units(
        self, name: str, limit: int = 30, language: str | None = None
    ) -> list[tuple[Unit, str, str]]:
        """Exact symbol definition lookup by name or qualname."""
        name = name.strip()
        if not name:
            return []
        rows = self.conn.execute(
            """
            SELECT u.*, f.path, f."commit"
            FROM units u JOIN files f ON f.id = u.file_id
            WHERE u.kind = 'symbol' AND u.unit_type != 'import'
              AND (u.name = ? OR u.qualname = ? OR u.qualname LIKE ? ESCAPE '\\')
              AND (? = '' OR f.language = ?)
            ORDER BY (u.qualname = ?) DESC, (u.name = ?) DESC,
                     (u.unit_type = 'class' OR u.unit_type = 'struct'
                      OR u.unit_type = 'interface' OR u.unit_type = 'enum') DESC,
                     u.name
            LIMIT ?
            """,
            (
                name,
                name,
                "%." + name.replace("%", r"\%").replace("_", r"\_"),
                language or "",
                language or "",
                name,
                name,
                limit,
            ),
        ).fetchall()
        return [(self._row_to_unit(r), r["path"], r["commit"]) for r in rows]

    def children_of(self, parent_id: int) -> list[Unit]:
        rows = self.conn.execute(
            "SELECT u.* FROM units u JOIN files f ON f.id = u.file_id "
            "WHERE u.parent_id = ? ORDER BY u.start_line, u.start_col",
            (parent_id,),
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def siblings_of(self, unit_id: int) -> list[Unit]:
        row = self.conn.execute("SELECT parent_id FROM units WHERE id = ?", (unit_id,)).fetchone()
        if row is None or row["parent_id"] is None:
            return []
        return self.children_of(row["parent_id"])

    def importers(self, target: str, limit: int = 50) -> list[dict]:
        """Files/units that import a module or symbol (dependents).

        Matches import_aliases targets and import-unit qualnames against
        `target` (exact or sub-module prefix). C/C++ `#include` targets are
        matched by path-component suffix: a query for
        `src/encoder/foo.h` finds files with `#include "encoder/foo.h"`."""
        target = target.strip()
        if not target:
            return []
        target = target.replace("\\", "/").removeprefix("./")
        esc = target.replace("%", r"\%").replace("_", r"\_")
        out: dict[str, dict] = {}
        rows = self.conn.execute(
            """
            SELECT DISTINCT f.path, ia.alias, ia.target, NULL AS unit_id, f."commit"
            FROM import_aliases ia JOIN files f ON f.id = ia.file_id
            WHERE ia.target = ? OR ia.target LIKE ? ESCAPE '\\'
            ORDER BY f.path
            LIMIT ?
            """,
            (target, esc + ".%", limit),
        ).fetchall()
        for r in rows:
            out.setdefault(r["path"], dict(r))
        urows = self.conn.execute(
            """
            SELECT u.id AS unit_id, f.path, '' AS alias, u.qualname AS target, f."commit"
            FROM units u JOIN files f ON f.id = u.file_id
            WHERE u.unit_type = 'import' AND (u.qualname = ? OR u.qualname LIKE ? ESCAPE '\\')
            ORDER BY f.path
            LIMIT ?
            """,
            (target, esc + ".%", limit),
        ).fetchall()
        for r in urows:
            key = r["path"]
            if key not in out or out[key]["unit_id"] is None:
                out.setdefault(key, dict(r))
        # include-style suffix matching (C/C++, and quoted
        # import qualnames that carry quotes/angle brackets)
        base = target.rsplit("/", 1)[-1]
        if base:
            irows = self.conn.execute(
                """
                SELECT u.id AS unit_id, f.path, '' AS alias, u.qualname AS target, f."commit"
                FROM units u JOIN files f ON f.id = u.file_id
                WHERE u.unit_type = 'import' AND u.qualname LIKE ? ESCAPE '\\'
                ORDER BY f.path
                """,
                (f"%{base.replace('%', chr(92) + '%').replace('_', chr(92) + '_')}%",),
            ).fetchall()
            tparts = [p for p in target.split("/") if p]
            for r in irows:
                inc = r["target"].strip().strip('"<> ')
                if not inc:
                    continue
                iparts = [p for p in inc.split("/") if p]
                n = min(len(tparts), len(iparts))
                if n and tparts[-n:] == iparts[-n:]:
                    key = r["path"]
                    if key not in out or out[key]["unit_id"] is None:
                        out.setdefault(key, dict(r))
        if not out:
            # symbol target: fall back to dependents of the defining file
            srow = self.conn.execute(
                "SELECT f.path FROM units u JOIN files f ON f.id = u.file_id "
                "WHERE u.kind = 'symbol' AND u.unit_type NOT IN ('import', 'config_key') "
                "AND (u.name = ? OR u.qualname = ?) LIMIT 1",
                (target, target),
            ).fetchone()
            if srow and srow["path"] != target:
                return self.importers(srow["path"], limit=limit)
        return list(out.values())[:limit]
