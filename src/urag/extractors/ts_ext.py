"""TypeScript/JavaScript extractor: functions, classes, interfaces, imports."""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_typescript as tsts
import tree_sitter_javascript as tsjs

from ..models import Unit, UNIT_KIND_SYMBOL
from .base import (
    ByteIndexedSource,
    Extractor,
    MAX_SUMMARY_CHARS,
    collapse_ws,
    leading_comments,
    valid_aliases,
    walk_calls,
)

_FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
    "abstract_method_signature",
    "method_signature",
}

_TYPE_TYPES = {
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "class_declaration",
    "abstract_class_declaration",
}

_PARSERS: dict[str, Parser] = {}


def _parser(language: str) -> Parser:
    if language not in _PARSERS:
        p = Parser()
        if language == "javascript":
            p.language = Language(tsjs.language())
        elif language == "typescript":
            p.language = Language(tsts.language_typescript())
        else:
            p.language = Language(tsts.language_tsx())
        _PARSERS[language] = p
    return _PARSERS[language]


def _point(n: Node, which: str) -> tuple[int, int]:
    p = getattr(n, f"{which}_point")
    return (p.row + 1, p.column)


def _name_of(n: Node, source: str) -> str:
    name = n.child_by_field_name("name")
    if name:
        return source[name.start_byte : name.end_byte]
    left = n.child_by_field_name("left")
    if left:
        return source[left.start_byte : left.end_byte]
    return "anonymous"


class TsExtractor(Extractor):
    def __init__(self, language: str = "typescript"):
        self.language = language

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        tree = _parser(self.language).parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, rel_path, lines, prefix="", units=units)
        return units

    def collect_calls(self, source: str) -> list:
        from ..models import CallSite

        source = ByteIndexedSource(source)
        tree = _parser(self.language).parse(source.encode("utf-8"))
        out: list[CallSite] = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """(alias, target): `import * as ns`, `import {A as B}`, `import D`,
        and bare named imports `import {A}`."""
        source = ByteIndexedSource(source)
        tree = _parser(self.language).parse(source.encode("utf-8"))
        out: list[tuple[str, str]] = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "import_statement":
                clause = next(
                    (c for c in cur.named_children if c.type == "import_clause"), None
                )
                source_node = cur.child_by_field_name("source")
                mod = (
                    source[source_node.start_byte : source_node.end_byte].strip("'\"")
                    if source_node
                    else ""
                )
                mod = mod.strip("@").replace("/", ".")
                if clause is None:
                    continue
                ns = next(
                    (c for c in clause.named_children if c.type == "namespace_import"),
                    None,
                )
                if ns is not None:
                    name = ns.child_by_field_name("name")
                    if name is None:
                        name = next(
                            (c for c in ns.named_children if c.type == "identifier"),
                            None,
                        )
                    if name:
                        out.append((source[name.start_byte : name.end_byte], mod))
                else:
                    ident = next(
                        (c for c in clause.named_children if c.type == "identifier"),
                        None,
                    )
                    if ident:
                        out.append(
                            (
                                source[ident.start_byte : ident.end_byte],
                                f"{mod}.default",
                            )
                        )
                    for ni in clause.named_children:
                        if ni.type != "named_imports":
                            continue
                        for spec in ni.named_children:
                            if spec.type != "import_specifier":
                                continue
                            name = spec.child_by_field_name("name")
                            alias = spec.child_by_field_name("alias")
                            imported = (
                                source[name.start_byte : name.end_byte] if name else ""
                            )
                            if alias:
                                out.append(
                                    (
                                        source[alias.start_byte : alias.end_byte],
                                        f"{mod}.{imported}",
                                    )
                                )
                            elif imported:
                                out.append((imported, f"{mod}.{imported}"))
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _walk(
        self,
        node: Node,
        source: str,
        rel_path: str,
        lines: list[str],
        prefix: str,
        units: list[Unit],
    ) -> None:
        for child in node.named_children:
            t = child.type
            if t in ("function_declaration", "generator_function_declaration"):
                units.append(self._func(child, source, lines, prefix))
            elif t in ("class_declaration", "abstract_class_declaration"):
                self._class(child, source, lines, prefix, units)
            elif t in (
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            ):
                units.append(self._type(child, source, lines, prefix))
                if t == "interface_declaration":
                    body = child.child_by_field_name("body")
                    if body:
                        qualname = _name_of(child, source)
                        if prefix:
                            qualname = f"{prefix}.{qualname}"
                        for member in body.named_children:
                            if member.type in (
                                "method_signature",
                                "abstract_method_signature",
                            ):
                                units.append(
                                    self._func(member, source, lines, qualname)
                                )
            elif t in ("import_statement", "import_from_statement"):
                units.append(self._import(child, source))
            elif t == "export_statement":
                decl = child.named_children[-1] if child.named_children else None
                if decl and decl.type in _FUNCTION_TYPES | _TYPE_TYPES | {
                    "lexical_declaration"
                }:
                    self._walk(child, source, rel_path, lines, prefix, units)
            elif t == "lexical_declaration":
                for v in child.named_children:
                    if v.type == "variable_declarator":
                        val = v.child_by_field_name("value")
                        if val and val.type in (
                            "arrow_function",
                            "function_expression",
                        ):
                            units.append(self._func(v, source, lines, prefix))

    def _func(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        body = n.child_by_field_name("body")
        sig_text = source[n.start_byte : (body.start_byte if body else n.end_byte)]
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        doc = leading_comments(lines, sl, marker="//") or leading_comments(
            lines, sl, marker="/*"
        )
        params = n.child_by_field_name("parameters")
        concepts = self._identifiers(n, source, 12)
        qualname = f"{prefix}.{name}" if prefix else name
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="method"
            if n.type
            in ("method_definition", "abstract_method_signature", "method_signature")
            else "function",
            name=name,
            qualname=qualname,
            signature=collapse_ws(sig_text),
            summary=doc,
            concepts=", ".join(concepts),
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )

    def _class(
        self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        name = _name_of(n, source)
        body = n.child_by_field_name("body")
        sig_text = source[n.start_byte : (body.start_byte if body else n.end_byte)]
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        doc = leading_comments(lines, sl, marker="//") or leading_comments(
            lines, sl, marker="/*"
        )
        qualname = f"{prefix}.{name}" if prefix else name
        bases: list[str] = []
        for f in ("extends_clause", "class_heritage", "implements_clause"):
            clause = n.child_by_field_name(f) or (n.child_by_field_name("heritage"))
            if clause:
                for c in clause.named_children:
                    bases.append(source[c.start_byte : c.end_byte])
        units.append(
            Unit(
                file_id=0,
                kind=UNIT_KIND_SYMBOL,
                unit_type="class",
                name=name,
                qualname=qualname,
                signature=collapse_ws(sig_text),
                summary=doc,
                concepts=", ".join(bases),
                start_line=sl,
                end_line=el,
                start_col=sc,
                end_col=ec,
                byte_start=n.start_byte,
                byte_end=n.end_byte,
            )
        )
        if body:
            for m in body.named_children:
                if m.type in (
                    "method_definition",
                    "abstract_method_signature",
                    "property_signature",
                ):
                    units.append(self._func(m, source, lines, qualname))
                elif m.type == "class_static_block":
                    continue

    def _type(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        doc = leading_comments(lines, sl, marker="//") or leading_comments(
            lines, sl, marker="/*"
        )
        qualname = f"{prefix}.{name}" if prefix else name
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type=n.type.replace("_declaration", "").replace("_", "_"),
            name=name,
            qualname=qualname,
            signature=collapse_ws(source[n.start_byte : n.end_byte]),
            summary=doc,
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )

    def _import(self, n: Node, source: str) -> Unit:
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        source_str = source[n.start_byte : n.end_byte]
        clause = n.child_by_field_name("source")
        mod = source[clause.start_byte : clause.end_byte].strip("'\"") if clause else ""
        names: list[str] = []
        import_clause = next(
            (c for c in n.named_children if c.type == "import_clause"), None
        )
        if import_clause:
            stack: list[Node] = [import_clause]
            while stack:
                cur = stack.pop()
                if cur.type == "import_specifier":
                    nm = cur.child_by_field_name("name")
                    if nm:
                        names.append(source[nm.start_byte : nm.end_byte])
                elif cur.type == "namespace_import":
                    names.append(
                        source[cur.start_byte : cur.end_byte]
                        .replace("* as", "*")
                        .strip()
                    )
                elif cur.type == "named_imports":
                    stack.extend(cur.named_children)
                else:
                    stack.extend(cur.named_children)
            if not names:
                nm = import_clause.child_by_field_name("name")
                if nm:
                    names.append(source[nm.start_byte : nm.end_byte])
        first = f"{mod}.{names[0]}" if names and mod else (names[0] if names else mod)
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=first.split(".")[-1],
            qualname=first,
            signature=collapse_ws(source_str),
            summary="import",
            concepts=", ".join(names[:6]),
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )

    @staticmethod
    def _identifiers(n: Node, source: str, limit: int = 6) -> list[str]:
        out: list[str] = []
        stack: list[Node] = [n]
        while stack and len(out) < limit:
            cur = stack.pop()
            for ch in cur.named_children:
                if ch.type in ("identifier", "property_identifier"):
                    out.append(source[ch.start_byte : ch.end_byte])
                elif len(out) < limit:
                    stack.append(ch)
        return out
