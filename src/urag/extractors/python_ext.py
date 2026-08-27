"""Python extractor: functions, classes, methods, imports via tree-sitter."""

from __future__ import annotations

from typing import Iterator

from tree_sitter import Language, Node, Parser
import tree_sitter_python as tsp

from ..models import Unit, UNIT_KIND_SYMBOL
from .base import (
    ByteIndexedSource,
    Extractor,
    MAX_SUMMARY_CHARS,
    collapse_ws,
    dedupe_refs,
    valid_aliases,
    walk_calls,
)

_LANG = Language(tsp.language())
_PARSER: Parser | None = None


def _parser() -> Parser:
    global _PARSER
    if _PARSER is None:
        _PARSER = Parser()
        _PARSER.language = _LANG
    return _PARSER


def _point(n: Node, which: str) -> tuple[int, int]:
    p = getattr(n, f"{which}_point")
    return (p.row + 1, p.column)


def _docstring(n: Node, source: str) -> str:
    if n.type != "block":
        return ""
    first = n.children[0] if n.children else None
    if first is None or first.type != "expression_statement":
        return ""
    inner = first.named_children[0] if first.named_children else None
    if inner is None or inner.type != "string":
        return ""
    text = source[inner.start_byte : inner.end_byte]
    try:
        import ast

        val = ast.literal_eval(text)
    except Exception:
        val = text.strip("'\"")
    if isinstance(val, str):
        return " ".join(val.split())[:MAX_SUMMARY_CHARS]
    return ""


def _identifiers(n: Node, source: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    stack: list[Node] = [n]
    while stack and len(out) < limit:
        cur = stack.pop()
        for ch in cur.named_children:
            if ch.type == "identifier":
                out.append(source[ch.start_byte : ch.end_byte])
            elif len(out) < limit:
                stack.append(ch)
    return out


class PythonExtractor(Extractor):
    language = "python"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        tree = _parser().parse(source.encode("utf-8"))
        units: list[Unit] = []
        self._walk(
            tree.root_node, source, rel_path, prefix="", parent=None, units=units
        )
        return units

    def collect_calls(self, source: str) -> list:
        from ..models import CallSite

        source = ByteIndexedSource(source)
        tree = _parser().parse(source.encode("utf-8"))
        out: list[CallSite] = []
        walk_calls(tree.root_node, source, {"call"}, out)
        return out

    def collect_references(self, source: str) -> list:
        """Python: class bases, type annotations, isinstance/issubclass casts,
        and identifiers inside subscripted annotations."""
        import re

        from ..models import Reference

        source = ByteIndexedSource(source)
        tree = _parser().parse(source.encode("utf-8"))
        out: list[Reference] = []
        containers = {
            "list",
            "dict",
            "set",
            "tuple",
            "frozenset",
            "optional",
            "union",
            "callable",
            "iterable",
            "iterator",
            "sequence",
            "mapping",
            "mutablemapping",
            "type",
            "literal",
            "annotated",
            "final",
            "classvar",
            "typing",
            "collections",
            "bool",
            "int",
            "float",
            "str",
            "bytes",
            "bytearray",
            "object",
            "none",
        }

        def append(node, kind: str) -> None:
            if node is None or node.type not in ("identifier", "attribute"):
                return
            text = source[node.start_byte : node.end_byte]
            base = text.split("[", 1)[0]
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", base)
            if not names or names[-1].lower() in containers:
                return
            out.append(
                Reference(
                    target=names[-1],
                    target_full=text,
                    kind=kind,
                    line=node.start_point.row + 1,
                    byte_start=node.start_byte,
                    byte_end=node.end_byte,
                )
            )

        def walk_type(node, kind: str) -> None:
            if node is None:
                return
            if node.type == "type":
                node = next(iter(node.named_children), None)
            if node is None:
                return
            if node.type == "subscript":
                for child in node.named_children:
                    walk_type(child, kind)
                return
            if node.type in ("union_type", "intersection_type"):
                for child in node.named_children:
                    walk_type(child, kind)
                return
            append(node, kind)

        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            t = cur.type
            if t == "class_definition":
                supers = cur.child_by_field_name("superclasses") or next(
                    (c for c in cur.named_children if c.type == "argument_list"),
                    None,
                )
                if supers is not None:
                    for child in supers.named_children:
                        if child.type in ("identifier", "attribute"):
                            append(child, "base")
            elif t in ("typed_parameter", "typed_default_parameter", "assignment"):
                walk_type(cur.child_by_field_name("type"), "type")
            elif t == "function_definition":
                walk_type(cur.child_by_field_name("return_type"), "type")
            elif t == "call":
                fn = cur.child_by_field_name("function")
                if fn is not None and fn.type == "identifier":
                    name = source[fn.start_byte : fn.end_byte]
                    if name in ("isinstance", "issubclass"):
                        args = cur.child_by_field_name("arguments")
                        if args is not None and len(args.named_children) >= 2:
                            walk_type(args.named_children[1], "cast")
            stack.extend(reversed(cur.named_children))
        return dedupe_refs(out)

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """(alias, fully-qualified target) for `import x as y`, `from m import n`
        and `from m import n as y`."""
        source = ByteIndexedSource(source)
        tree = _parser().parse(source.encode("utf-8"))
        out: list[tuple[str, str]] = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "import_statement":
                for child in cur.named_children:
                    if child.type == "aliased_import":
                        name = child.child_by_field_name("name")
                        alias = child.child_by_field_name("alias")
                        if name is not None and alias is not None:
                            out.append(
                                (
                                    source[alias.start_byte : alias.end_byte],
                                    source[name.start_byte : name.end_byte],
                                )
                            )
            elif cur.type == "import_from_statement":
                mod = cur.child_by_field_name("module_name")
                mod_name = source[mod.start_byte : mod.end_byte] if mod else ""
                for child in cur.named_children:
                    if child == mod:
                        continue
                    if child.type == "aliased_import":
                        name = child.child_by_field_name("name")
                        alias = child.child_by_field_name("alias")
                        if name is not None and alias is not None:
                            out.append(
                                (
                                    source[alias.start_byte : alias.end_byte],
                                    f"{mod_name}.{source[name.start_byte : name.end_byte]}",
                                )
                            )
                    elif child.type == "dotted_name":
                        out.append(
                            (
                                source[child.start_byte : child.end_byte],
                                f"{mod_name}.{source[child.start_byte : child.end_byte]}",
                            )
                        )
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _walk(
        self,
        node: Node,
        source: str,
        rel_path: str,
        prefix: str,
        parent: Node | None,
        units: list[Unit],
    ) -> None:
        for child in node.named_children:
            if child.type in ("function_definition", "async_function_definition"):
                unit = self._func(child, source, rel_path, prefix, parent)
                units.append(unit)
                body = child.child_by_field_name("body")
                if body is not None:
                    self._walk(body, source, rel_path, unit.qualname, None, units)
            elif child.type == "class_definition":
                self._class(child, source, rel_path, prefix, parent, units)
            elif child.type == "decorated_definition":
                inner = child.named_children[-1] if child.named_children else None
                if inner is None:
                    continue
                if inner.type in ("function_definition", "async_function_definition"):
                    unit = self._func(inner, source, rel_path, prefix, parent, child)
                    units.append(unit)
                    body = inner.child_by_field_name("body")
                    if body is not None:
                        self._walk(body, source, rel_path, unit.qualname, None, units)
                elif inner.type == "class_definition":
                    self._class(
                        inner, source, rel_path, prefix, parent, units, decorated=child
                    )
            elif child.type in ("import_statement", "import_from_statement"):
                units.append(self._imports(child, source, rel_path, prefix))
            else:
                self._walk(child, source, rel_path, prefix, parent, units)

    def _func(
        self,
        n: Node,
        source: str,
        rel_path: str,
        prefix: str,
        parent: Node | None,
        decorated: Node | None = None,
    ) -> Unit:
        name_node = n.child_by_field_name("name")
        name = (
            source[name_node.start_byte : name_node.end_byte]
            if name_node
            else "anonymous"
        )
        params = n.child_by_field_name("parameters")
        body = n.child_by_field_name("body")
        sig_text = source[n.start_byte : (body.start_byte if body else n.end_byte)]
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        doc = _docstring(body, source) if body else ""
        concepts = _identifiers(n, source, 12)
        decorators = self._decorator_names(decorated or n, source)
        qualname = f"{prefix}.{name}" if prefix else name
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="method" if parent else "function",
            name=name,
            qualname=qualname,
            signature=collapse_ws(sig_text),
            summary=doc,
            concepts=", ".join(concepts),
            relationships=decorators,
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )

    def _class(
        self,
        n: Node,
        source: str,
        rel_path: str,
        prefix: str,
        parent: Node | None,
        units: list[Unit],
        decorated: Node | None = None,
    ) -> None:
        name_node = n.child_by_field_name("name")
        name = (
            source[name_node.start_byte : name_node.end_byte]
            if name_node
            else "anonymous"
        )
        body = n.child_by_field_name("body")
        sig_text = source[n.start_byte : (body.start_byte if body else n.end_byte)]
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        doc = _docstring(body, source) if body else ""
        qualname = f"{prefix}.{name}" if prefix else name
        bases: list[str] = []
        sup = n.child_by_field_name("superclasses")
        if sup:
            bases = [source[c.start_byte : c.end_byte] for c in sup.named_children]
        decorators = self._decorator_names(decorated or n, source)
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
                relationships=decorators,
                start_line=sl,
                end_line=el,
                start_col=sc,
                end_col=ec,
                byte_start=n.start_byte,
                byte_end=n.end_byte,
            )
        )
        if body:
            self._walk(body, source, rel_path, prefix=qualname, parent=n, units=units)

    def _imports(self, n: Node, source: str, rel_path: str, prefix: str) -> Unit:
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        names: list[str] = []
        if n.type == "import_from_statement":
            mod = n.child_by_field_name("module_name")
            mod_name = source[mod.start_byte : mod.end_byte] if mod else ""
            for c in n.children_by_field_name("name"):
                names.append(
                    f"{mod_name}.{source[c.start_byte : c.end_byte].split(' as ')[0].strip()}"
                )
        else:
            for c in n.children_by_field_name("name"):
                names.append(
                    source[c.start_byte : c.end_byte].replace(" as ", " ").split()[0]
                )
        if not names:
            names = [source[n.start_byte : n.end_byte]]
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=names[0].split(".")[-1] if names else "import",
            qualname=names[0] if names else source[n.start_byte : n.end_byte],
            signature=collapse_ws(source[n.start_byte : n.end_byte]),
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
    def _decorator_names(n: Node, source: str) -> str:
        out: list[str] = []
        for ch in n.children:
            if ch.type == "decorator":
                inner = ch.named_children[0] if ch.named_children else None
                if inner:
                    out.append(source[inner.start_byte : inner.end_byte])
        return ", ".join(out)
