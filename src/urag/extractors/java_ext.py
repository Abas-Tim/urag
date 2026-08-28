"""Java extractor: classes, methods, types, imports, calls, references."""

from __future__ import annotations

from tree_sitter import Node

from ..models import UNIT_KIND_SYMBOL, Reference, Unit
from .base import (
    ByteIndexedSource,
    Extractor,
    collapse_ws,
    dedupe_refs,
    split_callee,
    valid_aliases,
)
from .native_common import (
    _concepts,
    _field,
    _make,
    _name_of,
    _parser,
    _point,
    _walk_invocations,
)


class JavaExtractor(Extractor):
    language = "java"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        tree = _parser("java").parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(
        self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        for child in node.named_children:
            t = child.type
            if t in (
                "method_declaration",
                "constructor_declaration",
                "compact_constructor_declaration",
            ):
                units.append(self._method(child, source, lines, prefix))
            elif t in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ):
                self._type(child, source, lines, prefix, units)
            elif t == "import_declaration":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        source = ByteIndexedSource(source)
        tree = _parser("java").parse(source.encode("utf-8"))
        out: list = []
        _walk_invocations(tree.root_node, source, out, {"method_invocation"})
        from ..models import CallSite

        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "object_creation_expression":
                type_node = cur.child_by_field_name("type")
                if type_node is not None:
                    text = source[type_node.start_byte : type_node.end_byte]
                    out.append(
                        CallSite(
                            callee=split_callee(text),
                            callee_full=text,
                            line=cur.start_point.row + 1,
                            byte_start=cur.start_byte,
                            byte_end=cur.end_byte,
                        )
                    )
            stack.extend(reversed(cur.named_children))
        return out

    def collect_references(self, source: str) -> list:
        """Java: constructions, declared types, bases/interfaces, generics, casts."""
        import re

        from ..models import Reference

        source = ByteIndexedSource(source)
        tree = _parser("java").parse(source.encode("utf-8"))
        out: list[Reference] = []
        primitives = {
            "boolean",
            "byte",
            "char",
            "double",
            "float",
            "int",
            "long",
            "short",
            "void",
        }

        def append(node, kind: str) -> None:
            if node is None or node.type in (
                "integral_type",
                "floating_point_type",
                "boolean_type",
                "void_type",
            ):
                return
            text = source[node.start_byte : node.end_byte]
            base = text.split("<", 1)[0].strip()
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", base)
            if not names or names[-1] in primitives:
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

        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            t = cur.type
            if t == "object_creation_expression":
                append(cur.child_by_field_name("type"), "construct")
            elif t in (
                "formal_parameter",
                "field_declaration",
                "method_declaration",
                "local_variable_declaration",
                "variable_declarator",
                "cast_expression",
                "type_cast",
            ):
                append(cur.child_by_field_name("type"), "type")
            elif t == "class_declaration":
                for field in ("superclass", "superinterfaces"):
                    node = cur.child_by_field_name(field)
                    if node is None:
                        continue
                    children = node.named_children if field == "superinterfaces" else [node]
                    for child in children:
                        append(child, "base")
            elif t == "generic_type":
                args = cur.child_by_field_name("type_arguments") or next(
                    (c for c in cur.named_children if c.type == "type_arguments"),
                    None,
                )
                if args is not None:
                    for child in args.named_children:
                        append(child, "generic")
            stack.extend(reversed(cur.named_children))
        return dedupe_refs(out)

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        source = ByteIndexedSource(source)
        tree = _parser("java").parse(source.encode("utf-8"))
        out: list[tuple[str, str]] = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "import_declaration":
                raw = source[cur.start_byte : cur.end_byte]
                imported = next(
                    (child for child in cur.named_children if child.type == "scoped_identifier"),
                    None,
                )
                if imported and ".*" not in raw:
                    target = source[imported.start_byte : imported.end_byte]
                    if not target.endswith(".*"):
                        out.append((target.rsplit(".", 1)[-1], target))
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _method(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        body = _field(n, "body")
        qualname = f"{prefix}.{name}" if prefix else name
        return _make(
            n,
            source,
            lines,
            "",
            "constructor" if n.type == "constructor_declaration" else "method",
            name,
            qualname,
            body,
            concepts=_concepts(n, source),
        )

    def _type(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name = _name_of(n, source)
        qualname = f"{prefix}.{name}" if prefix else name
        body_field = {
            "class_declaration": "body",
            "interface_declaration": "body",
            "record_declaration": "body",
        }.get(n.type)
        body = _field(n, body_field) if body_field else None
        unit_type = n.type.replace("_declaration", "")
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))
        if body:
            self._walk(body, source, lines, qualname, units)

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("import ", "").replace(";", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=name.split(".")[-1],
            qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]),
            summary="import",
            concepts=name,
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )
