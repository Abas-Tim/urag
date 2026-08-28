"""C# extractor: classes, interfaces, methods, usings, calls, references."""

from __future__ import annotations

import re

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
from .native_common import _concepts, _field, _make, _name_of, _parser, _point


class CSharpExtractor(Extractor):
    """C#: classes, interfaces, methods, constructors, usings, call sites."""

    language = "csharp"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        p = _parser("csharp")
        tree = p.parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def collect_calls(self, source: str) -> list:
        source = ByteIndexedSource(source)
        p = _parser("csharp")
        tree = p.parse(source.encode("utf-8"))
        out: list = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            chain = None
            if cur.type == "invocation_expression":
                first = cur.named_children[0] if cur.named_children else None
                if first is not None and (
                    first.type == "member_access_expression"
                    or first.type in ("identifier", "generic_name")
                ):
                    chain = source[first.start_byte : first.end_byte]
            elif cur.type == "object_creation_expression":
                type_node = cur.child_by_field_name("type")
                if type_node is not None and type_node.type not in (
                    "predefined_type",
                    "implicit_type",
                ):
                    chain = source[type_node.start_byte : type_node.end_byte].split("<", 1)[0]
            elif cur.type == "assignment_expression":
                op = cur.child_by_field_name("operator")
                right = cur.child_by_field_name("right")
                if (
                    op is not None
                    and right is not None
                    and source[op.start_byte : op.end_byte] in ("+=", "-=")
                    and (right.type == "identifier" or right.type == "member_access_expression")
                ):
                    chain = source[right.start_byte : right.end_byte]
            if chain:
                from ..models import CallSite

                out.append(
                    CallSite(
                        callee=split_callee(chain),
                        callee_full=chain,
                        line=cur.start_point.row + 1,
                        byte_start=cur.start_byte,
                        byte_end=cur.end_byte,
                    )
                )
            stack.extend(reversed(cur.named_children))
        return out

    def collect_references(self, source: str) -> list:
        """Type references: constructions, declared types, bases, generics,
        casts, typeof, attributes, x:Class-style bindings are handled by the
        XML extractor."""
        source = ByteIndexedSource(source)
        p = _parser("csharp")
        tree = p.parse(source.encode("utf-8"))
        out: list = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            t = cur.type
            if t == "base_list":
                for child in cur.named_children:
                    self._append_ref(out, child, source, "base")
            elif t == "generic_name":
                args = cur.child_by_field_name("type_arguments") or next(
                    (c for c in cur.named_children if c.type == "type_argument_list"),
                    None,
                )
                if args is not None:
                    for child in args.named_children:
                        self._append_ref(out, child, source, "generic")
            elif t == "attribute":
                self._append_ref(out, cur.child_by_field_name("name"), source, "attribute")
            elif t == "object_creation_expression":
                self._append_ref(out, cur.child_by_field_name("type"), source, "construct")
            elif t in ("typeof_expression", "cast_expression", "declaration_pattern"):
                self._append_ref(out, cur.child_by_field_name("type"), source, "cast")
            elif t == "as_expression":
                self._append_ref(out, cur.child_by_field_name("right"), source, "cast")
            elif t in (
                "variable_declaration",
                "parameter",
                "property_declaration",
                "field_declaration",
            ):
                self._append_ref(out, cur.child_by_field_name("type"), source, "type")
            elif t == "method_declaration":
                self._append_ref(out, cur.child_by_field_name("returns"), source, "type")
            stack.extend(reversed(cur.named_children))
        return dedupe_refs(out)

    @staticmethod
    def _append_ref(out: list, node, source, kind: str) -> None:
        if node is None or node.type in ("predefined_type", "implicit_type"):
            return
        text = source[node.start_byte : node.end_byte]
        base = text.split("<", 1)[0].strip()
        names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", base)
        if not names:
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

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """`using Alias = Namespace.Type;` -> (Alias, Namespace.Type)."""
        source = ByteIndexedSource(source)
        p = _parser("csharp")
        tree = p.parse(source.encode("utf-8"))
        out: list[tuple[str, str]] = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "using_directive":
                parts = [
                    c for c in cur.named_children if c.type in ("identifier", "qualified_name")
                ]
                text = source[cur.start_byte : cur.end_byte]
                if "=" in text and len(parts) == 2:
                    out.append(
                        (
                            source[parts[0].start_byte : parts[0].end_byte],
                            source[parts[1].start_byte : parts[1].end_byte],
                        )
                    )
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _walk(
        self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        for child in node.named_children:
            t = child.type
            if t in (
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "record_declaration",
                "enum_declaration",
            ):
                self._type(child, source, lines, prefix, units)
            elif t in ("method_declaration", "constructor_declaration"):
                units.append(self._method(child, source, lines, prefix))
            elif t == "namespace_declaration":
                ns = next(
                    (c for c in child.named_children if c.type in ("qualified_name", "identifier")),
                    None,
                )
                ns_name = source[ns.start_byte : ns.end_byte] if ns else "ns"
                decl = next(
                    (c for c in child.named_children if c.type == "declaration_list"),
                    None,
                )
                if decl:
                    self._walk(
                        decl,
                        source,
                        lines,
                        f"{prefix}{ns_name}." if prefix else f"{ns_name}.",
                        units,
                    )
            elif t == "using_directive":
                units.append(self._import(child, source))

    def _method(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = self._method_name(n, source)
        body = next((c for c in n.named_children if c.type == "block"), None)
        qualname = f"{prefix}{name}" if prefix else name
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

    @staticmethod
    def _method_name(n: Node, source: str) -> str:
        children = n.named_children
        for i, c in enumerate(children):
            if c.type == "identifier":
                nxt = children[i + 1] if i + 1 < len(children) else None
                if nxt and nxt.type in ("parameter_list", "type_parameter_list"):
                    return source[c.start_byte : c.end_byte]
        name = _field(n, "name")
        return source[name.start_byte : name.end_byte] if name else "anonymous"

    def _type(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name = _name_of(n, source)
        qualname = f"{prefix}{name}" if prefix else name
        body = next((c for c in n.named_children if c.type == "declaration_list"), None)
        unit_type = n.type.replace("_declaration", "")
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))
        if body:
            self._walk(body, source, lines, qualname + ".", units)

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("using ", "").replace(";", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=name.split(".")[-1],
            qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]),
            summary="using",
            concepts=name,
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )
