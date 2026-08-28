"""Rust extractor: functions, methods, types, imports, call sites via tree-sitter."""

from __future__ import annotations

from tree_sitter import Node

from ..models import UNIT_KIND_SYMBOL, Unit
from .base import ByteIndexedSource, Extractor, collapse_ws, valid_aliases, walk_calls
from .native_common import _concepts, _field, _make, _name_of, _parser, _point


class RustExtractor(Extractor):
    language = "rust"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        tree = _parser("rust").parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(
        self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_item":
                units.append(self._func(child, source, lines, prefix))
            elif t in ("struct_item", "enum_item", "trait_item"):
                self._type(child, source, lines, prefix, units)
            elif t == "impl_item":
                type_node = _field(child, "type") or next(
                    (c for c in child.named_children if c.type == "type_identifier"),
                    None,
                )
                impl_type = (
                    source[type_node.start_byte : type_node.end_byte] if type_node else "impl"
                )
                decl = next(
                    (c for c in child.named_children if c.type == "declaration_list"),
                    None,
                )
                if decl:
                    self._walk(decl, source, lines, impl_type, units)
            elif t == "use_declaration":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        source = ByteIndexedSource(source)
        tree = _parser("rust").parse(source.encode("utf-8"))
        out: list = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """`use path as alias;` -> (alias, path)."""
        source = ByteIndexedSource(source)
        out: list[tuple[str, str]] = []
        tree = _parser("rust").parse(source.encode("utf-8"))
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "use_as_clause":
                parts = cur.named_children
                if len(parts) < 2:
                    continue
                path_node, alias_node = parts[0], parts[-1]
                alias = source[alias_node.start_byte : alias_node.end_byte]
                path = source[path_node.start_byte : path_node.end_byte]
                parent = cur.parent
                prefixes: list[str] = []
                while parent is not None:
                    if parent.type == "scoped_use_list" and parent.named_children:
                        base = parent.named_children[0]
                        prefixes.insert(0, source[base.start_byte : base.end_byte])
                    parent = parent.parent
                if prefixes and "::" not in path:
                    path = "::".join([*prefixes, path])
                if alias.isidentifier() and path:
                    out.append((alias, path))
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _func(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        body = _field(n, "body")
        qualname = f"{prefix}::{name}" if prefix else name
        return _make(
            n,
            source,
            lines,
            "",
            "method" if prefix else "function",
            name,
            qualname,
            body,
            concepts=_concepts(n, source),
        )

    def _type(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name = _name_of(n, source)
        qualname = f"{prefix}::{name}" if prefix else name
        body = next(
            (
                c
                for c in n.named_children
                if c.type in ("field_declaration_list", "enum_variant_list", "declaration_list")
            ),
            None,
        )
        unit_type = {
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
        }[n.type]
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("use ", "").replace(";", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=name.split("::")[-1],
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
