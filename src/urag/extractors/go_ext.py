"""Go extractor: functions, methods, types, imports, call sites via tree-sitter."""

from __future__ import annotations

from tree_sitter import Node

from ..models import UNIT_KIND_SYMBOL, Unit
from .base import ByteIndexedSource, Extractor, collapse_ws, valid_aliases, walk_calls
from .native_common import _concepts, _field, _make, _name_of, _parser, _point


class GoExtractor(Extractor):
    language = "go"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        tree = _parser("go").parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(
        self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_declaration":
                units.append(self._func(child, source, lines, prefix))
            elif t == "method_declaration":
                recv = _field(child, "receiver")
                recv_type = ""
                if recv:
                    recv_text = source[recv.start_byte : recv.end_byte]
                    parts = recv_text.strip("()").replace("*", "").split()
                    recv_type = parts[-1] if parts else ""
                units.append(self._func(child, source, lines, recv_type or prefix))
            elif t == "type_declaration":
                for spec in child.named_children:
                    if spec.type == "type_spec":
                        self._type_spec(spec, source, lines, prefix, units)
            elif t == "import_declaration":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        source = ByteIndexedSource(source)
        tree = _parser("go").parse(source.encode("utf-8"))
        out: list = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """`import alias "path"` -> (alias, dotted path)."""
        source = ByteIndexedSource(source)
        tree = _parser("go").parse(source.encode("utf-8"))
        out: list[tuple[str, str]] = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "import_spec":
                alias = _field(cur, "name")
                path = _field(cur, "path")
                if alias is not None and path is not None:
                    out.append(
                        (
                            source[alias.start_byte : alias.end_byte],
                            source[path.start_byte : path.end_byte].strip('"').replace("/", "."),
                        )
                    )
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _func(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        body = _field(n, "body")
        qualname = f"{prefix}.{name}" if prefix else name
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

    def _type_spec(
        self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        name = _name_of(n, source)
        type_node = _field(n, "type") or next(
            (c for c in n.named_children if c.type in ("struct_type", "interface_type")),
            None,
        )
        unit_type = "struct" if type_node and type_node.type == "struct_type" else "interface"
        qualname = f"{prefix}.{name}" if prefix else name
        body = None
        if type_node:
            body = next(
                (
                    c
                    for c in type_node.named_children
                    if c.type in ("field_declaration_list", "method_spec_list")
                ),
                None,
            )
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))

    def _import(self, n: Node, source: str) -> Unit:
        specs = [c for c in n.named_children if c.type == "import_spec"]
        names = []
        for s in specs:
            path = source[s.start_byte : s.end_byte].strip('"').strip()
            alias = _field(s, "name")
            names.append(f"{source[alias.start_byte : alias.end_byte]}:{path}" if alias else path)
        first = names[0] if names else "import"
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=first.split(".")[-1].split("/")[-1],
            qualname=first,
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
