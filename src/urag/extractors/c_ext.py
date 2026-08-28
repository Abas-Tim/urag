"""C/C++ extractor: functions, structs, enums, typedefs, imports, calls."""

from __future__ import annotations

from tree_sitter import Node

from ..models import UNIT_KIND_SYMBOL, Unit
from .base import ByteIndexedSource, Extractor, collapse_ws, walk_calls
from .native_common import _IDENT_TYPES, _concepts, _field, _make, _parser, _point


class CExtractor(Extractor):
    """C and C++ (mode='cpp' adds namespaces, classes, templates)."""

    def __init__(self, language: str = "c"):
        self.language = language
        self._class_qualnames: set[str] = set()

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        source = ByteIndexedSource(source)
        tree = _parser(self.language).parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._class_qualnames.clear()
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(
        self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_definition":
                self._append_function(units, self._func(child, source, lines, prefix))
            elif t == "template_declaration":
                decl = next(
                    (
                        c
                        for c in child.named_children
                        if c.type
                        in (
                            "function_definition",
                            "class_specifier",
                            "struct_specifier",
                            "type_definition",
                        )
                    ),
                    None,
                )
                if decl and decl.type == "function_definition":
                    self._append_function(units, self._func(decl, source, lines, prefix))
                elif decl and decl.type in ("class_specifier", "struct_specifier"):
                    self._type(decl, source, lines, prefix, units)
            elif t in (
                "class_specifier",
                "struct_specifier",
                "union_specifier",
                "enum_specifier",
            ):
                self._type(child, source, lines, prefix, units)
            elif t == "type_definition":
                self._typedef(child, source, lines, prefix, units)
            elif t == "namespace_definition":
                ns = _field(child, "name") or next(
                    (c for c in child.named_children if c.type == "namespace_identifier"),
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
                        f"{prefix}{ns_name}::" if prefix else f"{ns_name}::",
                        units,
                    )
            elif t == "preproc_include":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        source = ByteIndexedSource(source)
        tree = _parser(self.language).parse(source.encode("utf-8"))
        out: list = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def _func(
        self,
        n: Node,
        source: str,
        lines: list[str],
        prefix: str,
        is_method: bool = False,
    ) -> Unit:
        declarator = _field(n, "declarator")
        body = _field(n, "body")
        params = None
        name = "anonymous"
        if declarator:
            params = next(
                (c for c in declarator.named_children if c.type == "parameter_list"),
                None,
            )
            name = self._declarator_name(declarator, params, source)
        declared_prefix = ""
        if declarator:
            qualified = next(
                (c for c in declarator.named_children if c.type == "qualified_identifier"),
                None,
            )
            if qualified:
                qualified_name = source[qualified.start_byte : qualified.end_byte]
                owner, separator, qualified_method = qualified_name.rpartition("::")
                if separator:
                    declared_prefix = f"{owner}::"
                    name = qualified_method
        effective_prefix = f"{prefix}{declared_prefix}"
        qualname = f"{effective_prefix}{name}" if effective_prefix else name
        owner = effective_prefix.removesuffix("::")
        unit_type = "method" if is_method or owner in self._class_qualnames else "function"
        return _make(
            n,
            source,
            lines,
            "",
            unit_type,
            name,
            qualname,
            body,
            concepts=_concepts(n, source),
        )

    @staticmethod
    def _parameter_key(signature: str) -> str:
        start = signature.find("(")
        return signature[start:].rstrip(" ;") if start >= 0 else signature

    def _append_function(self, units: list[Unit], unit: Unit) -> None:
        if self.language != "cpp" or unit.unit_type != "method":
            units.append(unit)
            return
        key = (unit.qualname, self._parameter_key(unit.signature))
        for index, existing in enumerate(units):
            if (
                existing.unit_type == "method"
                and (existing.qualname, self._parameter_key(existing.signature)) == key
            ):
                if unit.byte_end - unit.byte_start > existing.byte_end - existing.byte_start:
                    units[index] = unit
                return
        units.append(unit)

    @staticmethod
    def _declarator_name(declarator: Node, params: Node | None, source: str) -> str:
        stack: list[Node] = [declarator]
        while stack:
            cur = stack.pop()
            if cur == params:
                continue
            if cur.type in _IDENT_TYPES:
                return source[cur.start_byte : cur.end_byte]
            stack.extend(reversed(cur.named_children))
        return "anonymous"

    def _type(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name_node = _field(n, "name") or next(
            (c for c in n.named_children if c.type in _IDENT_TYPES), None
        )
        if name_node is None:
            return
        name = source[name_node.start_byte : name_node.end_byte]
        qualname = f"{prefix}{name}" if prefix else name
        if self.language == "cpp" and n.type in ("class_specifier", "struct_specifier"):
            self._class_qualnames.add(qualname)
        body = next((c for c in n.named_children if c.type == "field_declaration_list"), None)
        unit_type = n.type.replace("_specifier", "")
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))
        if self.language == "cpp" and body and n.type in ("class_specifier", "struct_specifier"):
            for member in body.named_children:
                if member.type == "function_definition":
                    self._append_function(
                        units,
                        self._func(member, source, lines, qualname + "::", is_method=True),
                    )
                elif member.type == "field_declaration":
                    declarator = next(
                        (c for c in member.named_children if c.type == "function_declarator"),
                        None,
                    )
                    if declarator:
                        params = next(
                            (c for c in declarator.named_children if c.type == "parameter_list"),
                            None,
                        )
                        name = self._declarator_name(declarator, params, source)
                        if name != "anonymous":
                            m = _make(
                                member,
                                source,
                                lines,
                                "",
                                "method",
                                name,
                                qualname + "::" + name,
                                None,
                                concepts=_concepts(params, source) if params else "",
                            )
                            self._append_function(units, m)

    def _typedef(
        self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]
    ) -> None:
        spec = next(
            (
                c
                for c in n.named_children
                if c.type in ("struct_specifier", "union_specifier", "enum_specifier")
            ),
            None,
        )
        name_node = next((c for c in n.named_children if c.type == "type_identifier"), None)
        name = source[name_node.start_byte : name_node.end_byte] if name_node else "anonymous"
        qualname = f"{prefix}{name}" if prefix else name
        body = next(
            (
                c
                for c in (spec.named_children if spec else [])
                if c.type == "field_declaration_list"
            ),
            None,
        )
        units.append(_make(n, source, lines, "", "typedef", name, qualname, body))

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("#include", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="import",
            name=name.strip('"<>').split("/")[-1],
            qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]),
            summary="include",
            concepts=name,
            start_line=sl,
            end_line=el,
            start_col=sc,
            end_col=ec,
            byte_start=n.start_byte,
            byte_end=n.end_byte,
        )


