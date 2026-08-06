"""Extractors for Go, Rust, Java, C and C++ via tree-sitter.

Shares the urag unit model: functions/methods/classes/structs/interfaces/
enums/imports with signatures, doc comments, byte-precise spans and
qualname prefixes (receiver types, impl types, namespaces, classes).
"""

from __future__ import annotations

from tree_sitter import Language, Node, Parser
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
import tree_sitter_java as tsjava
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

from ..models import Unit, UNIT_KIND_SYMBOL
from .base import Extractor, collapse_ws, leading_comments, split_callee, valid_aliases, walk_calls

_PARSERS: dict[str, Parser] = {}

_IDENT_TYPES = {"identifier", "type_identifier", "field_identifier"}

_CALL_TYPES = {"call_expression": "function", "method_invocation": None}


def _parser(language: str) -> Parser:
    if language not in _PARSERS:
        mod = {"go": tsgo, "rust": tsrust, "java": tsjava, "c": tsc, "cpp": tscpp}[language]
        p = Parser()
        p.language = Language(mod.language())
        _PARSERS[language] = p
    return _PARSERS[language]


def _point(n: Node, which: str) -> tuple[int, int]:
    p = getattr(n, f"{which}_point")
    return (p.row + 1, p.column)


def _field(n: Node, name: str) -> Node | None:
    return n.child_by_field_name(name)


def _doc(lines: list[str], start_line: int) -> str:
    return leading_comments(lines, start_line, "//") or leading_comments(lines, start_line, "/*")


def _concepts(n: Node, source: str, limit: int = 6) -> str:
    out: list[str] = []
    stack: list[Node] = [n]
    while stack and len(out) < limit:
        cur = stack.pop()
        for ch in cur.named_children:
            if ch.type in _IDENT_TYPES or ch.type == "primitive_type":
                out.append(source[ch.start_byte : ch.end_byte])
            elif len(out) < limit:
                stack.append(ch)
    return ", ".join(out)


def _signature(n: Node, source: str, body: Node | None) -> str:
    end = body.start_byte if body else n.end_byte
    return collapse_ws(source[n.start_byte : end])


def _make(
    n: Node,
    source: str,
    lines: list[str],
    rel_path: str,
    unit_type: str,
    name: str,
    qualname: str,
    body: Node | None,
    concepts: str = "",
) -> Unit:
    (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
    return Unit(
        file_id=0,
        kind=UNIT_KIND_SYMBOL,
        unit_type=unit_type,
        name=name,
        qualname=qualname,
        signature=_signature(n, source, body),
        summary=_doc(lines, sl),
        concepts=concepts,
        start_line=sl,
        end_line=el,
        start_col=sc,
        end_col=ec,
        byte_start=n.start_byte,
        byte_end=n.end_byte,
    )


def _name_of(n: Node, source: str) -> str:
    name = _field(n, "name")
    if name:
        return source[name.start_byte : name.end_byte]
    stack: list[Node] = [n]
    while stack:
        cur = stack.pop()
        if cur.type in _IDENT_TYPES:
            return source[cur.start_byte : cur.end_byte]
        stack.extend(reversed(cur.named_children))
    return "anonymous"


def _walk_units(
    node: Node, source: str, lines: list[str], prefix: str, units: list[Unit],
    func_types: set[str], class_types: dict[str, str], walk_fn,
) -> None:
    for child in node.named_children:
        walk_fn(child, source, lines, prefix, units)


def _walk_invocations(
    node: Node, source: str, out: list, invoc_types: set[str],
    name_field: str = "name", object_field: str = "object",
) -> None:
    """Collect calls for grammars where the callee is a `name` field."""
    from ..models import CallSite

    stack: list[Node] = [node]
    while stack:
        cur = stack.pop()
        if cur.type in invoc_types:
            name = cur.child_by_field_name(name_field)
            if name is not None:
                full = source[name.start_byte : name.end_byte]
                obj = cur.child_by_field_name(object_field)
                chain = f"{source[obj.start_byte : obj.end_byte]}.{full}" if obj else full
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


class GoExtractor(Extractor):
    language = "go"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        tree = _parser("go").parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
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
        tree = _parser("go").parse(source.encode("utf-8"))
        out: list = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """`import alias "path"` -> (alias, dotted path)."""
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
        params = _field(n, "parameters") or _field(n, "result")
        qualname = f"{prefix}.{name}" if prefix else name
        return _make(n, source, lines, "", "method" if prefix else "function", name, qualname, body,
                     concepts=_concepts(params, source))

    def _type_spec(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name = _name_of(n, source)
        type_node = _field(n, "type") or next((c for c in n.named_children if c.type in ("struct_type", "interface_type")), None)
        unit_type = "struct" if type_node and type_node.type == "struct_type" else "interface"
        qualname = f"{prefix}.{name}" if prefix else name
        body = None
        if type_node:
            body = next((c for c in type_node.named_children if c.type in ("field_declaration_list", "method_spec_list")), None)
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
            file_id=0, kind=UNIT_KIND_SYMBOL, unit_type="import",
            name=first.split(".")[-1].split("/")[-1], qualname=first,
            signature=collapse_ws(source[n.start_byte : n.end_byte]), summary="import",
            concepts=", ".join(names[:6]), start_line=sl, end_line=el,
            start_col=sc, end_col=ec, byte_start=n.start_byte, byte_end=n.end_byte,
        )


class RustExtractor(Extractor):
    language = "rust"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        tree = _parser("rust").parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_item":
                units.append(self._func(child, source, lines, prefix))
            elif t in ("struct_item", "enum_item", "trait_item"):
                self._type(child, source, lines, prefix, units)
            elif t == "impl_item":
                type_node = _field(child, "type") or next((c for c in child.named_children if c.type == "type_identifier"), None)
                impl_type = source[type_node.start_byte : type_node.end_byte] if type_node else "impl"
                decl = next((c for c in child.named_children if c.type == "declaration_list"), None)
                if decl:
                    self._walk(decl, source, lines, impl_type, units)
            elif t == "use_declaration":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        tree = _parser("rust").parse(source.encode("utf-8"))
        out: list = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """`use path as alias;` -> (alias, path)."""
        out: list[tuple[str, str]] = []
        tree = _parser("rust").parse(source.encode("utf-8"))
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "use_declaration":
                text = source[cur.start_byte : cur.end_byte]
                body = text.strip().rstrip(";").removeprefix("use ").strip()
                if " as " in body:
                    path, alias = [p.strip() for p in body.rsplit(" as ", 1)]
                    if alias.isidentifier() and path:
                        out.append((alias, path))
            stack.extend(reversed(cur.named_children))
        return valid_aliases(out)

    def _func(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        body = _field(n, "body")
        params = _field(n, "parameters")
        qualname = f"{prefix}::{name}" if prefix else name
        return _make(n, source, lines, "", "method" if prefix else "function", name, qualname, body,
                     concepts=_concepts(params, source))

    def _type(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name = _name_of(n, source)
        qualname = f"{prefix}::{name}" if prefix else name
        body = next((c for c in n.named_children if c.type in ("field_declaration_list", "enum_variant_list", "declaration_list")), None)
        unit_type = {"struct_item": "struct", "enum_item": "enum", "trait_item": "trait"}[n.type]
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("use ", "").replace(";", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0, kind=UNIT_KIND_SYMBOL, unit_type="import",
            name=name.split("::")[-1], qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]), summary="import",
            concepts=name, start_line=sl, end_line=el,
            start_col=sc, end_col=ec, byte_start=n.start_byte, byte_end=n.end_byte,
        )


class JavaExtractor(Extractor):
    language = "java"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        tree = _parser("java").parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        for child in node.named_children:
            t = child.type
            if t in ("method_declaration", "constructor_declaration", "compact_constructor_declaration"):
                units.append(self._method(child, source, lines, prefix))
            elif t in ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"):
                self._type(child, source, lines, prefix, units)
            elif t == "import_declaration":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        tree = _parser("java").parse(source.encode("utf-8"))
        out: list = []
        _walk_invocations(tree.root_node, source, out, {"method_invocation"})
        return out

    def _method(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = _name_of(n, source)
        body = _field(n, "body")
        params = _field(n, "parameters")
        qualname = f"{prefix}.{name}" if prefix else name
        return _make(n, source, lines, "", "constructor" if n.type == "constructor_declaration" else "method",
                     name, qualname, body, concepts=_concepts(params, source))

    def _type(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        name = _name_of(n, source)
        qualname = f"{prefix}.{name}" if prefix else name
        body_field = {"class_declaration": "body", "interface_declaration": "body", "record_declaration": "body"}.get(n.type)
        body = _field(n, body_field) if body_field else None
        unit_type = n.type.replace("_declaration", "")
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))
        if body:
            self._walk(body, source, lines, qualname, units)

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("import ", "").replace(";", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0, kind=UNIT_KIND_SYMBOL, unit_type="import",
            name=name.split(".")[-1], qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]), summary="import",
            concepts=name, start_line=sl, end_line=el,
            start_col=sc, end_col=ec, byte_start=n.start_byte, byte_end=n.end_byte,
        )


class CExtractor(Extractor):
    """C and C++ (mode='cpp' adds namespaces, classes, templates)."""

    def __init__(self, language: str = "c"):
        self.language = language

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        tree = _parser(self.language).parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def _walk(self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_definition":
                units.append(self._func(child, source, lines, prefix))
            elif t == "template_declaration":
                decl = next((c for c in child.named_children if c.type in ("function_definition", "class_specifier", "struct_specifier", "type_definition")), None)
                if decl and decl.type == "function_definition":
                    units.append(self._func(decl, source, lines, prefix))
                elif decl and decl.type in ("class_specifier", "struct_specifier"):
                    self._type(decl, source, lines, prefix, units)
            elif t in ("class_specifier", "struct_specifier", "union_specifier", "enum_specifier"):
                self._type(child, source, lines, prefix, units)
            elif t == "type_definition":
                self._typedef(child, source, lines, prefix, units)
            elif t == "namespace_definition":
                ns = _field(child, "name") or next((c for c in child.named_children if c.type == "namespace_identifier"), None)
                ns_name = source[ns.start_byte : ns.end_byte] if ns else "ns"
                decl = next((c for c in child.named_children if c.type == "declaration_list"), None)
                if decl:
                    self._walk(decl, source, lines, f"{prefix}{ns_name}::" if prefix else f"{ns_name}::", units)
            elif t == "preproc_include":
                units.append(self._import(child, source))

    def collect_calls(self, source: str) -> list:
        tree = _parser(self.language).parse(source.encode("utf-8"))
        out: list = []
        walk_calls(tree.root_node, source, {"call_expression"}, out)
        return out

    def _func(self, n: Node, source: str, lines: list[str], prefix: str, is_method: bool = False) -> Unit:
        declarator = _field(n, "declarator")
        body = _field(n, "body")
        params = None
        name = "anonymous"
        if declarator:
            params = next((c for c in declarator.named_children if c.type == "parameter_list"), None)
            name = self._declarator_name(declarator, params, source)
        qualname = f"{prefix}{name}" if prefix else name
        unit_type = "method" if is_method else "function"
        return _make(n, source, lines, "", unit_type, name, qualname, body,
                     concepts=_concepts(params, source) if params else "")

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
        name_node = _field(n, "name") or next((c for c in n.named_children if c.type in _IDENT_TYPES), None)
        if name_node is None:
            return
        name = source[name_node.start_byte : name_node.end_byte]
        qualname = f"{prefix}{name}" if prefix else name
        body = next((c for c in n.named_children if c.type == "field_declaration_list"), None)
        unit_type = n.type.replace("_specifier", "")
        units.append(_make(n, source, lines, "", unit_type, name, qualname, body))
        if self.language == "cpp" and body and n.type in ("class_specifier", "struct_specifier"):
            for member in body.named_children:
                if member.type == "function_definition":
                    units.append(self._func(member, source, lines, qualname + "::", is_method=True))
                elif member.type == "field_declaration":
                    declarator = next((c for c in member.named_children if c.type == "function_declarator"), None)
                    if declarator:
                        params = next((c for c in declarator.named_children if c.type == "parameter_list"), None)
                        name = self._declarator_name(declarator, params, source)
                        if name != "anonymous":
                            m = _make(member, source, lines, "", "method", name, qualname + "::" + name, None,
                                      concepts=_concepts(params, source) if params else "")
                            units.append(m)

    def _typedef(self, n: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        spec = next((c for c in n.named_children if c.type in ("struct_specifier", "union_specifier", "enum_specifier")), None)
        name_node = next((c for c in n.named_children if c.type == "type_identifier"), None)
        name = source[name_node.start_byte : name_node.end_byte] if name_node else "anonymous"
        qualname = f"{prefix}{name}" if prefix else name
        body = next((c for c in (spec.named_children if spec else []) if c.type == "field_declaration_list"), None)
        units.append(_make(n, source, lines, "", "typedef", name, qualname, body))

    def _import(self, n: Node, source: str) -> Unit:
        name = source[n.start_byte : n.end_byte].replace("#include", "").strip()
        (sl, sc), (el, ec) = _point(n, "start"), _point(n, "end")
        return Unit(
            file_id=0, kind=UNIT_KIND_SYMBOL, unit_type="import",
            name=name.strip('"<>').split("/")[-1], qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]), summary="include",
            concepts=name, start_line=sl, end_line=el,
            start_col=sc, end_col=ec, byte_start=n.start_byte, byte_end=n.end_byte,
        )


class CSharpExtractor(Extractor):
    """C#: classes, interfaces, methods, constructors, usings, call sites."""

    language = "csharp"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        import tree_sitter_c_sharp as tscs

        p = Parser()
        p.language = Language(tscs.language())
        tree = p.parse(source.encode("utf-8"))
        lines = source.splitlines()
        units: list[Unit] = []
        self._walk(tree.root_node, source, lines, prefix="", units=units)
        return units

    def collect_calls(self, source: str) -> list:
        import tree_sitter_c_sharp as tscs

        p = Parser()
        p.language = Language(tscs.language())
        tree = p.parse(source.encode("utf-8"))
        out: list = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "invocation_expression":
                first = cur.named_children[0] if cur.named_children else None
                if first is not None:
                    if first.type == "member_access_expression":
                        chain = source[first.start_byte : first.end_byte]
                    elif first.type in ("identifier", "generic_name"):
                        chain = source[first.start_byte : first.end_byte]
                    else:
                        chain = None
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

    def collect_import_aliases(self, source: str) -> list[tuple[str, str]]:
        """`using Alias = Namespace.Type;` -> (Alias, Namespace.Type)."""
        import tree_sitter_c_sharp as tscs

        p = Parser()
        p.language = Language(tscs.language())
        tree = p.parse(source.encode("utf-8"))
        out: list[tuple[str, str]] = []
        stack: list[Node] = [tree.root_node]
        while stack:
            cur = stack.pop()
            if cur.type == "using_directive":
                parts = [c for c in cur.named_children if c.type in ("identifier", "qualified_name")]
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

    def _walk(self, node: Node, source: str, lines: list[str], prefix: str, units: list[Unit]) -> None:
        for child in node.named_children:
            t = child.type
            if t in ("class_declaration", "interface_declaration", "struct_declaration", "record_declaration", "enum_declaration"):
                self._type(child, source, lines, prefix, units)
            elif t in ("method_declaration", "constructor_declaration"):
                units.append(self._method(child, source, lines, prefix))
            elif t == "namespace_declaration":
                ns = next((c for c in child.named_children if c.type in ("qualified_name", "identifier")), None)
                ns_name = source[ns.start_byte : ns.end_byte] if ns else "ns"
                decl = next((c for c in child.named_children if c.type == "declaration_list"), None)
                if decl:
                    self._walk(decl, source, lines, f"{prefix}{ns_name}." if prefix else f"{ns_name}.", units)
            elif t == "using_directive":
                units.append(self._import(child, source))

    def _method(self, n: Node, source: str, lines: list[str], prefix: str) -> Unit:
        name = self._method_name(n, source)
        body = next((c for c in n.named_children if c.type == "block"), None)
        params = next((c for c in n.named_children if c.type == "parameter_list"), None)
        qualname = f"{prefix}{name}" if prefix else name
        return _make(n, source, lines, "", "constructor" if n.type == "constructor_declaration" else "method",
                     name, qualname, body, concepts=_concepts(params, source) if params else "")

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
            file_id=0, kind=UNIT_KIND_SYMBOL, unit_type="import",
            name=name.split(".")[-1], qualname=name,
            signature=collapse_ws(source[n.start_byte : n.end_byte]), summary="using",
            concepts=name, start_line=sl, end_line=el,
            start_col=sc, end_col=ec, byte_start=n.start_byte, byte_end=n.end_byte,
        )
