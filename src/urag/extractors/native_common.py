"""Shared helpers for the native-language tree-sitter extractors.

Grammar compilation is cached per language; the unit model, span helpers,
signature/doc extraction, and invocation walking are shared by the Go, Rust,
Java, C/C++ and C# extractors.
"""

from __future__ import annotations

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_rust as tsrust
from tree_sitter import Language, Node, Parser

from ..models import UNIT_KIND_SYMBOL, Unit
from .base import collapse_ws, leading_comments, split_callee

_PARSERS: dict[str, Parser] = {}

_IDENT_TYPES = {"identifier", "type_identifier", "field_identifier"}


def _parser(language: str) -> Parser:
    if language not in _PARSERS:
        if language == "csharp":
            import tree_sitter_c_sharp as tscs

            mod = tscs
        else:
            mod = {
                "go": tsgo,
                "rust": tsrust,
                "java": tsjava,
                "c": tsc,
                "cpp": tscpp,
            }[language]
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


def _walk_invocations(
    node: Node,
    source: str,
    out: list,
    invoc_types: set[str],
    name_field: str = "name",
    object_field: str = "object",
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


