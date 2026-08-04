"""Extractor base + shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import CallSite, Unit

MAX_SUMMARY_CHARS = 220


class Extractor(ABC):
    language: str = ""

    @abstractmethod
    def extract(self, source: str, rel_path: str) -> list[Unit]:
        """Extract searchable units from source text."""

    def collect_calls(self, source: str) -> list[CallSite]:
        """Collect call sites. Default: no calls known."""
        return []


def split_callee(full: str) -> str:
    """Last segment of a dotted / :: / / chain."""
    for sep in (".", "::", "/", "global::"):
        if sep in full:
            return full.rsplit(sep, 1)[-1]
    return full


def walk_calls(node, source: str, call_types: set[str], out: list[CallSite], fn_field: str = "function") -> None:
    """Generic call collector: finds call nodes and reads the callee chain."""
    stack: list = [node]
    while stack:
        cur = stack.pop()
        if cur.type in call_types:
            fn = cur.child_by_field_name(fn_field)
            if fn is not None:
                full = source[fn.start_byte : fn.end_byte]
                out.append(
                    CallSite(
                        callee=split_callee(full),
                        callee_full=full,
                        line=cur.start_point.row + 1,
                        byte_start=cur.start_byte,
                        byte_end=cur.end_byte,
                    )
                )
        stack.extend(reversed(cur.named_children))
    return out


def collapse_ws(text: str, max_len: int = 300) -> str:
    """Collapse whitespace/newlines in a signature into a single line."""
    out = " ".join(text.split())
    return out[:max_len]


def _is_comment_line(line: str, marker: str) -> bool:
    if marker in ("//", "#"):
        return line.startswith(marker)
    if marker == "/*":
        return line.startswith(("/*", "*", "*/"))
    return line.startswith(marker)


def leading_comments(source_lines: list[str], start_line: int, marker: str) -> str:
    """Collect contiguous comment lines immediately above start_line (1-based)."""
    out: list[str] = []
    i = start_line - 2
    while i >= 0:
        stripped = source_lines[i].strip()
        if _is_comment_line(stripped, marker):
            cleaned = stripped.strip("/*").strip().lstrip("*").strip()
            out.append(cleaned)
            i -= 1
        elif stripped == "" and out:
            i -= 1
        else:
            break
    return " ".join(reversed(out))[:MAX_SUMMARY_CHARS]
