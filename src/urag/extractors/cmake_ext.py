"""CMake extractor: options, cache variables, targets, tests, dependencies.

CMake files (CMakeLists.txt, *.cmake) are indexed line-based: each command
invocation becomes a searchable unit so agents can answer "which option
controls X / which target links Y" without reading whole files.
"""

from __future__ import annotations

import re

from ..models import UNIT_KIND_SYMBOL, Unit
from .base import Extractor, collapse_ws
from .config_ext import _byte_offsets

MAX_ARGS_CHARS = 220

_COMMAND = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)

_KINDS: dict[str, str] = {
    "option": "config_key",
    "set": "config_key",
    "add_executable": "target",
    "add_library": "target",
    "add_custom_target": "target",
    "add_test": "test",
    "find_package": "dependency",
}


def _first_arg(args: str) -> str:
    for token in re.split(r"\s+", args.strip()):
        token = token.strip('"')
        if not token or token.upper() in ("CACHE", "BOOL", "STRING"):
            continue
        if token.startswith("${"):
            continue
        return token
    return ""


def _help_text(args: str) -> str:
    m = re.search(r'"([^"]+)"', args)
    if m:
        return m.group(1)
    return collapse_ws(args, MAX_ARGS_CHARS)


class CMakeExtractor(Extractor):
    language = "cmake"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        i = 0
        while i < len(lines):
            m = _COMMAND.match(lines[i])
            if not m or m.group(1).lower() not in _KINDS:
                i += 1
                continue
            command = m.group(1).lower()
            start_line = i + 1
            text = lines[i]
            j = i
            while text.count("(") > text.count(")") and j + 1 < len(lines):
                j += 1
                text += "\n" + lines[j]
            inner = text[text.index("(") + 1 :]
            depth = 1
            end = len(inner)
            for k, ch in enumerate(inner):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
            args = inner[:end]
            name = _first_arg(args)
            if name:
                unit_type = _KINDS[command]
                summary = (
                    _help_text(args) if command == "option" else collapse_ws(args, MAX_ARGS_CHARS)
                )
                units.append(
                    Unit(
                        file_id=0,
                        kind=UNIT_KIND_SYMBOL,
                        unit_type=unit_type,
                        name=name.strip('"'),
                        qualname=name.strip('"'),
                        signature=collapse_ws(text, 220),
                        summary=summary,
                        concepts=collapse_ws(args, MAX_ARGS_CHARS),
                        start_line=start_line,
                        end_line=j + 1,
                        byte_start=offsets[start_line - 1],
                        byte_end=offsets[j + 1],
                    )
                )
            i = j + 1
        return units
