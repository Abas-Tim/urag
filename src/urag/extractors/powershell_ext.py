"""PowerShell extractor: function definitions (best-effort, regex based).

No tree-sitter grammar is bundled for PowerShell; functions are found
by their `function Name {` header and spans end at the matching brace.
This restores searchability for build/qualify scripts that would otherwise be
invisible to the index.
"""

from __future__ import annotations

import re

from ..models import UNIT_KIND_SYMBOL, Unit
from .base import Extractor, collapse_ws, leading_comments

_FUNCTION = re.compile(
    r"^\s*function\s+([A-Za-z0-9_\-\.:]+)\s*(?:\(([^)]*)\))?\s*\{?",
    re.IGNORECASE,
)


def _byte_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")) + 1)
    return offsets


class PowerShellExtractor(Extractor):
    language = "powershell"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        i = 0
        while i < len(lines):
            m = _FUNCTION.match(lines[i])
            if not m:
                i += 1
                continue
            name = m.group(1)
            start_line = i + 1
            depth = m.group(0).count("{") - m.group(0).count("}")
            j = i
            if depth == 0 and "{" not in m.group(0):
                depth = 1
            while j + 1 < len(lines) and depth > 0:
                j += 1
                depth += lines[j].count("{") - lines[j].count("}")
            end_line = j + 1
            params = collapse_ws(m.group(2) or "", 120)
            summary = leading_comments(lines, start_line, "#") or leading_comments(
                lines, start_line, "<#"
            )
            units.append(
                Unit(
                    file_id=0,
                    kind=UNIT_KIND_SYMBOL,
                    unit_type="function",
                    name=name,
                    qualname=name,
                    signature=collapse_ws(f"function {name} {params}".strip(), 220),
                    summary=summary,
                    concepts=params,
                    start_line=start_line,
                    end_line=end_line,
                    byte_start=offsets[start_line - 1],
                    byte_end=offsets[end_line - 1] + len(lines[end_line - 1].encode("utf-8")),
                    relationships="",
                )
            )
            i = j + 1
        return units
