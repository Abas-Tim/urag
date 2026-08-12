"""Markdown extractor: heading-based doc chunks with hierarchy in qualname."""

from __future__ import annotations

import re

from ..models import Unit, UNIT_KIND_CHUNK
from .base import Extractor, MAX_SUMMARY_CHARS, collapse_ws

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MAX_CHUNK_LINES = 150


class MarkdownExtractor(Extractor):
    language = "markdown"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode("utf-8")))
        headings: list[tuple[int, int, str, str]] = []
        in_fence = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                headings.append((i, level, title, "#" * level + " " + title))
        if not headings:
            return self._chunk(
                lines, offsets, 0, len(lines), f"#{rel_path}", "", rel_path
            )
        units: list[Unit] = []
        if headings[0][0] > 0:
            units.extend(
                self._chunk(
                    lines,
                    offsets,
                    0,
                    headings[0][0],
                    f"#{rel_path}#preamble",
                    "",
                    rel_path,
                )
            )
        for idx, (line_no, level, title, full) in enumerate(headings):
            end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
            chunks = self._chunk(
                lines, offsets, line_no, end, f"#{rel_path}#{title}", full, rel_path
            )
            units.extend(chunks)
        return units

    def _chunk(
        self,
        lines: list[str],
        offsets: list[int],
        start: int,
        end: int,
        qualname: str,
        title: str,
        rel_path: str,
    ) -> list[Unit]:
        content = lines[start:end]
        out: list[Unit] = []
        if not content:
            return out
        for i in range(0, len(content), MAX_CHUNK_LINES):
            seg = content[i : i + MAX_CHUNK_LINES]
            out.append(
                self._make(
                    lines,
                    offsets,
                    start + i,
                    start + i + len(seg),
                    qualname,
                    title,
                    seg,
                    rel_path,
                )
            )
        return out

    def _make(
        self,
        lines: list[str],
        offsets: list[int],
        start: int,
        end: int,
        qualname: str,
        title: str,
        body: list[str],
        rel_path: str,
    ) -> Unit:
        content = "\n".join(body).strip()
        summary = " ".join(s.strip() for s in body if s.strip())[:MAX_SUMMARY_CHARS]
        return Unit(
            file_id=0,
            kind=UNIT_KIND_CHUNK,
            unit_type="doc_chunk",
            name=title,
            qualname=qualname,
            signature=collapse_ws(title),
            summary=summary,
            concepts="",
            start_line=start + 1,
            end_line=end,
            byte_start=offsets[start],
            byte_end=offsets[end],
        )
