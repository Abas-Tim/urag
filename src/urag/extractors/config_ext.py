"""Configuration extractor: JSON, YAML, TOML, INI, and env/dotenv files.

Config keys are indexed as `config_key` units with dotted qualnames
(`server.host`, `[database].port`, `LOG_LEVEL`) so agents can answer "where is
this setting / what env vars exist" without reading whole files. Extraction is
line-based and best-effort: spans point at the key's line, values are stored
as trimmed summaries.
"""

from __future__ import annotations

import json
import re

from ..models import Unit, UNIT_KIND_SYMBOL
from .base import Extractor, collapse_ws

MAX_VALUE_CHARS = 200

_JSON_KEY = re.compile(r'^"((?:[^"\\]|\\.)*)"\s*:')
_YAML_KEY = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.\-]*)\s*:")
_TOML_KEY = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.\-]*)\s*=")
_TOML_TABLE = re.compile(r"^\[\[?([A-Za-z0-9_.\-]+)\]?\]")
_ENV_KEY = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
_INI_SECTION = re.compile(r"^\[([^\]]+)\]")

_COMMENT_ONLY = ("#", ";", "//")


def _byte_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")) + 1)
    return offsets


def _scalar_str(v) -> str:
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


class ConfigExtractor(Extractor):
    language = "config"

    def __init__(self, fmt: str):
        self.fmt = fmt

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        if self.fmt == "json":
            return self._json(source, rel_path)
        if self.fmt == "yaml":
            return self._yaml(source, rel_path)
        if self.fmt == "toml":
            return self._toml(source, rel_path)
        if self.fmt == "ini":
            return self._ini(source, rel_path)
        return self._env(source, rel_path)

    def _make(
        self,
        offsets: list[int],
        line_no: int,
        line: str,
        qualname: str,
        value: str,
    ) -> Unit:
        name = qualname.split(".")[-1]
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type="config_key",
            name=name,
            qualname=qualname,
            signature=collapse_ws(line, 220),
            summary=value[:MAX_VALUE_CHARS],
            concepts="",
            start_line=line_no,
            end_line=line_no,
            byte_start=offsets[line_no - 1],
            byte_end=offsets[line_no],
        )

    def _strip_comment(self, line: str) -> str:
        for marker in _COMMENT_ONLY:
            if marker in line:
                line = line.split(marker, 1)[0]
        return line.rstrip()

    def _json(self, source: str, rel_path: str) -> list[Unit]:
        try:
            data = json.loads(source)
        except (json.JSONDecodeError, ValueError):
            return self._json_linebased(source, rel_path)
        pairs: list[tuple[str, str]] = []

        def walk(obj, prefix: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    path = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        walk(v, path)
                    else:
                        pairs.append((path, _scalar_str(v)))
            elif isinstance(obj, list):
                pairs.append((prefix, _scalar_str(obj)))

        walk(data, "")
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        key_lines: dict[str, list[int]] = {}
        for line_no, raw in enumerate(lines, 1):
            for m in re.finditer(r'"((?:[^"\\]|\\.)*)"\s*:', raw):
                key_lines.setdefault(m.group(1), []).append(line_no)
        seen: dict[str, int] = {}
        units: list[Unit] = []
        for path, value in pairs:
            key = re.sub(r"\[\d+\]$", "", path.rsplit(".", 1)[-1])
            occurrences = key_lines.get(key, [])
            used = seen.get(key, 0)
            line_no = (
                occurrences[used]
                if used < len(occurrences)
                else (occurrences[-1] if occurrences else 1)
            )
            seen[key] = used + 1
            raw_line = lines[line_no - 1] if line_no <= len(lines) else ""
            units.append(self._make(offsets, line_no, raw_line, path, value))
        return units

    def _json_linebased(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        stack: list[tuple[int, str]] = []
        for line_no, raw in enumerate(lines, 1):
            line = self._strip_comment(raw)
            stripped = line.strip()
            depth = len(line) - len(line.lstrip())
            while stack and depth <= stack[-1][0]:
                stack.pop()
            m = _JSON_KEY.match(stripped)
            if m:
                key = m.group(1)
                rest = stripped[m.end() :]
                qualname = ".".join([p for _, p in stack] + [key])
                stack.append((depth, key))
                if rest.strip().rstrip(",").endswith(("{", "[")):
                    continue
                value = self._json_value(rest)
                units.append(self._make(offsets, line_no, raw, qualname, value))
        return units

    @staticmethod
    def _json_value(rest: str) -> str:
        rest = rest.strip().rstrip(",")
        return collapse_ws(rest, MAX_VALUE_CHARS)

    def _yaml(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        stack: list[tuple[int, str]] = []
        for line_no, raw in enumerate(lines, 1):
            if not raw.strip() or raw.lstrip().startswith(_COMMENT_ONLY):
                continue
            stripped = raw.strip()
            if stripped.startswith("- "):
                continue
            depth = len(raw) - len(raw.lstrip(" "))
            while stack and depth <= stack[-1][0]:
                stack.pop()
            m = _YAML_KEY.match(stripped)
            if m:
                key = m.group(1)
                qualname = ".".join([p for _, p in stack] + [key])
                value = collapse_ws(stripped[m.end() :], MAX_VALUE_CHARS)
                units.append(self._make(offsets, line_no, raw, qualname, value))
                stack.append((depth, key))
        return units

    def _toml(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        table: list[str] = []
        for line_no, raw in enumerate(lines, 1):
            line = self._strip_comment(raw)
            stripped = line.strip()
            if not stripped:
                continue
            tm = _TOML_TABLE.match(stripped)
            if tm:
                table = [p for p in tm.group(1).split(".") if p]
                continue
            m = _TOML_KEY.match(stripped)
            if m:
                key = m.group(1)
                qualname = ".".join(table + [key])
                value = collapse_ws(stripped[m.end() :], MAX_VALUE_CHARS)
                units.append(self._make(offsets, line_no, raw, qualname, value))
        return units

    def _ini(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        section: list[str] = []
        for line_no, raw in enumerate(lines, 1):
            line = self._strip_comment(raw)
            stripped = line.strip()
            if not stripped:
                continue
            sm = _INI_SECTION.match(stripped)
            if sm:
                section = [p for p in sm.group(1).split(".") if p]
                continue
            if "=" in stripped:
                key, _, value = stripped.partition("=")
            elif ":" in stripped:
                key, _, value = stripped.partition(":")
            else:
                continue
            key = key.strip()
            if not key or " " in key:
                continue
            qualname = ".".join(section + [key])
            value = collapse_ws(value, MAX_VALUE_CHARS)
            units.append(self._make(offsets, line_no, raw, qualname, value))
        return units

    def _env(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines()
        offsets = _byte_offsets(lines)
        units: list[Unit] = []
        for line_no, raw in enumerate(lines, 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith(_COMMENT_ONLY):
                continue
            m = _ENV_KEY.match(stripped)
            if not m:
                continue
            key = m.group(1)
            value = collapse_ws(stripped[m.end() :], MAX_VALUE_CHARS)
            units.append(self._make(offsets, line_no, raw, key, value))
        return units
