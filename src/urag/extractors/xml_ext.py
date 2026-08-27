"""XML/XAML extractor: closes the markup blind spot for GUI projects.

Indexes .xaml/.axaml/.xml/.csproj/.props/.targets files as best-effort
structural units using the stdlib XML parser (no tree-sitter grammar for
XAML). Each element with an interesting attribute becomes a searchable
unit; symbol mentions inside markup (x:Class, DataType, {x:Static},
{StaticResource}, event handlers) become reference edges so agents can
answer "who references this converter/control" including the markup side.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from bisect import bisect_right

from ..models import Reference, Unit, UNIT_KIND_SYMBOL
from .base import Extractor, collapse_ws, dedupe_refs

MAX_CHUNK_LINES = 200

_EVENT_ATTRS = {
    "activated",
    "attachedtovisualtree",
    "checked",
    "click",
    "closing",
    "closed",
    "datacontextchanged",
    "deactivated",
    "detachedfromvisualtree",
    "doubletapped",
    "dragleave",
    "dragover",
    "drop",
    "gotfocus",
    "initialized",
    "keydown",
    "keyup",
    "layoutupdated",
    "loaded",
    "lostfocus",
    "opened",
    "pointerentered",
    "pointerexited",
    "pointermoved",
    "pointerpressed",
    "pointerreleased",
    "pointerwheelchanged",
    "selectionchanged",
    "sizechanged",
    "tapped",
    "textchanged",
    "unchecked",
    "unloaded",
}

_XSTATIC_RE = re.compile(r"\{x:Static\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\}")
_XTYPE_RE = re.compile(r"\{x:Type\s+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\}")
_RESOURCE_RE = re.compile(r"\{(?:StaticResource|DynamicResource)\s+([A-Za-z_][\w.]*)\}")

_TYPED_ATTRS = ("x:datatype",)

_XAML_NS = "{http://schemas.microsoft.com/winfx/2006/xaml}"


def _attr(attrs: dict, name: str) -> str | None:
    """Attribute lookup tolerant of namespace expansion (ElementTree turns
    `x:Class` into `{uri}Class` when xmlns is declared)."""
    if name in attrs:
        return attrs[name]
    local = name.split(":", 1)[-1]
    for key, value in attrs.items():
        if key.startswith("{") and key.rsplit("}", 1)[-1] == local:
            return value
    return None


def _attr_local(key: str) -> str:
    """Strip a namespace URI or prefix from an attribute name."""
    return key.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()


# Common markup controls that are framework types, not project symbols.
# Element tags outside this set are treated as project type references.
_CONTROL_TAGS = {
    "button",
    "border",
    "canvas",
    "checkbox",
    "combobox",
    "contentcontrol",
    "contextmenu",
    "datagrid",
    "datatemplate",
    "dockpanel",
    "dragdrop",
    "ellipse",
    "expander",
    "flyout",
    "grid",
    "image",
    "itemscontrol",
    "listbox",
    "menu",
    "menuitem",
    "panel",
    "path",
    "popup",
    "progressbar",
    "radiobutton",
    "rectangle",
    "resources",
    "resource",
    "scrollviewer",
    "separator",
    "setter",
    "slider",
    "stackpanel",
    "style",
    "tabcontrol",
    "tabitem",
    "template",
    "textblock",
    "textbox",
    "tooltip",
    "treeview",
    "treeviewitem",
    "usercontrol",
    "viewbox",
    "window",
    "wrapanel",
    "wrappanel",
}


class XmlExtractor(Extractor):
    language = "xml"

    def extract(self, source: str, rel_path: str) -> list[Unit]:
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode("utf-8")))
        units: list[Unit] = []
        try:
            root = ET.fromstring(source)
        except ET.ParseError:
            return self._chunk_units(lines, offsets, rel_path)
        units.append(
            self._make(
                lines,
                offsets,
                0,
                0,
                len(lines),
                f"file:{rel_path}",
                rel_path.split("/")[-1],
                "file",
                collapse_ws("".join(lines), 300),
            )
        )
        cursor = 0
        for el in root.iter():
            attrs = el.attrib
            if not attrs:
                continue
            tag = self._local(el.tag)
            xclass = _attr(attrs, "Class")
            if xclass:
                self._unit_at(
                    units, lines, offsets, source, cursor, xclass, tag, "class"
                )
                cursor = self._advance(source, cursor, xclass)
            key = _attr(attrs, "Key")
            if key and key != xclass:
                self._unit_at(
                    units, lines, offsets, source, cursor, key, tag, "resource"
                )
                cursor = self._advance(source, cursor, key)
            name = _attr(attrs, "Name")
            if name and name not in (xclass, key):
                self._unit_at(
                    units, lines, offsets, source, cursor, name, tag, "element"
                )
                cursor = self._advance(source, cursor, name)
            dtype = _attr(attrs, "DataType")
            if dtype:
                type_name = _XTYPE_RE.match(dtype.strip())
                if type_name:
                    dtype = type_name.group(1)
                self._unit_at(
                    units,
                    lines,
                    offsets,
                    source,
                    cursor,
                    dtype,
                    tag,
                    "template",
                )
                cursor = self._advance(source, cursor, dtype)
            for attr, value in attrs.items():
                if _attr_local(attr) in _EVENT_ATTRS and re.fullmatch(
                    r"[A-Za-z_][\w]*", value
                ):
                    self._unit_at(
                        units, lines, offsets, source, cursor, value, tag, "event"
                    )
                    cursor = self._advance(source, cursor, value)
        return units

    def collect_references(self, source: str) -> list:
        try:
            root = ET.fromstring(source)
        except ET.ParseError:
            return []
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode("utf-8")))
        out: list[Reference] = []
        cursor = 0
        for el in root.iter():
            attrs = el.attrib
            tag = self._local(el.tag)
            if (
                tag
                and tag[0].isupper()
                and tag.lower() not in _CONTROL_TAGS
                and re.fullmatch(r"[A-Za-z_][\w]*", tag)
            ):
                ref = self._ref(source, offsets, tag, "xaml_type", cursor)
                if ref is not None:
                    out.append(ref)
                cursor = self._advance(source, cursor, tag)
            xclass = _attr(attrs, "Class")
            if xclass:
                ref = self._ref(source, offsets, xclass, "xaml_type", cursor)
                if ref is not None:
                    out.append(ref)
                cursor = self._advance(source, cursor, xclass)
            dtype = _attr(attrs, "DataType")
            if dtype:
                ref = self._ref(source, offsets, dtype, "xaml_type", cursor)
                if ref is not None:
                    out.append(ref)
                cursor = self._advance(source, cursor, dtype)
            for attr, value in attrs.items():
                if _attr_local(attr) in _EVENT_ATTRS and re.fullmatch(
                    r"[A-Za-z_][\w]*", value
                ):
                    ref = self._ref(source, offsets, value, "xaml_event", cursor)
                    if ref is not None:
                        out.append(ref)
                    cursor = self._advance(source, cursor, value)
                owner = self._attached_owner(attr)
                if owner:
                    ref = self._ref(source, offsets, owner, "xaml_member", cursor)
                    if ref is not None:
                        out.append(ref)
                    cursor = self._advance(source, cursor, owner)
                for pattern, kind in (
                    (_XSTATIC_RE, "xaml_member"),
                    (_XTYPE_RE, "xaml_type"),
                    (_RESOURCE_RE, "xaml_resource"),
                ):
                    for m in pattern.finditer(value):
                        target = m.group(1)
                        ref = self._ref(
                            source,
                            offsets,
                            target,
                            kind,
                            source.find(value),
                        )
                        if ref is not None:
                            out.append(ref)
        return dedupe_refs(out)

    @staticmethod
    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]

    @staticmethod
    def _attached_owner(attr: str) -> str | None:
        """Owner type of an attached-property attribute
        (`behaviors:DragDropBehavior.EnableDragDrop` -> DragDropBehavior)."""
        text = attr.rsplit("}", 1)[-1].rsplit(":", 1)[-1]
        if "." not in text:
            return None
        owner = text.split(".", 1)[0]
        if not owner or not re.fullmatch(r"[A-Za-z_][\w]*", owner):
            return None
        if not owner[0].isupper() or owner.lower() in _CONTROL_TAGS:
            return None
        return owner

    @staticmethod
    def _advance(source: str, cursor: int, needle: str) -> int:
        idx = source.find(needle, cursor)
        return idx + len(needle) if idx >= 0 else cursor

    def _unit_at(
        self,
        units: list[Unit],
        lines: list[str],
        offsets: list[int],
        source: str,
        cursor: int,
        name: str,
        tag: str,
        unit_type: str,
    ) -> None:
        idx = source.find(name, cursor)
        if idx < 0:
            return
        line = self._line_of(offsets, idx)
        qualname = name.split(".")[-1]
        units.append(
            self._make(
                lines,
                offsets,
                line - 1,
                line - 1,
                line,
                qualname,
                qualname,
                unit_type,
                f"<{tag}> {unit_type} {name}",
            )
        )

    @staticmethod
    def _line_of(offsets: list[int], byte: int) -> int:
        return bisect_right(offsets, byte)

    def _ref(
        self, source: str, offsets: list[int], text: str, kind: str, cursor: int
    ) -> Reference | None:
        idx = source.find(text, cursor)
        if idx < 0:
            idx = source.find(text)
        if idx < 0:
            return None
        line = self._line_of(offsets, idx)
        base = text.split("<", 1)[0]
        names = re.findall(r"[A-Za-z_][\w]*", base)
        target = names[-1] if names else text
        return Reference(
            target=target,
            target_full=text,
            kind=kind,
            line=line,
            byte_start=idx,
            byte_end=idx + len(text),
        )

    def _chunk_units(
        self, lines: list[str], offsets: list[int], rel_path: str
    ) -> list[Unit]:
        out: list[Unit] = []
        for i in range(0, len(lines), MAX_CHUNK_LINES):
            end = min(i + MAX_CHUNK_LINES, len(lines))
            out.append(
                self._make(
                    lines,
                    offsets,
                    i,
                    i,
                    end,
                    f"file:{rel_path}#{i}",
                    rel_path.split("/")[-1],
                    "file",
                    collapse_ws("".join(lines[i:end]), 300),
                )
            )
        return out

    @staticmethod
    def _make(
        lines: list[str],
        offsets: list[int],
        start: int,
        start_byte: int,
        end: int,
        qualname: str,
        name: str,
        unit_type: str,
        summary: str,
    ) -> Unit:
        return Unit(
            file_id=0,
            kind=UNIT_KIND_SYMBOL,
            unit_type=unit_type,
            name=name,
            qualname=qualname,
            signature=collapse_ws(summary, 220),
            summary=summary[:220],
            concepts="",
            start_line=start + 1,
            end_line=end,
            byte_start=offsets[start_byte],
            byte_end=offsets[end] if end < len(offsets) else offsets[-1],
        )
