"""Core data models for urag.

Design follows the "retrieval keys vs evidence payloads" principle:
units store compact, discriminative L0/L1 records; L2 evidence (exact
source spans) is loaded lazily from the filesystem on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UNIT_KIND_SYMBOL = "symbol"
UNIT_KIND_CHUNK = "chunk"

SYMBOL_TYPES = {
    "function",
    "class",
    "method",
    "import",
    "interface",
    "type_alias",
    "variable",
    "enum",
    "file",
}

CHUNK_TYPES = {"doc_chunk", "doc_file"}


@dataclass
class Unit:
    """A single searchable record (retrieval key + payload pointer)."""

    file_id: int
    kind: str
    unit_type: str
    name: str
    qualname: str = ""
    signature: str = ""
    summary: str = ""
    concepts: str = ""
    relationships: str = ""
    start_line: int = 0
    end_line: int = 0
    start_col: int = 0
    end_col: int = 0
    byte_start: int = 0
    byte_end: int = 0
    parent_id: int | None = None
    id: int | None = None

    @property
    def retrieval_key(self) -> str:
        """The compact text that gets embedded and lexically indexed."""
        parts = [
            self.qualname or self.name,
            self.signature,
            self.summary,
            self.concepts,
            self.relationships,
        ]
        return "\n".join(p for p in parts if p)

    @property
    def is_symbol(self) -> bool:
        return self.kind == UNIT_KIND_SYMBOL

    @property
    def is_chunk(self) -> bool:
        return self.kind == UNIT_KIND_CHUNK


@dataclass
class SourceFile:
    """A tracked file in the index."""

    id: int | None = None
    path: str = ""
    kind: str = "source"  # source | doc | config
    language: str = ""
    size: int = 0
    mtime: float = 0.0
    sha256: str = ""
    commit: str = ""
    indexed_at: str = ""

    @property
    def is_stale(self) -> bool:
        return self.size == 0


@dataclass
class RetrievedUnit:
    """A unit returned from retrieval, with fusion score and evidence."""

    unit: Unit
    file_path: str
    score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None
    commit: str = ""
    stale: bool = False
    caller_of: str = ""
    call_line: int = 0
    hop: int = 0
    resolved_target: str = ""

    def to_dict(self, include_evidence: bool = False) -> dict:
        u = self.unit
        d = {
            "id": u.id,
            "kind": u.kind,
            "type": u.unit_type,
            "name": u.name,
            "qualname": u.qualname,
            "signature": u.signature,
            "summary": u.summary,
            "concepts": u.concepts,
            "relationships": u.relationships,
            "file": self.file_path,
            "lines": [u.start_line, u.end_line],
            "score": round(self.score, 4),
            "commit": self.commit,
            "stale": self.stale,
            "hop": self.hop,
        }
        if self.caller_of:
            d["calls"] = self.caller_of
            d["call_line"] = self.call_line
        if self.resolved_target:
            d["resolved_to"] = self.resolved_target
        if include_evidence:
            d["evidence"] = self.evidence
        return d

    @property
    def evidence(self) -> str | None:
        return None


@dataclass
class CallSite:
    """A function call inside source: callee (last segment) + full chain."""

    callee: str
    callee_full: str
    line: int
    byte_start: int
    byte_end: int


@dataclass
class IndexStats:
    files: int = 0
    units: int = 0
    embedded: int = 0
    size_bytes: int = 0
    by_language: dict[str, int] = field(default_factory=dict)
    last_indexed: str = ""
