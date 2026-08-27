"""Adaptive query classification: rule-based, zero-cost routing to budget tiers.

Every query is classified before retrieval so that context budgets are a
per-query decision, not a fixed constant. Tiers follow current RAG research:
retrieving more evidence is not monotonically beneficial.

Tiers:
  symbol    — exact symbol/identifier lookup: lexical-only, tiny budget
  local     — single-module implementation question
  debugging — cross-module behavior, error hunting
  impact    — architecture / "what breaks if X" / callers & dependencies
"""

from __future__ import annotations

import re

CLASSES = ("symbol", "local", "debugging", "impact")

BUDGETS: dict[str, dict] = {
    "symbol": {"tokens": 800, "top_k": 3},
    "local": {"tokens": 2000, "top_k": 5},
    "debugging": {"tokens": 4000, "top_k": 8},
    "impact": {"tokens": 6000, "top_k": 10},
}

MODE_BY_CLASS = {
    "symbol": "lexical",
    "local": "hybrid",
    "debugging": "hybrid",
    "impact": "hybrid",
}

# Deterministic signals, checked in order. The router itself must stay cheap
# (no model call) — per adaptive-RAG research, routing overhead can otherwise
# offset retrieval savings.

_DOTTED = re.compile(r"^[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)+$")
_SNAKE = re.compile(r"^[a-zA-Z_]\w*_[\w]+$")
_CAMEL = re.compile(r"^[A-Z][a-zA-Z0-9]*([a-z][A-Z]|[A-Z][a-z])[a-zA-Z0-9]*$")
_UPPER = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")
_FILEPATH = re.compile(
    r"(^|[\s(])([a-zA-Z0-9_\-./]+\.(py|ts|js|tsx|jsx|go|rs|java|c|cc|cpp|h|hpp|md))\b"
)
_IMPACT_WORDS = (
    "what calls",
    "who calls",
    "who uses",
    "what uses",
    "callers",
    "callees",
    "dependencies",
    "depends on",
    "imports",
    "what breaks",
    "impact",
    "blast radius",
    "downstream",
    "referenced by",
    "invoked by",
    "used by",
    "references",
    "reference",
    "who mentions",
    "is dead",
    "dead code",
    "unused",
    "unreferenced",
)
_DEBUG_WORDS = (
    "debug",
    "error",
    "crash",
    "fails",
    "failure",
    "broken",
    "exception",
    "traceback",
    "segfault",
    "hang",
    "why is",
    "why does",
    "not working",
    "cross-module",
    "across",
    "between",
    "flow from",
)
_CONCEPT_WORDS = (
    "architecture",
    "design",
    "how does",
    "how is",
    "explain",
    "overview",
    "understand",
    "works",
    "mechanism",
    "lifecycle",
    "module",
    "system",
)


def classify(query: str) -> str:
    q = query.strip()
    lowered = q.lower()
    if len(q) < 2:
        return "local"
    if (
        _DOTTED.match(q)
        or _SNAKE.match(q)
        or _CAMEL.match(q)
        or _UPPER.match(q.split()[0] if q.split() else q)
    ):
        return "symbol"
    if _FILEPATH.search(q):
        return "symbol"
    if any(w in lowered for w in _IMPACT_WORDS):
        return "impact"
    if any(w in lowered for w in _DEBUG_WORDS):
        return "debugging"
    if any(w in lowered for w in _CONCEPT_WORDS):
        return "local"
    if len(q.split()) <= 3:
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", q)
        non_plain = any(
            _DOTTED.match(t) or _SNAKE.match(t) or _CAMEL.match(t) or _UPPER.match(t)
            for t in toks
        )
        return "symbol" if non_plain else "local"
    return "local"


def top_k_for(qclass: str) -> int:
    return BUDGETS.get(qclass, BUDGETS["local"])["top_k"]
