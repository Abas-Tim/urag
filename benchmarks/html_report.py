"""Render a urag eval JSON report as a self-contained, detailed HTML page.

Shows aggregate metrics, per-question retrieval samples from every system
(urag + opencode-style baselines), and auto-generated explanations of why
urag's retrieval is better — or where a baseline wins.

Usage:

    uv run python benchmarks/html_report.py benchmarks/reports/<name>.json
    uv run python benchmarks/html_report.py a.json b.json      # one HTML each
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_META = {
    "urag-auto": {
        "label": "urag adaptive",
        "kind": "urag",
        "desc": "Production urag retrieval with automatic query classification and graph routing.",
    },
    "urag-hybrid": {
        "label": "urag hybrid",
        "kind": "urag",
        "desc": "urag structure-aware retrieval: exact symbol + dense + lexical fusion over units, signatures and summaries.",
    },
    "urag-lexical": {
        "label": "urag lexical",
        "kind": "urag",
        "desc": "urag lexical-only retrieval (FTS/BM25 over the same structure-aware index).",
    },
    "urag-callers": {
        "label": "urag callers",
        "kind": "urag",
        "desc": "urag call-graph lookup: exact callers of a symbol from indexed call edges (alias-aware).",
    },
    "urag-transitive": {
        "label": "urag transitive",
        "kind": "urag",
        "desc": "urag multi-hop call-graph traversal: callers of callers, up to N hops, shortest-hop deduplicated.",
    },
    "rg": {
        "label": "grep (opencode Grep)",
        "kind": "opencode",
        "desc": "Emulates the opencode Grep tool: ripgrep with the query terms, ranked by matching-line counts; context = matching lines.",
    },
    "read": {
        "label": "read file (opencode Read)",
        "kind": "opencode",
        "desc": "Emulates the standard opencode agent loop: grep to locate candidate files, then Read the top files whole; context = entire file contents.",
    },
    "chunk": {
        "label": "naive chunk RAG",
        "kind": "baseline",
        "desc": "Structure-free RAG baseline: fixed 600-char chunks of every file, embedded and cosine-ranked.",
    },
    "oracle": {
        "label": "oracle",
        "kind": "oracle",
        "desc": "Upper bound: the exact gold evidence spans (definition/body of every gold unit).",
    },
}

CSS = """
:root {
  --urag: #2f81f7; --opencode: #e3b341; --baseline: #9e9e9e; --oracle: #8250df;
  --good: #1a7f37; --bad: #cf222e; --bg: #0d1117; --card: #161b22; --line: #30363d; --text: #e6edf3; --dim: #8b949e;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); }
a { color: var(--urag); }
header { padding: 24px 32px; border-bottom: 1px solid var(--line); background: linear-gradient(180deg, #12161d, var(--bg)); }
header h1 { margin: 0 0 4px; font-size: 22px; }
header .meta { color: var(--dim); font-size: 13px; }
section { padding: 20px 32px; max-width: 1280px; margin: 0 auto; }
h2 { font-size: 18px; border-bottom: 1px solid var(--line); padding-bottom: 6px; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--dim); font-weight: 600; }
tr.best td { background: rgba(26, 127, 55, .14); }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.kind-urag { background: rgba(47,129,247,.18); color: #79b8ff; }
.kind-opencode { background: rgba(227,179,65,.16); color: #e3b341; }
.kind-baseline { background: rgba(158,158,158,.15); color: #b9b9b9; }
.kind-oracle { background: rgba(130,80,223,.18); color: #bc8cff; }
.chip { display: inline-block; margin: 2px 4px 2px 0; padding: 1px 8px; border-radius: 4px; background: var(--card); border: 1px solid var(--line); font-size: 11px; }
.chip.good { border-color: rgba(26,127,55,.6); color: #3fb950; }
.chip.bad { border-color: rgba(207,34,46,.6); color: #f85149; }
.chip.dim { color: var(--dim); }
.bars { display: flex; gap: 2px; align-items: flex-end; height: 130px; margin: 8px 0 24px; }
.bars .col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; min-width: 0; }
.bars .bar { width: 70%; background: var(--dim); border-radius: 3px 3px 0 0; min-height: 2px; }
.bars .bar.urag { background: var(--urag); }
.bars .bar.opencode { background: var(--opencode); }
.bars .bar.baseline { background: #6e7681; }
.bars .bar.oracle { background: var(--oracle); }
.bars .val { font-size: 10px; color: var(--dim); margin-bottom: 2px; }
.bars .name { font-size: 10px; color: var(--dim); margin-top: 4px; text-align: center; word-break: break-word; }
.scoreboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.card .big { font-size: 26px; font-weight: 700; }
.card .lbl { color: var(--dim); font-size: 12px; }
article.q { border: 1px solid var(--line); border-radius: 8px; margin: 16px 0; overflow: hidden; }
article.q > header { padding: 10px 16px; border-bottom: 1px solid var(--line); }
article.q h3 { margin: 0; font-size: 15px; }
article.q .gold { color: var(--dim); font-size: 12px; margin-top: 4px; }
.sysgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; padding: 12px 16px; }
.sys { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: var(--bg); }
.sys h4 { margin: 0 0 4px; font-size: 13px; display: flex; justify-content: space-between; gap: 6px; }
.metrics { font-size: 11px; color: var(--dim); margin-bottom: 6px; }
.hit { border-top: 1px dashed var(--line); padding: 6px 0; font-size: 12px; }
.hit .rank { display: inline-block; width: 16px; height: 16px; line-height: 16px; text-align: center; border-radius: 3px; background: #21262d; font-size: 10px; margin-right: 6px; }
.hit.gold { background: rgba(26,127,55,.10); border-left: 3px solid var(--good); padding-left: 6px; }
.hit .title { font-weight: 600; }
.hit .file { color: var(--dim); }
.hit pre { margin: 4px 0 0; padding: 6px; background: #0a0e14; border-radius: 4px; font-size: 11px; overflow-x: auto; max-height: 140px; color: #9fb3c8; white-space: pre-wrap; }
.story { padding: 10px 16px; border-top: 1px solid var(--line); font-size: 13px; background: rgba(47,129,247,.06); }
.story.loss { background: rgba(227,179,65,.06); }
.story strong { color: #79b8ff; }
.story.loss strong { color: #e3b341; }
.note { color: var(--dim); font-size: 12px; }
details.toc summary { cursor: pointer; color: var(--urag); }
.hidden { display: none; }
footer { padding: 20px 32px; color: var(--dim); font-size: 12px; border-top: 1px solid var(--line); }
"""

FILTER_JS = """
<script>
function filterQs() {
  const q = document.getElementById("qfilter").value.trim().toLowerCase();
  const labels = Array.from(document.querySelectorAll("input[name=lab]:checked")).map(e => e.value);
  const sys = document.getElementById("sysfilter").value;
  document.querySelectorAll("article.q").forEach(a => {
    const text = a.textContent.toLowerCase();
    const showQ = !q || text.includes(q);
    const showL = labels.length === 0 || labels.includes(a.dataset.label);
    const showS = sys === "all" || a.dataset.systems.split(",").includes(sys);
    a.classList.toggle("hidden", !(showQ && showL && showS));
  });
}
function showOnlySys(name) {
  document.getElementById("sysfilter").value = name;
  filterQs();
}
</script>
"""


def _esc(text: str) -> str:
    return html_lib.escape(str(text), quote=True)


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.0f}%"


def _num(v: float | None, nd: int = 0) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def system_meta(name: str) -> dict:
    return SYSTEM_META.get(
        name,
        {"label": name, "kind": "baseline", "desc": ""},
    )


def explain(
    question: dict, metrics: dict[str, dict], systems: list[str]
) -> tuple[str, bool]:
    """Auto-generated, data-driven explanation of the urag vs opencode outcome."""
    gold_units = question.get("gold_unit_ids") or []
    gold_files = question.get("gold_files") or (
        [question["gold_file"]] if question.get("gold_file") else []
    )
    urag_systems = [s for s in systems if system_meta(s)["kind"] == "urag"]
    codec_systems = [s for s in systems if system_meta(s)["kind"] == "opencode"]

    def best(named: list[str], key: str) -> tuple[str, dict] | None:
        cands = [(s, metrics[s]) for s in named if s in metrics and metrics[s]]
        if not cands:
            return None
        return max(
            cands,
            key=lambda kv: (
                kv[1].get(key, 0),
                kv[1].get("mrr", 0),
                -kv[1].get("tokens", 0),
            ),
        )

    urag = best(urag_systems, "unit_recall")
    codec = best(codec_systems, "unit_recall")
    if not urag:
        return "No urag system ran for this question.", False
    _, um = urag
    codec_label = "the opencode tools" if not codec else system_meta(codec[0])["label"]
    cm = codec[1] if codec else {"unit_recall": 0, "tokens": 0, "mrr": 0}

    parts: list[str] = []
    win = um.get("unit_recall", 0) > cm.get("unit_recall", 0) + 1e-9
    loss = um.get("unit_recall", 0) < cm.get("unit_recall", 0) - 1e-9
    tie = not win and not loss

    found_u = um.get("unit_recall", 0) * len(gold_units) if gold_units else 0
    found_c = cm.get("unit_recall", 0) * len(gold_units) if gold_units else 0

    if win:
        parts.append(
            f"<strong>{system_meta(urag[0])['label']}</strong> retrieved "
            f"{found_u:.0f}/{len(gold_units)} gold units vs {found_c:.0f}/{len(gold_units)} for "
            f"{codec_label}."
        )
        if um.get("mrr", 0) == 1.0 and cm.get("mrr", 0) < 1.0:
            parts.append(
                "urag ranked a gold unit at position 1 — exact symbol/structure match — while the "
                f"{codec_label} results only surfaced it later (MRR {cm.get('mrr', 0):.2f})."
            )
    elif loss:
        parts.append(
            f"{codec_label} beat urag here ({found_c:.0f}/{len(gold_units)} vs {found_u:.0f}/{len(gold_units)} "
            f"gold units)."
        )
        if question.get("label") == "definition" and gold_files:
            parts.append(
                "Definition questions with a single gold unit spread across a file favor whole-file "
                f"read: the file is the only place the answer can be."
            )
    else:
        parts.append(
            f"Tie on recall ({found_u:.0f}/{len(gold_units)} gold units for both "
            f"{system_meta(urag[0])['label']} and {codec_label})."
        )

    if "read" in metrics and metrics["read"]:
        read_tokens = metrics["read"].get("tokens", 0)
        u_tok = um.get("tokens", 0)
        if read_tokens > 0 and u_tok > 0 and read_tokens > u_tok:
            savings = (1 - u_tok / read_tokens) * 100
            parts.append(
                f"Token efficiency: urag needed <strong>{u_tok:.0f}</strong> tokens vs "
                f"<strong>{read_tokens:.0f}</strong> for read-file ({savings:.0f}% less) — "
                "the Read tool drags in whole files while urag returns the exact unit spans."
            )
        elif read_tokens > 0 and u_tok > 0:
            parts.append(
                f"Token parity: urag {u_tok:.0f} vs read-file {read_tokens:.0f} tokens."
            )
    if "rg" in metrics and metrics["rg"]:
        rg_tok = metrics["rg"].get("tokens", 0)
        if rg_tok > 0 and um.get("tokens", 0) < rg_tok:
            parts.append(
                f"urag ({um.get('tokens', 0):.0f} tokens) also undercut grep's matching-line "
                f"context ({rg_tok:.0f} tokens)."
            )

    if not parts:
        parts.append(
            f"{system_meta(urag[0])['label']} and {codec_label} performed equivalently."
        )
    return " ".join(parts), not (win or tie) and loss


def render_report(report: dict, title: str) -> str:
    systems = list(report.get("systems", {}).keys())
    metrics = report.get("per_query", {})
    questions = report.get("questions", [])
    top_k = report.get("top_k", 5)
    hits = report.get("hits", {})
    q_systems: list[str] = []
    for i, _q in enumerate(questions):
        present = [
            s
            for s in systems
            if i < len(metrics.get(s, [])) and isinstance(metrics.get(s, [])[i], dict)
        ]
        q_systems.append(",".join(present))

    urag_names = [s for s in systems if system_meta(s)["kind"] == "urag"]
    codec_names = [s for s in systems if system_meta(s)["kind"] == "opencode"]

    def per_q(i: int) -> dict[str, dict]:
        out = {}
        for s in systems:
            lst = metrics.get(s, [])
            row = lst[i] if i < len(lst) else None
            out[s] = row if isinstance(row, dict) else {}
        return out

    rows = []
    for name in systems:
        a = report["systems"].get(name, {})
        rows.append((name, a))

    def best_cell(key: str, better_higher: bool = True) -> set[str]:
        vals = [
            (s, a.get(key)) for s, a in rows if isinstance(a.get(key), (int, float))
        ]
        if not vals:
            return set()
        if better_higher:
            bv = max(v for _, v in vals)
        else:
            bv = min(v for _, v in vals)
        return {s for s, v in vals if v == bv}

    best = {k: best_cell(k) for k in ("unit_recall", "mrr", "precision")}
    best["mean_tokens"] = best_cell("mean_tokens", better_higher=False)
    best["p50_sec"] = best_cell("p50_sec", better_higher=False)

    out: list[str] = []
    out.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    out.append(f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>")

    meta = [
        ("urag version", report.get("urag_version", "?")),
        ("schema", report.get("schema_version", "?")),
        ("questions", str(len(questions))),
        ("top_k", str(top_k)),
    ]
    if report.get("index_seconds") is not None:
        meta.append(("fresh index build", f"{report['index_seconds']:.1f}s"))
    if report.get("chunk_load_seconds") is not None:
        meta.append(("chunk baseline load", f"{report['chunk_load_seconds']:.1f}s"))
    out.append(
        "<header><h1>urag benchmark report</h1>"
        f"<div class='meta'>{_esc(title)} &nbsp;·&nbsp; "
        + " &nbsp;·&nbsp; ".join(f"{k}: <b>{_esc(v)}</b>" for k, v in meta)
        + "</div></header>"
    )

    out.append("<section><h2>How to read this</h2><p class='note'>")
    out.append(
        "Each system answers the same questions and the context it retrieved is scored against "
        "provable gold (the units that actually define/call the queried symbol). The comparison of "
        "interest: <b>urag</b> (structure-aware index + call graph) vs the tools an opencode agent "
        "normally uses to gather context — <b>Grep</b> (matching lines) and <b>Read</b> "
        "(whole files located via grep). urag's thesis: the same or better recall at a fraction of "
        "the tokens, with exact symbol ranking instead of textual luck."
    )
    out.append("</p></section>")

    out.append(
        "<section><h2>Systems under test</h2><table><tr><th>system</th><th>what it returns</th></tr>"
    )
    for name in systems:
        m = system_meta(name)
        out.append(
            f"<tr><td><span class='badge kind-{m['kind']}'>{_esc(m['label'])}</span></td>"
            f"<td class='note'>{_esc(m['desc'])}</td></tr>"
        )
    out.append("</table></section>")

    out.append("<section><h2>Aggregate results</h2>")
    out.append(
        "<table><tr><th>system</th><th>unit recall</th><th>file recall</th><th>precision</th>"
        "<th>indirect recall</th><th>MRR</th><th>mean tokens</th><th>p50</th><th>p95</th></tr>"
    )
    for name, a in rows:
        if not a:
            out.append(
                f"<tr><td><span class='badge kind-baseline'>{_esc(name)}</span></td><td colspan=8 class='note'>no results</td></tr>"
            )
            continue
        m = system_meta(name)
        cls = ""
        if name in best["unit_recall"]:
            cls = " class='best'"
        out.append(
            f"<tr{cls}><td><span class='badge kind-{m['kind']}'>{_esc(m['label'])}</span></td>"
            f"<td>{_pct(a.get('unit_recall'))}</td>"
            f"<td>{_pct(a.get('file_recall'))}</td>"
            f"<td>{_pct(a.get('precision'))}</td>"
            f"<td>{_pct(a.get('indirect_recall'))}</td>"
            f"<td>{_num(a.get('mrr'), 2)}</td>"
            f"<td>{_num(a.get('mean_tokens'), 0)}</td>"
            f"<td>{_num(a.get('p50_sec', 0) * 1000, 0)} ms</td>"
            f"<td>{_num(a.get('p95_sec', 0) * 1000, 0)} ms</td></tr>"
        )
    out.append("</table>")

    for chart, key, unit in (
        ("Unit recall", "unit_recall", "%"),
        ("Precision", "precision", "%"),
        ("MRR", "mrr", ""),
    ):
        vals = {s: a.get(key, 0) for s, a in rows if a}
        maxv = max(vals.values()) if vals else 1
        out.append(f"<h2 style='font-size:14px'>{chart}</h2><div class='bars'>")
        for name, v in vals.items():
            m = system_meta(name)
            out.append(
                f"<div class='col'><span class='val'>{_pct(v) if unit == '%' else _num(v, 2)}</span>"
                f"<div class='bar {m['kind']}' style='height: {max(2, v / maxv * 100):.1f}%'></div>"
                f"<span class='name'>{_esc(m['label'])}</span></div>"
            )
        out.append("</div>")

    max_tok = max((a.get("mean_tokens", 0) for _, a in rows if a), default=1)
    out.append(
        "<h2 style='font-size:14px'>Mean tokens per retrieval (lower is better)</h2><div class='bars'>"
    )
    for name, a in rows:
        if not a:
            continue
        m = system_meta(name)
        v = a.get("mean_tokens", 0)
        out.append(
            f"<div class='col'><span class='val'>{v:.0f}</span>"
            f"<div class='bar {m['kind']}' style='height: {max(2, v / max_tok * 100):.1f}%'></div>"
            f"<span class='name'>{_esc(m['label'])}</span></div>"
        )
    out.append("</div></section>")

    wins = 0
    losses = 0
    ties = 0
    token_savings: list[float] = []
    for i, q in enumerate(questions):
        per = per_q(i)
        _, loss = explain(q, per, systems)
        urag_recall = max(
            (per[s].get("unit_recall", 0) for s in urag_names if per.get(s)), default=0
        )
        codec_recall = max(
            (per[s].get("unit_recall", 0) for s in codec_names if per.get(s)), default=0
        )
        if urag_recall > codec_recall + 1e-9:
            wins += 1
        elif urag_recall < codec_recall - 1e-9:
            losses += 1
        else:
            ties += 1
        if "read" in per and per["read"].get("tokens"):
            u_tok = max(
                (per[s].get("tokens", 0) for s in urag_names if per.get(s)), default=0
            )
            if u_tok > 0:
                token_savings.append(1 - u_tok / per["read"]["tokens"])

    median_savings = (
        sorted(token_savings)[len(token_savings) // 2] if token_savings else 0.0
    )
    out.append("<section><h2>Scoreboard vs opencode tools</h2><div class='scoreboard'>")
    cards = [
        (
            f"{wins}",
            f"questions where urag beat grep/read on unit recall (of {len(questions)})",
        ),
        (f"{ties}", "questions tied on recall"),
        (f"{losses}", "questions where opencode grep/read beat urag"),
        (
            f"{median_savings * 100:.0f}%",
            "median token saving of urag vs read-file (per question)",
        ),
    ]
    for big, lbl in cards:
        out.append(
            f"<div class='card'><div class='big'>{big}</div><div class='lbl'>{lbl}</div></div>"
        )
    out.append("</div></section>")

    out.append(
        "<section><h2>Per-question retrieval samples</h2>"
        "<div class='note' style='margin-bottom:8px'>"
        "<input id='qfilter' placeholder='filter by text…' oninput='filterQs()' style='background:var(--card);color:var(--text);border:1px solid var(--line);padding:4px 8px;border-radius:4px'> "
        "<select id='sysfilter' onchange='filterQs()' style='background:var(--card);color:var(--text);border:1px solid var(--line);padding:4px 8px;border-radius:4px'>"
        "<option value='all'>all systems</option>"
        + "".join(f"<option value='{_esc(s)}'>{_esc(s)}</option>" for s in systems)
        + "</select> labels: "
        + " ".join(
            f"<label class='chip dim'><input type='checkbox' name='lab' value='{_esc(q.get('label', ''))}' onchange='filterQs()' checked> {_esc(q.get('label', ''))}</label>"
            for q in {q.get("label", ""): q for q in questions}.values()
        )
        + f"</div>{FILTER_JS}"
    )

    ordered = sorted(
        range(len(questions)),
        key=lambda i: (
            max(
                (
                    per_q(i)[s].get("unit_recall", 0)
                    for s in urag_names
                    if per_q(i).get(s)
                ),
                default=0,
            )
            - max(
                (
                    per_q(i)[s].get("unit_recall", 0)
                    for s in codec_names
                    if per_q(i).get(s)
                ),
                default=0,
            ),
            max(
                (per_q(i)[s].get("mrr", 0) for s in urag_names if per_q(i).get(s)),
                default=0,
            ),
        ),
        reverse=True,
    )

    for i in ordered:
        q = questions[i]
        gold_units = q.get("gold_unit_ids") or []
        gold_files = q.get("gold_files") or (
            [q["gold_file"]] if q.get("gold_file") else []
        )
        per = per_q(i)
        story, loss = explain(q, per, systems)
        out.append(
            f"<article class='q' data-label='{_esc(q.get('label', ''))}' data-systems='{_esc(q_systems[i])}'>"
            f"<header><h3>Q{i + 1} · “{_esc(q['query'])}”</h3>"
            f"<div class='gold'><span class='chip dim'>label: {_esc(q.get('label', ''))}</span>"
        )
        if q.get("target"):
            out.append(f"<span class='chip dim'>target: {_esc(q['target'])}</span>")
        if q.get("depth", 1) > 1:
            out.append(f"<span class='chip dim'>depth: {q['depth']}</span>")
        out.append(
            f"<span class='chip dim'>gold units: {len(gold_units)}</span>"
            f"<span class='chip dim'>gold files: {_esc(', '.join(gold_files))}</span>"
            "</div></header><div class='sysgrid'>"
        )
        for name in systems:
            m = system_meta(name)
            row = per.get(name, {})
            out.append(
                f"<div class='sys'><h4><span><span class='badge kind-{m['kind']}'>{_esc(m['label'])}</span></span>"
                f"<a href='#' onclick='showOnlySys(\"{_esc(name)}\")' style='font-size:10px'>only</a></h4>"
            )
            if not row:
                out.append("<div class='note'>no results</div></div>")
                continue
            chips = [
                (
                    "recall",
                    _pct(row.get("unit_recall")),
                    "good" if row.get("unit_recall") else "dim",
                ),
                ("precision", _pct(row.get("precision")), "dim"),
                ("mrr", _num(row.get("mrr"), 2), "dim"),
                ("tokens", f"{_num(row.get('tokens'), 0)} tok", "dim"),
                ("time", f"{_num(row.get('seconds', 0) * 1000, 1)} ms", "dim"),
            ]
            out.append(
                "<div class='metrics'>"
                + "".join(
                    f"<span class='chip {cls}'>{k}: {v}</span>" for k, v, cls in chips
                )
                + "</div>"
            )
            qhits = hits.get(name, [])
            if i < len(qhits):
                for rank, h in enumerate(qhits[i][: top_k * 2], 1):
                    gold = (
                        h.get("unit_id") in gold_units
                        or bool(set(h.get("unit_ids", [])) & set(gold_units))
                        or h.get("file") in gold_files
                    )
                    title = h.get("title") or ""
                    detail = h.get("detail") or ""
                    out.append(
                        f"<div class='hit{' gold' if gold else ''}'><span class='rank'>{rank}</span>"
                        f"<span class='title'>{_esc(title)}</span> "
                        f"<span class='file'>{_esc(h.get('file', ''))} · {_esc(h.get('tokens', 0))} tok</span>"
                        + (" <span class='chip good'>gold</span>" if gold else "")
                        + (f"<pre>{_esc(detail)}</pre>" if detail else "")
                        + "</div>"
                    )
            out.append("</div>")
        out.append(
            f"</div><div class='story{' loss' if loss else ''}'>{story}</div></article>"
        )

    out.append("</section>")
    out.append(
        "<footer>Generated by <code>benchmarks/html_report.py</code> from a urag eval JSON report. "
        "Tokens are approximations (chars/4). Times exclude fresh-index construction unless noted. "
        "Grep/read baselines emulate the opencode agent tools, not the opencode runtime itself.</footer>"
    )
    out.append("</body></html>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+", type=Path, help="urag eval JSON report(s)")
    ap.add_argument("--out", type=Path, help="output path (single report only)")
    args = ap.parse_args()

    if args.out and len(args.reports) > 1:
        ap.error("--out requires exactly one report")
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        html = render_report(report, title=path.name or "urag eval report")
        out = args.out or path.with_suffix(".html")
        out.write_text(html, encoding="utf-8")
        print(f"wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
