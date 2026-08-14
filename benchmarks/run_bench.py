"""Benchmark runner for the call-graph suites (transitive + import-alias).

Usage (from repo root):

    uv run python benchmarks/run_bench.py                      # fixture project only
    uv run python benchmarks/run_bench.py --self --yes         # fixture + urag's own repo
    uv run python benchmarks/run_bench.py --transitive 30 --alias 30 --top-k 5
    uv run python benchmarks/run_bench.py --reuse-questions old.json   # same questions, new code
    uv run python benchmarks/run_bench.py --compare before.json after.json

Reports land in benchmarks/reports/<rev>-<timestamp>.json plus a detailed
self-contained <name>.html rendering. The target root's .urag/ index is wiped
and rebuilt fresh on every run (pass --yes to confirm non-fixture roots) so
indexer schema changes never leak stale data into results.

For before/after comparisons: run the "before" command on the base branch,
then on the feature branch run --reuse-questions <before-report> (gold is
re-derived against the new index via --reresolve), or diff two finished
reports with --compare.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "callgraph_fixture"
REPORTS = ROOT / "benchmarks" / "reports"


def sh(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def urag_sh(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return sh([sys.executable, "-m", "urag", *args], cwd)


def git_head(cwd: Path) -> str:
    b = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    r = sh(["git", "rev-parse", "--short", "HEAD"], cwd)
    branch = b.stdout.strip() or "?"
    rev = r.stdout.strip() or "?"
    return f"{branch}@{rev}"


def bench(
    root: Path,
    transitive: int,
    alias: int,
    autogen: int | None,
    top_k: int,
    systems: str,
    out: Path,
    reuse_questions: Path | None,
) -> dict:
    urag_dir = root / ".urag"
    if urag_dir.exists():
        shutil.rmtree(urag_dir)
    print(f"[bench] fresh index {root}")
    t0 = time.perf_counter()
    r = urag_sh(["init", "--root", str(root), "--full"], ROOT)
    index_seconds = time.perf_counter() - t0
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    print(f"[bench] indexed in {index_seconds:.1f}s")
    qfile = reuse_questions
    if reuse_questions:
        data = json.loads(reuse_questions.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "questions" in data:
            qfile = reuse_questions.with_suffix(".questions.jsonl")
            qfile.write_text(
                "\n".join(json.dumps(q, ensure_ascii=False) for q in data["questions"])
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[bench] reusing {len(data['questions'])} questions from "
                f"{reuse_questions.name}"
            )
    args = [
        "eval",
        "--root",
        str(root),
        "--top-k",
        str(top_k),
        "--systems",
        systems,
        "--json",
        "--report",
        str(out),
    ]
    if qfile:
        args += ["--questions", str(qfile), "--reresolve"]
    else:
        if autogen:
            args += ["--autogen", str(autogen)]
        if transitive:
            args += ["--transitive", str(transitive)]
        if alias:
            args += ["--alias", str(alias)]
    print(f"[bench] eval {root} ({' '.join(args[1:])})")
    r = urag_sh(args, ROOT)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    report = json.loads(out.read_text(encoding="utf-8"))
    report["index_seconds"] = round(index_seconds, 3)
    return report


def summarize(tag: str, report: dict) -> None:
    print(f"\n=== {tag} ===")
    for name, a in report["systems"].items():
        if not a:
            print(f"  {name:<16} (no results — system produced nothing)")
            continue
        print(
            f"  {name:<16} recall={a.get('unit_recall', 0):.3f} "
            f"precision={a.get('precision', 0):.3f} "
            f"indirect_recall={a.get('indirect_recall', 0):.3f} "
            f"mrr={a.get('mrr', 0):.3f} "
            f"tokens={a.get('mean_tokens', 0):.0f} "
            f"p50={a.get('p50_sec', 0) * 1000:.0f}ms"
        )
    if report.get("index_seconds") is not None:
        print(f"  index build: {report['index_seconds']:.1f}s")


def compare(old_path: Path, new_path: Path) -> None:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    oq = [
        (q.get("query"), q.get("label"), q.get("target"))
        for q in old.get("questions", [])
    ]
    nq = [
        (q.get("query"), q.get("label"), q.get("target"))
        for q in new.get("questions", [])
    ]
    print(f"\n=== compare: {old_path.name} -> {new_path.name} ===")
    if len(oq) != len(nq):
        print(f"  WARNING: question sets differ ({len(oq)} vs {len(nq)} questions)")
    elif set(oq) != set(nq):
        print("  WARNING: question contents differ (labels/targets)")
    systems = sorted(set(old.get("systems", {})) | set(new.get("systems", {})))
    for name in systems:
        a, b = (
            old.get("systems", {}).get(name, {}),
            new.get("systems", {}).get(name, {}),
        )
        if not a or not b:
            print(f"  {name:<16} present in only one report")
            continue
        for key, fmt in (
            ("unit_recall", ".3f"),
            ("precision", ".3f"),
            ("mrr", ".3f"),
            ("indirect_recall", ".3f"),
        ):
            da, db = a.get(key) or 0.0, b.get(key) or 0.0
            sign = "+" if db >= da else ""
            print(
                f"  {name:<16} {key:<16} {da:{fmt}} -> {db:{fmt}} ({sign}{db - da:{fmt}})"
            )
        ta, tb = a.get("mean_tokens") or 0.0, b.get("mean_tokens") or 0.0
        print(f"  {name:<16} {'mean_tokens':<16} {ta:.0f} -> {tb:.0f} ({tb - ta:+.0f})")
        pa, pb = (a.get("p50_sec") or 0.0) * 1000, (b.get("p50_sec") or 0.0) * 1000
        print(f"  {name:<16} {'p50_ms':<16} {pa:.1f} -> {pb:.1f} ({pb - pa:+.1f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", action="append", default=[], help="extra project roots to bench"
    )
    ap.add_argument("--self", action="store_true", help="also bench urag's own repo")
    ap.add_argument(
        "--yes",
        action="store_true",
        help="confirm wiping .urag of non-fixture roots without prompting",
    )
    ap.add_argument("--transitive", type=int, default=25)
    ap.add_argument("--alias", type=int, default=25)
    ap.add_argument(
        "--autogen",
        type=int,
        default=None,
        help="N definition + N call questions (default: eval default)",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--systems",
        default="urag-hybrid,urag-callers,urag-transitive,rg,read",
    )
    ap.add_argument(
        "--reuse-questions",
        type=Path,
        help="reuse questions from a previous report JSON (gold re-derived, no autogen)",
    )
    ap.add_argument(
        "--compare", nargs=2, metavar=("OLD", "NEW"), help="diff two reports and exit"
    )
    ap.add_argument(
        "--no-html", action="store_true", help="skip HTML report generation"
    )
    args = ap.parse_args()

    if args.top_k < 1:
        ap.error("--top-k must be >= 1")
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return

    REPORTS.mkdir(parents=True, exist_ok=True)
    rev = git_head(ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    roots = [FIXTURE] + [Path(r) for r in args.root]
    if args.self:
        roots.append(ROOT)
    for root in roots:
        if root.resolve() != FIXTURE.resolve() and not args.yes:
            answer = input(f"wipe .urag/ of {root} and rebuild? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("skipped")
                continue

    for root in roots:
        out = REPORTS / f"{root.name}-{rev.replace('/', '-')}-{stamp}.json"
        report = bench(
            root,
            args.transitive,
            args.alias,
            args.autogen,
            args.top_k,
            args.systems,
            out,
            args.reuse_questions,
        )
        report["rev"] = rev
        report["root"] = str(root)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summarize(f"{root.name} ({rev})", report)
        print(f"  report: {out.relative_to(ROOT)}")
        if not args.no_html:
            from html_report import render_report

            html = render_report(report, title=out.name)
            html_path = out.with_suffix(".html")
            html_path.write_text(html, encoding="utf-8")
            print(f"  html:   {html_path.relative_to(ROOT)} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
