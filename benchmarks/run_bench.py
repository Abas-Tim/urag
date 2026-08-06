"""Benchmark runner for the call-graph suites (transitive + import-alias).

Usage (from repo root):

    uv run python benchmarks/run_bench.py                      # fixture project only
    uv run python benchmarks/run_bench.py --self               # fixture + urag's own repo
    uv run python benchmarks/run_bench.py --transitive 30 --alias 30 --top-k 5

Reports land in benchmarks/reports/<rev>-<timestamp>.json and a summary is
printed per root. Run the same command on feat/benchmarks (before) and the
feature branches (after) to quantify the improvement. The target root's
.urag/ index is wiped and rebuilt fresh on every run so indexer schema
changes never leak stale data into results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "callgraph_fixture"
REPORTS = ROOT / "benchmarks" / "reports"


def sh(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def git_head(cwd: Path) -> str:
    b = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    r = sh(["git", "rev-parse", "--short", "HEAD"], cwd)
    branch = b.stdout.strip() or "?"
    rev = r.stdout.strip() or "?"
    return f"{branch}@{rev}"


def bench(root: Path, transitive: int, alias: int, top_k: int, systems: str, out: Path) -> dict:
    urag_dir = root / ".urag"
    if urag_dir.exists():
        shutil.rmtree(urag_dir)
    print(f"[bench] fresh index {root}")
    r = sh(["urag", "init", "--root", str(root), "--full"], ROOT)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    args = [
        "urag", "eval", "--root", str(root), "--top-k", str(top_k),
        "--systems", systems, "--json", "--report", str(out),
    ]
    if transitive:
        args += ["--transitive", str(transitive)]
    if alias:
        args += ["--alias", str(alias)]
    print(f"[bench] eval {root} ({' '.join(args[2:])})")
    r = sh(args, ROOT)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)
    return json.loads(out.read_text(encoding="utf-8"))


def summarize(tag: str, report: dict) -> None:
    print(f"\n=== {tag} ===")
    for name, a in report["systems"].items():
        print(
            f"  {name:<16} recall={a.get('unit_recall', 0):.3f} "
            f"precision={a.get('precision', 0):.3f} "
            f"indirect_recall={a.get('indirect_recall', 0):.3f} "
            f"mrr={a.get('mrr', 0):.3f} "
            f"tokens={a.get('mean_tokens', 0):.0f} "
            f"p50={a.get('p50_sec', 0) * 1000:.0f}ms"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", default=[], help="extra project roots to bench")
    ap.add_argument("--self", action="store_true", help="also bench urag's own repo")
    ap.add_argument("--transitive", type=int, default=25)
    ap.add_argument("--alias", type=int, default=25)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--systems", default="urag-callers,urag-transitive,urag-hybrid")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    rev = git_head(ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    roots = [FIXTURE] + [Path(r) for r in args.root]
    if args.self:
        roots.append(ROOT)

    for root in roots:
        out = REPORTS / f"{root.name}-{rev.replace('/', '-')}-{stamp}.json"
        report = bench(root, args.transitive, args.alias, args.top_k, args.systems, out)
        report["rev"] = rev
        report["root"] = str(root)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        summarize(f"{root.name} ({rev})", report)
        print(f"  report: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
