"""Git awareness: commit provenance, diff-based invalidation, staleness.

All operations are best-effort: on any failure (not a repo, no commits,
git not installed) the indexer falls back to mtime-based checks and
staleness reports are skipped.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class Git:
    def __init__(self, root: Path):
        self.root = root
        self._head: str | None = None
        self._head_checked = False

    def _run(self, args: list[str], cwd: Path | None = None) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(cwd or self.root),
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout

    def is_repo(self) -> bool:
        return self._run(["rev-parse", "--is-inside-work-tree"]) is not None

    def head(self, refresh: bool = False) -> str | None:
        if refresh:
            self._head_checked = False
        if not self._head_checked:
            self._head_checked = True
            out = self._run(["rev-parse", "HEAD"])
            self._head = out.strip() if out else None
        return self._head

    def changed_paths(self) -> tuple[set[str], set[str]]:
        """(changed, deleted) repo-relative paths vs the working tree."""
        out = self._run(["status", "--porcelain", "-z", "--untracked-files=all"])
        changed: set[str] = set()
        deleted: set[str] = set()
        if not out:
            return changed, deleted
        entries = out.split("\0")
        i = 0
        while i < len(entries):
            if len(entries[i]) < 2:
                i += 1
                continue
            x, y = entries[i][0], entries[i][1]
            i += 1
            if i >= len(entries):
                break
            path = entries[i]
            i += 1
            if x == "R":
                if i < len(entries):  # rename: old path already consumed, next is new
                    i += 1
            rel = Path(path).as_posix()
            if x in ("D", "U") or (x == " " and y == "D"):
                deleted.add(rel)
            else:
                changed.add(rel)
        return changed, deleted

    def changed_since(self, commit: str) -> set[str]:
        """Files that differ between commit and the working tree."""
        out = self._run(["diff", "--name-only", "-z", commit])
        if not out:
            return set()
        return {Path(p).as_posix() for p in out.split("\0") if p}
