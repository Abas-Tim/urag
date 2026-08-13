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

    def _parse_status_porcelain(self, out: str) -> tuple[set[str], set[str], set[str]]:
        """(changed, deleted, untracked) from `git status --porcelain -z`.

        Entries are `XY <path>` NUL-terminated; renames carry the target
        path as an extra NUL-separated field."""
        changed: set[str] = set()
        deleted: set[str] = set()
        untracked: set[str] = set()
        if not out:
            return changed, deleted, untracked
        entries = out.split("\0")
        i = 0
        while i < len(entries):
            e = entries[i]
            i += 1
            if len(e) < 4:
                continue
            x, y = e[0], e[1]
            path = Path(e[3:]).as_posix()
            if x == "R" and i < len(entries):
                target = Path(entries[i]).as_posix()
                i += 1
                if target:
                    changed.add(target)
                deleted.add(path)
                continue
            if x == "?":
                untracked.add(path)
            elif x in ("D", "U") or (x == " " and y == "D"):
                deleted.add(path)
            elif path:
                changed.add(path)
        return changed, deleted, untracked

    def changed_paths(self) -> tuple[set[str], set[str]]:
        """(changed, deleted) repo-relative paths vs the working tree."""
        changed, deleted, _ = self._parse_status_porcelain(
            self._run(["status", "--porcelain", "-z", "--untracked-files=all"]) or ""
        )
        return changed, deleted

    def changed_since(self, commit: str) -> set[str]:
        """Files that differ between commit and the working tree."""
        out = self._run(["diff", "--name-only", "-z", commit])
        if not out:
            return set()
        return {Path(p).as_posix() for p in out.split("\0") if p}

    def current_branch(self) -> str | None:
        out = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return out.strip() if out else None

    def recent_changes(self, limit: int = 20) -> dict:
        """Working-tree changes plus recent commit file lists (best-effort)."""
        result: dict = {
            "branch": self.current_branch(),
            "head": self.head(refresh=True),
            "working": {"changed": [], "deleted": [], "untracked": []},
            "commits": [],
        }
        if not self.is_repo():
            return result
        changed, deleted, untracked = self._parse_status_porcelain(
            self._run(["status", "--porcelain", "-z", "--untracked-files=all"]) or ""
        )
        result["working"]["deleted"] = sorted(deleted)
        result["working"]["changed"] = sorted(changed)
        result["working"]["untracked"] = sorted(untracked)
        log = self._run(
            [
                "log",
                f"-n{max(1, limit)}",
                "--name-only",
                "--pretty=format:%H%x1f%h%x1f%s%x1f%cI",
            ]
        )
        if log:
            for block in log.split("\n\n"):
                lines = [l for l in block.splitlines() if l]
                if not lines:
                    continue
                fields = lines[0].split("\x1f")
                if len(fields) < 4:
                    continue
                result["commits"].append(
                    {
                        "commit": fields[0],
                        "short": fields[1],
                        "subject": fields[2],
                        "date": fields[3],
                        "files": [Path(l).as_posix() for l in lines[1:]],
                    }
                )
        return result
