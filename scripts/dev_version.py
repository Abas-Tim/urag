"""Derive the next dev (pre-release) version for a merged main.

Version scheme: <next patch of the latest stable tag>.dev<commits since tag>.
Example: latest stable tag v0.2.0, 3 commits on main -> 0.2.1.dev3.

Prints the version to stdout, or "SKIP" (details on stderr) when there is
nothing to publish (no stable tags yet, or HEAD is a tagged commit).

Side effects: rewrites the `version` line in pyproject.toml and
`__version__` in src/urag/__init__.py so the built artifacts carry the
derived version. Intended for CI runners only.
"""

import re
import subprocess
import sys
from pathlib import Path

PEP440_DEV = re.compile(r"^\d+\.\d+(\.\d+)?\.dev\d+$")
STABLE_TAG = re.compile(r"^v\d+\.\d+(\.\d+)?$")
VERSION_LINE = re.compile(r'^version\s*=\s*"[^"]+"')

root = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def skip(reason: str) -> None:
    print(reason, file=sys.stderr)
    print("SKIP")


tags = git("tag", "--list", "v[0-9]*").splitlines()
stable = [t[1:] for t in tags if STABLE_TAG.match(t)]
if not stable:
    skip("no stable tags yet")
    raise SystemExit(0)

last = max(stable, key=key)
parts = last.split(".")
parts[-1] = str(int(parts[-1]) + 1)
base = ".".join(parts)

count = int(git("rev-list", "--count", f"v{last}..HEAD"))
if count == 0:
    skip("HEAD is a tagged commit")
    raise SystemExit(0)

version = f"{base}.dev{count}"
if not PEP440_DEV.match(version):
    skip(f"computed invalid version: {version}")
    raise SystemExit(0)

pyproject = root / "pyproject.toml"
lines = pyproject.read_text(encoding="utf-8").splitlines()
in_project = False
for i, line in enumerate(lines):
    if line.startswith("[project]"):
        in_project = True
        continue
    if in_project and line.startswith("["):
        break
    if in_project and VERSION_LINE.match(line):
        lines[i] = f'version = "{version}"'
        break
else:
    skip("could not find the version line in pyproject.toml")
    raise SystemExit(0)
pyproject.write_text("\n".join(lines) + "\n", encoding="utf-8")

init = root / "src" / "urag" / "__init__.py"
text = init.read_text(encoding="utf-8")
text, n = re.subn(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', text, count=1)
if n != 1:
    skip("could not update __version__ in src/urag/__init__.py")
    raise SystemExit(0)
init.write_text(text, encoding="utf-8")

print(version)
