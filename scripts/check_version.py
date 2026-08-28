"""Release consistency check: pyproject version, __init__ version, and tag.

Usage: python scripts/check_version.py [vX.Y.Z]

Exits non-zero (with a joined message) when:
- src/urag/__init__.py __version__ != pyproject [project].version
- the git tag (without leading 'v') != pyproject [project].version
"""

import re
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
pyproject_version = data["project"]["version"]

init = (root / "src" / "urag" / "__init__.py").read_text(encoding="utf-8")
match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
init_version = match.group(1) if match else None

errors: list[str] = []
if init_version != pyproject_version:
    errors.append(
        f"src/urag/__init__.py __version__={init_version!r} does not match "
        f"pyproject version {pyproject_version!r}"
    )
tag = sys.argv[1] if len(sys.argv) > 1 else ""
if tag.startswith("v") and tag[1:] != pyproject_version:
    errors.append(f"tag {tag!r} does not match pyproject version {pyproject_version!r}")

if errors:
    raise SystemExit("\n".join(errors))
print(f"version ok: {pyproject_version}")
