# Releasing

urag publishes two kinds of releases to PyPI:

- **Stable releases** (`vX.Y.Z`) — the default for `pip install urag-cli`.
- **Pre-releases** (`vX.Y.ZaN` / `vX.Y.ZrcN`) — opt-in for testers. PyPI
  accepts them, but installers exclude pre-releases by default, so a
  pre-release never shadows the latest stable version.

Both flows use the same `publish` workflow (`.github/workflows/publish.yml`),
which builds the wheel + sdist, uploads them to PyPI, and attaches the
artifacts to a GitHub release (flagged "pre-release" when the tag contains
an `a`/`b`/`rc`/`dev` segment). The workflow verifies that the tag, the
`[project] version` in `pyproject.toml`, and `__version__` in
`src/urag/__init__.py` all agree, so a mismatch can never publish.

## Prerequisites

- `PYPI_TOKEN` secret (PyPI API token for the `urag-cli` project) set on the
  repository.
- The branch/tag being published passes CI (tests + lint).

## Stable release

1. Merge the feature PRs into `main`.
2. Bump the version in **two places**:
   - `pyproject.toml` → `[project] version`
   - `src/urag/__init__.py` → `__version__`
3. Update `CHANGELOG.md`: rename the `## Unreleased` section to
   `## X.Y.Z - YYYY-MM-DD`.
4. Commit (`chore(release): prepare X.Y.Z`), push, and tag:

   ```bash
   git tag -a v0.2.0 -m "urag 0.2.0"
   git push origin v0.2.0
   ```

5. The publish workflow runs on the tag, uploads to PyPI, and creates the
   GitHub release. Verify with:

   ```bash
   pip install urag-cli            # resolves to the new stable
   urag --version
   ```

## Pre-release (based on a PR branch)

Use when testers should try a pending change set (e.g. the latest PR)
before it is finalized:

1. From the branch with the changes, bump the version to the next
   pre-release of the upcoming release line, e.g. `0.2.0a1` (or `0.2.0rc1`
   when the branch is believed final). Update both version locations and
   the CHANGELOG header.
2. Commit and tag:

   ```bash
   git tag -a v0.2.0a1 -m "urag 0.2.0a1 (pre-release)"
   git push origin v0.2.0a1
   ```

3. The workflow publishes to PyPI and creates a GitHub **pre-release**.
   Testers install it explicitly (pre-releases are never the default):

   ```bash
   pip install urag-cli==0.2.0a1
   # or the latest pre-release of the project:
   pip install --pre urag-cli
   ```

4. When the branch merges into `main`, follow the stable flow with the
   final version (e.g. `0.2.0`).

## Manual re-publish

The workflow also accepts a `workflow_dispatch` with a `ref` input (the tag
name). Use it to re-run a failed publish without re-tagging; the version
checks still apply.
