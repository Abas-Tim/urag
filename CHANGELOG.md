# Changelog

## Unreleased

- Added agent-facing tools: `fetch_units`, `callees`, `dependents`, `resolve`,
  `children`, `list_files`, `list_symbols`, `read_file`, and `recent_changes`
  (MCP + CLI).
- Added a configuration extractor for JSON, YAML, TOML, INI, and `.env` files.
- Added `init_project(embed=false)` for fast lexical-only indexing, and git
  branch/HEAD reporting in `status`.
- Fixed dead `include_evidence` in `RetrievedUnit.to_dict` and split the
  evidence budget across results instead of applying it per result.
- Fixed `git status --porcelain -z` parsing for changed/deleted/untracked files.

## 0.1.3 - 2026-08-12

- Optimized dense retrieval hydration and embedding storage.
- Improved incremental call-graph resolution and stale target cleanup.
- Corrected UTF-8 and Markdown byte-range handling.
- Strengthened evaluation fixtures and test assertions.

## 0.1.2 - 2026-08-07

- Hardened incremental indexing, database migrations, embeddings, and call-graph updates.
- Improved Markdown, Python, TypeScript, and native-language extraction.
- Improved hybrid retrieval, evidence selection, stale handling, and evaluation coverage.
