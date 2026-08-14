# Changelog

## 0.1.3.1 - 2026-08-13

- Benchmarks: added opencode-style baselines (`rg` grep emulation with
  matching-line context, `read` whole-file emulation of the grep-then-read
  agent loop), consistent token accounting (chars/4 of context actually
  returned), stopword/segment-aware grep terms, and a warmup query before
  timing.
- Benchmarks: reports now record index-build time, schema/version fields, and
  per-system-per-question hit details; `benchmarks/html_report.py` renders a
  self-contained HTML report with aggregate tables, charts, retrieval samples
  per system, and data-driven explanations of where urag wins or loses.
- Benchmarks: `run_bench.py` gained `--compare` (diff two reports),
  `--reuse-questions` (cross-branch comparison with gold re-derived from the
  current index via `eval --reresolve`), `--yes`, and HTML output;
  `benchmarks/reports/` is no longer tracked in git.
- Fixed the oracle baseline ignoring all but the first gold file, multi-file
  gold handling (`gold_files`) with fractional file recall, and per-query
  alignment for graph-only systems in JSON reports.
- Changed the default embedding model to `BAAI/bge-base-en-v1.5`
  (768-dimensional) and added a `urag embed` command to inspect or switch
  the model/provider; switching clears old vectors, removes the old model
  from the local cache, and re-embeds on the next index run.
- Model/dimension mismatches are now rejected with a clear error instead of
  failing mid-embedding.
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
