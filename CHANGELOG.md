# Changelog

## Unreleased

- Module structure: the call-graph/reference queries moved to
  `db_edges.py`, search/navigation queries to `db_search.py` (mixins over
  the storage core), and the native-language extractors split into
  per-language modules (`go_ext`, `rust_ext`, `java_ext`, `c_ext`,
  `csharp_ext`) with shared helpers in `native_common`; `native_ext` remains
  as a compatibility re-export. Public imports are unchanged.
- Incremental indexing in git repos no longer re-hashes files whose
  size+mtime match and whose working-tree content is clean vs HEAD
  (untracked/modified files still get content-verified). Non-git repos keep
  the exact-hash behavior.
- CLI: `read`, `status`, and `doctor` now support `--json` machine-readable
  output; `doctor --json` reports a structured health payload and exits
  non-zero on failure.
- Retrieval precision: definition queries no longer treat common English
  words (user, count, code, file, test, ...) as exact symbol identifiers,
  which reduces false-positive exact-match boosts.
- MCP: documented the tool-naming contract — server tools are short names
  (`search`, `fetch_unit`, ...) and agent harnesses expose them prefixed
  with the server name (e.g. `urag_search`); instructions updated in the
  server and the skill doc.
- Fixed a destructive read path: opening an index whose stored vector
  dimension differs from `embedding.dimension` now raises a clear error
  instead of silently dropping all embeddings; the rebuild only happens on
  explicit migrate opens (`urag index`, `urag init --full`,
  `urag embed --reindex`, MCP `index_now`/`init_project`).
- Fixed `http_timeout` being dropped when the config was rewritten; fixed
  the embedder cache key missing `dimension` in the MCP server; MCP logs
  embedding fallbacks to stderr instead of degrading silently.
- `urag embed` now verifies the new local model loads before purging the
  old model cache, and clears old embeddings only when something changed.
- Indexer: `refs_pending`/`extractor_version` upgrade flags are now written
  after a successful re-extraction (no half-migrated indexes after a crash);
  added a process-wide write lock so the watcher and manual runs can't
  interleave; `index_paths` now respects `max_file_bytes`.
- Watcher: fixed a debounce race where a stale timer flush could clear the
  pending set of a newer timer (duplicate index work).
- Schema: added indexes on `call_edges.callee_full`, `ref_edges.ref_full`,
  `import_aliases.target`, `units.unit_type`, `files.language`; orphan
  cleanup now runs on index opens only.
- Fixed `%` wildcard leaks in `resolve_units`/`importers`, an off-by-one in
  `load_evidence` for units with `start_line = 0`, and char-vs-byte offset
  attribution in the chunk baseline.
- `--evidence` budgets are now split evenly with no over-allocation floor;
  navigation budgets (`resolve`, `children`, `symbols`) respect
  `retrieval.max_evidence_tokens`; `top_k=0` is honored.
- Retrieval: definition-symbol extraction no longer captures trailing words
  ("where is X defined in tests"); regexes compiled once; eval loop closures
  now bind their question (previously all systems re-ran the last question).
- C# extractor reuses a cached parser (was recompiling the grammar per
  method); Python extractor defers tree-sitter grammar compilation to first
  use; eval orchestration moved out of the CLI into `eval.run_eval`.
- Eval harness: `--reresolve` now re-derives gold for reference questions
  from the current index (previously stale ids survived index rebuilds).
- Removed dead code (unused DB methods, model helpers, extractor helpers)
  and unused imports; added a ruff configuration and a CI lint step.

## 0.1.5 - 2026-08-26

- Added a reference index (type mentions, constructions, bases, generics,
  casts, attributes) for Python, TypeScript/JavaScript, Java, and C#, plus
  `urag references` / MCP `references` with multi-hop traversal. `callers`
  now also matches object constructions (`new MainWindow()`) and event
  subscriptions (`Handler += ...`).
- Added XML family support (`.xaml`, `.axaml`, `.xml`, `.csproj`, `.props`,
  `.targets`): `x:Class` classes, `x:Key` resources, `DataTemplate`
  templates, event handlers, and markup references (`{x:Static}`,
  `{StaticResource}`, element tags, attached properties) feed search and the
  reference index.
- Added `urag deadcode` / MCP `dead_symbols`: heuristic candidates with no
  incoming calls or references.
- Impact queries mentioning "references"/"uses" now route to the reference
  index instead of the call graph.
- Indexing UX: flushed live progress with embedding rate/ETA, an
  embedder-load notice (first run downloads the model), and resume guidance
  for interrupted first indexes. Indexes built by older extractors are
  re-extracted automatically on the next `urag index`.
- CLI: `urag read FILE START END` now accepts positional line ranges;
  `urag doctor` hints when XML family files exist but the config predates
  XAML support.
- Evaluation: `eval --reference N` auto-generates reference questions with
  provable gold and a `urag-references` system; the benchmark suite covers
  the new capability.

## 0.1.4 - 2026-08-26

- Retrieval: route explicit definition queries through exact structural symbol
  resolution and prioritize qualified exact matches.
- Retrieval: improve impact target parsing for C++ member chains, C# generic
  expressions, and symbols whose names overlap with English verbs.
- Benchmarks: add adaptive production retrieval and per-label metrics, and
  harden reusable question validation, root selection, and ambiguity handling.
- Benchmarks: validate retrieval improvements across Python, C#, C++, and the
  URAG repository.

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
