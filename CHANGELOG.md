# Changelog

## 0.4.0 - 2026-10-07

C/C++ coverage and graph fixes (found while dogfooding on a C++/Vulkan
JPEG XL encoder):

- C++ `extern "C"` blocks are now indexed. The C grammar turned
  `extern "C" { ... }` into ERROR nodes, so entire public C API surfaces were
  invisible; `.h` files are now parsed with the C++ grammar (a C superset
  that also handles `#ifdef`-guarded `extern "C"`), and `linkage_specification`
  plus `preproc_if`/`preproc_ifdef` bodies are walked. Call edges inside
  these blocks now resolve (impact analysis is no longer silently
  incomplete).
- Header API surface is searchable: `typedef enum`/`typedef struct` inside
  preprocessor guards are extracted (enumerator names land in the signature),
  and function prototypes in headers become `function` units, so
  `urag resolve gjxl_encode_rgb8` finds the declaration even before the
  definition.
- `urag dependents` works for C/C++: include units are matched by
  path-component suffix (`src/encoder/foo.h` matches
  `#include "encoder/foo.h"`), and symbol targets fall back to the
  dependents of the defining file plus files that reference/call the symbol.
- C/C++ reference edges: type mentions (`TreeMode mode`),
  qualified uses (`TreeMode::kSingle`), and `new` constructions now
  produce `ref_edges`, so `urag references <Type>` works in C++ projects.
- New languages (no new dependencies): GLSL (`.comp`, `.glsl`, ... parsed with
  the C++ grammar), CMake (`CMakeLists.txt`/`.cmake` → `option`/`set`/targets/
  tests/dependencies as units), PowerShell (`.ps1`/`.psm1` function
  definitions), and `.inc` files (as C++). `urag doctor` hints when a
  language present on disk is missing from `index.languages`.
- `urag callees` accepts a symbol name (not only a unit id), matching
  `callers`.
- Windows: piping output into a consumer that exits early
  (`... | Select-Object -First N`) no longer crashes with
  `OSError [Errno 22]` from rich's legacy console renderer;
  closed-pipe writes are silently dropped.
- `urag eval` fails loudly when `rg`/`read` baselines are requested but
  ripgrep is not installed, instead of scoring the baseline 0.00.
- Extractor version bumped to 4; existing indexes re-extract automatically.

## 0.3.0 - 2026-10-06

- Retrieval transparency: search responses now report `mode_requested`
  (what the caller asked for), the effective `mode`, `dense_ready`, and a
  `fallback` reason. Hybrid requests on a lexical-only index (provider
  `none` or no embeddings) run lexical search and say so instead of
  silently returning lexical results labeled `hybrid`.
- Fixed multi-word queries misclassified as exact symbol lookups (e.g. a
  query starting with `P7` or `GPU`), which exact-filtered the full phrase
  and returned zero results; exact symbol searches now fall back to broader
  lexical terms when nothing matches exactly.
- `urag_status` / MCP `urag_status` report dense readiness
  (`dense_ready`, `dense_note`), and `urag search` prints the fallback
  reason when one applies.
- `urag doctor` now includes a coverage report: files present on disk are
  classified as indexed, excluded, unsupported extension (e.g. `.ps1`,
  `.comp`), language-disabled, too large, or not indexed yet, with examples
  and top unsupported extensions.
- Lexical ranking is now intent-sensitive: units matching more distinct
  query terms rank first, single-term matches are dropped for longer
  queries when multi-term matches exist, and `config_key`/`import` units
  are demoted unless the query asks about config or imports. Q&A queries
  no longer surface MCP command entries and import declarations above
  implementation units.
- Provenance is explicit: search results carry `indexed_commit` (source
  attribution) and `stale_basis` (`sha256` content hash vs `git-diff`
  commit membership), and search responses include the current `head`.
  `fetch_unit`/`callees` also report the freshness basis. An unchanged file
  with an older indexing commit is correctly `stale: false`.
- Skill and MCP instructions now route known identifiers to `resolve`,
  impact questions to `callers`/`references`/`dependents`, and broad
  questions to multiple focused searches.

## 0.2.0 - 2026-08-28

- MCP tools renamed with a `urag_` namespace (`urag_search`,
  `urag_fetch_unit`, `urag_callers`, ...) so they never collide with other
  servers; instructions and the skill doc updated. If a harness adds its own
  server-name prefix, the names appear as `<server>_urag_*`.
- Incremental indexing no longer re-hashes files whose size+mtime match
  the stored values (any repo kind). Same-size-same-mtime content edits are
  intentionally not detected by plain re-index runs; the file watcher and
  `--full`-style rebuilds still pick up real edits. Legacy rows without a
  stored hash are re-indexed once to backfill.
- Retrieval precision: definition queries now ignore ~90 common English
  words as exact symbol identifiers (user, count, code, file, test, change,
  find, ...), reducing false-positive exact-match boosts.
- Module structure: the call-graph/reference queries moved to
  `db_edges.py`, search/navigation queries to `db_search.py` (mixins over
  the storage core), and the native-language extractors split into
  per-language modules (`go_ext`, `rust_ext`, `java_ext`, `c_ext`,
  `csharp_ext`) with shared helpers in `native_common`; `native_ext` remains
  as a compatibility re-export. Public imports are unchanged.
- CLI: `read`, `status`, and `doctor` now support `--json` machine-readable
  output; `doctor --json` reports a structured health payload and exits
  non-zero on failure.
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
