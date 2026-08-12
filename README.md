# urag

**Universal Repository Agent Graph**

Structure-aware retrieval and impact analysis for software repositories. urag
indexes code symbols, documentation, call sites, and source locations into a
portable SQLite database. It then returns small, cited result packets that are
useful to developers and AI agents without loading whole files into context.

urag is a retrieval layer, not an LLM. It finds the relevant repository
context; the developer or agent uses that context to answer the question.

See [urag in action](docs/urag-in-action.md) for a complete indexing and agent
response walkthrough.

## Why urag

Traditional text search is good at finding strings. Generic RAG is good at
finding similar text. Repository questions often need both, plus structural
information such as:

- Where is a function, class, interface, or type defined?
- How does a feature work across modules?
- What calls this function?
- What is the downstream impact of changing a symbol?
- Which exact source lines support the result?

urag addresses these questions with:

- Tree-sitter parsing for symbol-aware indexing.
- FTS5 lexical search for exact names and identifiers.
- sqlite-vec embeddings for conceptual search.
- Reciprocal Rank Fusion for the default hybrid search.
- A static call graph for direct and multi-hop caller queries.
- Import-alias resolution for selected languages.
- Lazy source retrieval, so search results stay compact.
- Incremental indexing and a file watcher for active repositories.
- A CLI and a stdio MCP server for agent harnesses.

## How It Works

```text
repository files
       |
       v
file discovery and incremental checks
       |
       v
tree-sitter extractors
       |
       +--> symbols, signatures, summaries, spans
       +--> call edges and import aliases
       |
       v
SQLite project index
       |
       +--> FTS5 lexical index
       +--> sqlite-vec dense index
       +--> files, units, calls, and aliases
       |
       v
ranked result packets
       |
       v
exact source span fetched only when needed
```

Each indexed unit has three practical layers:

| Layer | Contents | Purpose |
| --- | --- | --- |
| Retrieval key | Name, qualified name, signature, summary, concepts, relationships | Search and embedding input |
| Relationships | File, line, byte span, parent, calls, aliases | Navigation and impact analysis |
| Evidence | Exact source lines on disk | Final context, loaded on demand |

The index is stored per project in `.urag/index.db`. SQLite uses WAL mode, and
new embeddings are written in batches of 64 units.

## Installation

Requirements:

- Python 3.12 or newer.
- `git` is optional, but is used for commit provenance and changed-file
  detection when available.
- The default local embedding model downloads on first use and is cached
  locally. No API key is required for the default provider.

### From Source

```bash
git clone https://github.com/Abas-Tim/urag.git
cd urag
uv sync
uv run urag --version
```

To install the command globally with uv:

```bash
uv tool install .
urag --version
```

The repository also contains bootstrap installers for macOS, Linux, and
Windows under `bootstrap/`.

## Quick Start

Run these commands from a repository you want to index:

```bash
urag init --root /path/to/project --full
urag search "how does authentication work" --root /path/to/project
urag search "TokenValidator.validate" --mode lexical --top-k 3 --root /path/to/project
urag callers index_all --depth 3 --root /path/to/project
urag get UNIT_ID --root /path/to/project
```

`urag init` creates `.urag/`, the project configuration, and the database.
Use `--full` for the initial index. Later updates are incremental:

```bash
urag index --root /path/to/project
urag watch --root /path/to/project
```

`watch` debounces filesystem events and re-indexes changed files. It can also
run a periodic full rescan with `--rescan 30`.

## CLI

| Command | Description |
| --- | --- |
| `urag init` | Create project configuration and an empty index; add `.urag/` to `.gitignore` |
| `urag init --full` | Create the project index and index all eligible files |
| `urag index` | Incrementally index new, changed, and deleted files |
| `urag watch` | Keep the index updated while files change |
| `urag search QUERY` | Search symbols and documentation |
| `urag resolve NAME` | Find an exact symbol definition by name |
| `urag callers NAME` | Find direct callers of a symbol |
| `urag callers NAME --depth 3` | Find callers through multiple call-graph hops |
| `urag callees UNIT_ID` | List what a unit calls (its call sites) |
| `urag dependents NAME` | Find what imports (depends on) a module or symbol |
| `urag symbols FILE` | List every indexed unit in a file |
| `urag read FILE` | Read a file (or a `--start`/`--end` line range) |
| `urag get UNIT_ID` | Fetch the exact source span for a search result |
| `urag recent` | Show recent git changes (branch, working tree, commits) |
| `urag status` | Show file, unit, embedding, language, and database statistics |
| `urag doctor` | Check index and embedding health |
| `urag classify QUERY` | Show the query class and selected context budget |
| `urag eval` | Compare retrieval systems on a question set |
| `urag mcp` | Run the MCP server over stdio |

Search supports three modes:

| Mode | Best for |
| --- | --- |
| `lexical` | Exact names, identifiers, paths, and configuration keys |
| `dense` | Conceptual or natural-language questions |
| `hybrid` | The default; combines lexical and dense results with RRF |

Use `--json` for machine-readable output, `--language` to filter results, and
`--evidence` to include trimmed source spans. The `get` command returns the
full current span for a unit id.

## Adaptive Retrieval

Queries are routed with a deterministic, zero-model-cost classifier. The
classifier selects a default result count and evidence budget based on the
question shape:

| Class | Typical query | Default results | Evidence budget |
| --- | --- | ---: | ---: |
| `symbol` | `TokenValidator.validate` | 3 | 800 tokens |
| `local` | `how does token validation work` | 5 | 2,000 tokens |
| `debugging` | `why does this crash across modules` | 8 | 4,000 tokens |
| `impact` | `what calls parse_token` | 10 | 6,000 tokens |

Exact symbol queries are routed to lexical search. Impact queries are routed
to the call graph when a target symbol can be identified. `--top-k` and
`query_class` can override the defaults where supported.

The configured `retrieval.max_evidence_tokens` is a global ceiling and can
reduce the class budget; the generated default is 1,500 tokens.

## Call Graph And Impact Analysis

During indexing, urag scans supported source files for call expressions and
stores:

- The enclosing caller unit.
- The last callee segment, such as `validate`.
- The full written callee chain, such as `self.validate` or `os.path.exists`.
- The call-site line number.

This powers direct and multi-hop queries:

```bash
urag callers validate
urag callers validate --depth 3
urag search "what breaks if parse_token changes"
```

Multi-hop traversal uses breadth-first search, records the shortest hop for
each result, and terminates safely on cycles. Import aliases are resolved for
Python, TypeScript, Go, Rust, and C# when the index contains the relevant
bindings:

```bash
urag callers os.path.exists
urag callers core.http.fetch
```

The graph is static and intentionally approximate. Dynamic dispatch,
reflection, generated code, and some instance-method chains are not resolved.
The graph should be treated as impact evidence, not as a runtime dependency
model.

## Supported Languages

| Language | Extensions | Extracted information |
| --- | --- | --- |
| Python | `.py`, `.pyi` | Functions, classes, methods, imports, calls, aliases |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | Functions, classes, interfaces, types, enums, imports, calls, aliases |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | Functions, classes, methods, imports, calls, aliases |
| Go | `.go` | Functions, methods, structs, interfaces, imports, calls, aliases |
| Rust | `.rs` | Functions, methods, structs, traits, enums, imports, calls, aliases |
| Java | `.java` | Methods, constructors, classes, interfaces, enums, imports, calls |
| C | `.c`, `.h` | Functions, structs, unions, typedefs, includes, calls |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh` | Functions, methods, classes, structs, namespaces, includes, calls |
| C# | `.cs` | Classes, interfaces, structs, records, enums, methods, usings, calls, aliases |
| Markdown | `.md`, `.markdown`, `.mdx` | Heading-based document chunks and hierarchy |
| JSON | `.json` | `config_key` units with dotted qualnames |
| YAML | `.yaml`, `.yml` | `config_key` units with dotted qualnames |
| TOML | `.toml` | `config_key` units with dotted qualnames |
| INI | `.ini`, `.cfg`, `.conf`, `.properties` | `config_key` units with dotted qualnames |
| Env | `.env`, `.env.*` | `config_key` units for `KEY=value` entries |

Files are filtered by `.gitignore`, built-in exclusions, configured languages,
and a default maximum size of 1 MB. These settings are configurable in
`.urag/urag.toml`.

## Embeddings And Configuration

The default provider is a local ONNX model through FastEmbed:

```toml
[embedding]
provider = "local"
model = "BAAI/bge-small-en-v1.5"
dimension = 384
```

The model is downloaded on first use and cached in `%LOCALAPPDATA%/urag` on
Windows or `~/.cache/urag` on other systems. An OpenAI-compatible HTTP
embedding endpoint is also supported:

```toml
[embedding]
provider = "http"
dimension = 384
http_url = "http://localhost:11434/v1"
http_model = "nomic-embed-text"
http_api_key = ""
```

Keep API keys in the local `.urag/urag.toml` only and do not commit them.
Projects that already have an index can use `provider = "none"` for
lexical-only retrieval; dense retrieval and embedding new units require an
embedding provider.

## Git Freshness

For Git repositories, indexed files record the commit used at indexing time.
Search and `get` results include the short commit and a `stale` flag when the
file differs from that indexed revision. Non-Git projects use file metadata
checks instead.

Freshness detection is best-effort. Run `urag index` after a branch switch,
pull, or other large repository change before relying on old evidence. Static
analysis also cannot see runtime-generated relationships.

## MCP Server

Run urag as a stdio MCP server for an agent harness:

```bash
urag mcp --root /path/to/project
```

Example generic MCP configuration:

```json
{
  "mcpServers": {
    "urag": {
      "command": "urag",
      "args": ["mcp", "--root", "/path/to/project"]
    }
  }
}
```

The server exposes these tools:

- `search`: Search symbols and documentation with compact result packets.
- `fetch_unit`: Fetch exact source lines for a result id (includes metadata).
- `fetch_units`: Batch-fetch several unit ids in one call.
- `callers`: Query direct or multi-hop callers.
- `callees`: List what a unit calls (call sites).
- `dependents`: Find what imports a module or symbol.
- `resolve`: Exact symbol definition lookup by name.
- `children`: List the members (methods) of a class/struct.
- `list_files`: List indexed files with language and unit counts.
- `list_symbols`: List every indexed unit in a file.
- `read_file`: Read a file or a line range by path.
- `recent_changes`: Report git branch, HEAD, working-tree changes, and recent
  commits.
- `index_now`: Incrementally re-index changed files.
- `status`: Return index statistics, config, and git state.
- `init_project`: Create and populate an index for a project. Pass
  `embed=false` for a fast lexical-only index.

The intended agent workflow is to search with `top_k=3-5`, fetch exact spans
only for the most relevant one to three units, and call `index_now` after
changes. Agents can also browse with `list_files`/`list_symbols`/`read_file`,
resolve known symbols with `resolve`, and answer impact questions with
`callers`/`callees`/`dependents`.

## Agent Skill

The repository includes `skills/urag/SKILL.md`, a small instruction file for
agent harnesses. It teaches the search-first, fetch-evidence-on-demand
workflow and is included in built packages.

For harnesses that use local skill directories, install it with the harness's
normal skill installation mechanism. For example:

```bash
cp -r skills/urag ~/.config/opencode/skills/urag
cp -r skills/urag ~/.claude/skills/urag
```

## Measuring Retrieval

The evaluation harness compares urag with non-structural baselines on the same
questions:

```bash
urag eval --root . --autogen 10 --top-k 5
urag eval --root . --questions questions.jsonl
urag eval --root . --transitive 25 --alias 25 \
  --systems urag-callers,urag-transitive,urag-hybrid
uv run python benchmarks/run_bench.py --self
```

It measures fractional unit recall@k, file recall@k, precision, MRR, approximate tokens
per retrieval, and p50/p95 latency. Definition and call questions can be
generated with provable gold data from the index. Custom conceptual questions
can be supplied as JSONL:

```json
{"query": "where is TokenValidator defined", "gold_file": "src/auth.py"}
```

### Checked-In Benchmark Snapshot

The following values come from the reports in `benchmarks/reports/`, using
`top_k=5`. They are reference measurements from one environment, not
performance guarantees.

| Dataset | System | Questions | Unit recall | Precision | MRR | p50 | Mean compact tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic fixture | `urag-callers` | 22 | 1.000 | 1.000 | 1.000 | 0.14 ms | 5.0 |
| Synthetic fixture | `urag-transitive` | 22 | 1.000 | 1.000 | 1.000 | 0.11 ms | 6.4 |
| urag repository | `urag-callers` | 47 | 1.000 | 1.000 | 1.000 | 0.48 ms | 41.4 |
| urag repository | `urag-transitive` | 47 | 1.000 | 1.000 | 1.000 | 10.38 ms | 78.2 |
| urag repository | `urag-hybrid` | 57 | 0.316 | 0.077 | 0.252 | 4.38 ms | 134.7 |

The checked-in reports predate the current index-lifecycle and retrieval-quality
changes; rerun the benchmark before comparing new results. The transitive system
reached `indirect_recall=1.000` on the synthetic
fixture and `0.440` on the checked-in urag repository report. In the current
evaluation harness, unit recall is fractional across the gold units for a
question. Graph systems
also run on graph-eligible questions, while the hybrid row includes the full
generated question set, so the rows are directional rather than a universal
ranking.

The efficiency gain urag is designed to provide comes from returning compact
metadata first and loading source spans only when requested. Actual latency,
token count, and retrieval quality depend on repository size, embedding
provider, query type, and index state.

## Development

```bash
uv sync
uv run pytest -q
uv run urag --version
```

CI runs the test suite and CLI smoke check on Python 3.12 and 3.13. The test
suite covers extractors, call extraction, multi-hop traversal, alias
resolution, evaluation metrics, and benchmark fixtures.

## License

MIT. See [LICENSE](LICENSE).
