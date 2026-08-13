---
name: urag
description: Retrieves structure-aware project knowledge from a urag index. Use it to locate symbols, understand repository behavior, inspect documentation, find callers, and assess static impact without reading whole files. Trigger on codebase questions about definitions, architecture, implementation details, dependencies, callers, or configuration in an indexed repository.
---

# urag Project Retrieval

urag is a retrieval and static-analysis layer, not an LLM. Use it to find
relevant repository context, then reason from the returned evidence.

## Core Workflow

Search first, fetch evidence second:

1. Check that the project has an index with `status`.
2. If no index exists, initialize it with `init_project` or
   `urag init --root <root> --full`.
3. Search with `top_k=3-5` first. Prefer compact results over source dumps.
4. Use the returned file and line range to identify the most relevant units.
5. Fetch exact source for only the one to three units needed to answer the
   question.
6. Re-index after edits, pulls, branch switches, or other repository changes.
7. Treat a `stale: true` result as outdated until the project is re-indexed.

Do not read whole files when search results and unit evidence are sufficient.

## CLI

```bash
urag status --root <root>
urag init --root <root> --full
urag search "how does authentication work" --root <root> --top-k 5
urag search "TokenValidator.validate" --root <root> --mode lexical --top-k 3
urag search "cache invalidation" --root <root> --mode dense --top-k 3
urag callers validate --root <root>
urag callers validate --root <root> --depth 3
urag get <unit-id> --root <root>
urag index --root <root>
```

`urag init` creates the project configuration and database. It does not run
the initial full index unless `--full` is supplied. `urag watch --root <root>`
keeps an existing index updated through debounced filesystem events.

Use `--json` for machine-readable search or caller results. Use
`search --evidence` for trimmed source spans, or `get <unit-id>` for the full
current source span.

## Search Modes

- `hybrid`: Default. Combines FTS5 lexical search and sqlite-vec dense search
  with Reciprocal Rank Fusion. Use for most natural-language questions.
- `lexical`: Exact names, identifiers, paths, configuration keys, and error
  strings.
- `dense`: Conceptual or natural-language similarity search.

The deterministic query classifier chooses a default result count and evidence
budget:

| Class | Typical query | Top results | Evidence budget |
| --- | --- | ---: | ---: |
| `symbol` | `TokenValidator.validate` | 3 | 800 tokens |
| `local` | `how does token validation work` | 5 | 2,000 tokens |
| `debugging` | `why does this crash across modules` | 8 | 4,000 tokens |
| `impact` | `what calls parse_token` | 10 | 6,000 tokens |

Use `urag classify "<query>"` to inspect the selected class. Exact symbol
queries use lexical retrieval. Impact queries use the call graph when a target
symbol can be extracted. The configured `max_evidence_tokens` value is a
global ceiling and may reduce the class budget.

## Call Graph

Use the call graph for questions such as "what calls X" or "what breaks if X
changes":

```bash
urag callers parse_token --root <root>
urag callers parse_token --root <root> --depth 3
```

The MCP `callers` tool has the same behavior with `name`, `limit`, and `depth`.
Depth 1 returns direct callers. Larger depths perform breadth-first traversal,
return the shortest `hop` for each result, and terminate safely on cycles.

Fully qualified and aliased calls are supported where indexed bindings exist:

```bash
urag callers os.path.exists --root <root>
urag callers core.http.fetch --root <root>
```

Alias extraction covers Python, TypeScript, Go, Rust, and C#. The graph is
static and approximate. Dynamic dispatch, reflection, generated code, and
some instance-method chains are not resolved. Do not present the graph as a
complete runtime dependency graph.

## MCP Tools

Run the stdio server with:

```bash
urag mcp --root <root>
```

The server exposes these tools:

- `search(query, top_k?, mode?, language?, include_evidence?, query_class?)`
  returns compact packets with metadata, ranks, provenance, and optional
  trimmed evidence. The evidence budget is split across results, not applied
  per result.
- `fetch_unit(unit_id)` returns the exact source span, file, line range,
  symbol metadata, indexed commit, and a stale status.
- `fetch_units(unit_ids[], max_tokens?)` fetches several units in one call.
- `callers(name, limit?, depth?)` returns direct or multi-hop caller packets.
- `callees(unit_id)` returns what a unit calls (the inverse of callers).
- `dependents(target, limit?)` returns the files that import a module/symbol.
- `resolve(name, limit?)` returns exact symbol definitions by name/qualname.
- `children(unit_id, include_siblings?)` lists the members of a class/struct.
- `list_files(language?)` lists indexed files with language and unit counts.
- `list_symbols(file)` lists every indexed unit in a file.
- `read_file(path, start?, end?)` reads a file or a line range by path.
- `recent_changes(limit?)` reports git branch, HEAD, working-tree changes, and
  recent commits with their files.
- `index_now()` incrementally re-indexes changed files and reports statistics.
- `status()` reports the project root, files, units, embeddings, languages,
  provider, model, git branch/HEAD, and last index time.
- `init_project(embed?)` creates and populates the project index. Pass
  `embed=false` for a fast lexical-only index without a model download.

When using MCP, search with `top_k=3-5`, fetch only the most relevant one to
three units, and call `index_now` after changes. Use `resolve` for a known
symbol, `list_symbols` + `read_file` to browse files, and `callers`/`callees`/
`dependents` for impact questions.

## Result Fields

Search packets commonly include:

- `id`: Unit id used by `fetch_unit` or `urag get`.
- `qualname`, `type`, `signature`, and `summary`: Symbol or document metadata.
- `kind`, `concepts`, `relationships`, and `parent_id`: Structural context.
- `file` and `lines`: Repository location.
- `score` and `ranks`: Retrieval score and lexical/dense ranks.
- `commit` and `stale`: Git provenance when available.
- `calls`, `call_line`, `hop`, and `resolved_to`: Call-graph metadata when
  applicable.

Evidence is loaded from the current file on disk. It is not embedded in every
search result, which keeps the normal response compact.

## Supported Content

Language-aware extractors cover Python, TypeScript, JavaScript, Go, Rust,
Java, C, C++, and C#. They index functions, methods, classes or types,
interfaces, enums, imports, signatures, documentation comments, source spans,
and call sites according to the language. Markdown is indexed as heading-based
document chunks. JSON, YAML, TOML, INI, and `.env` files are indexed as
`config_key` units with dotted qualnames, so agents can answer "where is this
setting" and "what env vars exist".

Files are filtered by project `.gitignore`, built-in exclusions, configured
languages, and a default 1 MB file-size limit.

## Freshness And Limitations

Indexing is incremental and uses file metadata. In Git repositories, urag also
records the commit used for each file and marks changed evidence as stale.
Freshness detection is best-effort, so explicitly run `urag index` after a
branch switch, pull, or large change.

The default embedding provider is a local FastEmbed/ONNX model
(`BAAI/bge-base-en-v1.5`, 768d). It downloads on first use and needs no API
key. OpenAI-compatible HTTP embeddings and lexical-only retrieval are also
supported through `.urag/urag.toml`, and `urag embed` changes the model while
clearing the old vectors and cached model files.

The index stores compact retrieval keys and relationship pointers in a
per-project SQLite database. Exact source spans are fetched lazily to reduce
agent context and token usage.
