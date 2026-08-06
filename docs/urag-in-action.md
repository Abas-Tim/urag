# urag In Action

This document follows a small repository through the complete urag lifecycle:

1. Discover files that should be indexed.
2. Extract symbols, documentation, calls, and aliases.
3. Store the extracted data in SQLite.
4. Create lexical and dense retrieval indexes.
5. Return compact results to an agent.
6. Fetch exact source only when the agent needs evidence.
7. Answer direct and multi-hop impact questions from the call graph.

The JSON values below are representative. Unit ids, scores, commits, and
timestamps depend on the repository and index state.

## Example Repository

```text
demo-project/
  auth.py
  app.py
```

`auth.py`:

```python
def validate_token(token: str) -> bool:
    """Validate an access token."""
    return token != ""
```

`app.py`:

```python
from auth import validate_token

def login(token: str) -> bool:
    return validate_token(token)

def handle_request(token: str) -> bool:
    return login(token)
```

This example contains four indexable units:

| File | Unit | Type | Lines |
| --- | --- | --- | ---: |
| `auth.py` | `validate_token` | function | 1-3 |
| `app.py` | `auth.validate_token` | import | 1 |
| `app.py` | `login` | function | 3-4 |
| `app.py` | `handle_request` | function | 6-7 |

It also contains two call edges:

```text
login           --validate_token, line 4--> validate_token
handle_request  --login, line 7----------> login
```

The import creates an alias binding:

```text
validate_token -> auth.validate_token
```

That binding lets a fully qualified impact query find the bare imported call.

## 1. Initialize The Index

From the project root, run:

```bash
urag init --root /path/to/demo-project --full
```

`init` creates:

```text
demo-project/
  .urag/
    urag.toml
    index.db
  .gitignore
```

`--full` is important for the first run. Without it, `urag init` creates the
configuration and database but does not perform the initial full index.

The default configuration includes the supported languages, built-in
exclusions, a 1 MB maximum file size, and the local embedding provider:

```toml
[embedding]
provider = "local"
model = "BAAI/bge-small-en-v1.5"
dimension = 384

[index]
languages = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "csharp", "markdown"]
exclude = [".urag", ".git", "node_modules", "dist", "build", ".venv"]
```

The default local model is downloaded on first use and cached locally. An
OpenAI-compatible HTTP embedding provider or an existing lexical-only index
can be configured instead.

## 2. File Discovery

The indexer walks the project root recursively and applies these checks in
order:

1. Ignore directories and files matched by the project `.gitignore`, unless
   `ignore_gitignore = true` is configured.
2. Apply built-in and configured exclusions such as `.git`, `.urag`,
   `node_modules`, `dist`, `build`, and virtual environments.
3. Map the file extension to a configured language.
4. Skip languages not listed in `[index].languages`.
5. Skip files larger than `[index].max_file_bytes`, which defaults to 1 MB.

For every accepted file, urag stores file metadata in the `files` table:

| Field | Meaning |
| --- | --- |
| `path` | Project-relative POSIX path |
| `kind` | `source` or `doc` |
| `language` | Extractor language name |
| `size` | Current byte size |
| `mtime` | Current modification time |
| `sha256` | Content hash stored at indexing time |
| `commit` | Git revision used for the index, when available |
| `indexed_at` | Indexing timestamp |

On later runs, size and modification time determine whether a file needs to be
re-extracted. In a Git repository, paths reported by `git status` are also
considered changed. Deleted or excluded paths are removed from the index.

## 3. Language Extraction

Each accepted file is decoded as UTF-8 and passed to its extractor. The
extractor returns `Unit` objects with metadata and source coordinates. For the
example, the `validate_token` unit is conceptually:

```json
{
  "file_id": 1,
  "kind": "symbol",
  "unit_type": "function",
  "name": "validate_token",
  "qualname": "validate_token",
  "signature": "def validate_token(token: str) -> bool:",
  "summary": "Validate an access token.",
  "concepts": "token, str",
  "relationships": "",
  "start_line": 1,
  "end_line": 3,
  "start_col": 0,
  "end_col": 22,
  "byte_start": 0,
  "byte_end": 98
}
```

The exact columns and byte offsets depend on the source text. The important
parts are the searchable metadata and the pointer to the source span. The
source body itself is not copied into the `units` table.

The extractors cover:

- Python functions, classes, methods, imports, docstrings, calls, and aliases.
- TypeScript and JavaScript functions, classes, interfaces, types, enums,
  imports, comments, calls, and aliases.
- Go, Rust, Java, C, C++, and C# language constructs, comments, imports, and
  supported call forms.
- Markdown heading-based document chunks.

## 4. Store Units, Calls, And Aliases

For each file, the indexer:

1. Upserts the file metadata.
2. Deletes the file's previous units and their embeddings.
3. Inserts the newly extracted units.
4. Scans the file for call sites.
5. Maps each call site to its innermost enclosing symbol.
6. Stores call edges and import aliases.

The core relational tables are:

| Table | Contents |
| --- | --- |
| `files` | Tracked repository files and provenance |
| `units` | Functions, classes, imports, document chunks, and source pointers |
| `call_edges` | Caller unit, callee name, full callee chain, and line |
| `import_aliases` | File-local alias and fully qualified target |
| `meta` | Index timestamps and project metadata |

The example creates call-edge rows equivalent to:

```text
caller_unit_id | file_id | callee          | callee_full        | line
---------------+---------+-----------------+--------------------+-----
14             | 2       | validate_token  | validate_token     | 4
15             | 2       | login           | login              | 7
```

For a method call such as `self.validate()`, the stored values are the last
segment `validate` and the full chain `self.validate`. This supports useful
name-based matching without claiming to perform full type resolution.

## 5. Build Retrieval Indexes

Every unit has a compact retrieval key:

```text
qualname or name
signature
summary
concepts
```

Empty fields are omitted and the remaining fields are joined with newlines.
For the example, the retrieval key is approximately:

```text
validate_token
def validate_token(token: str) -> bool:
Validate an access token.
token, str, bool
```

The key is used by both retrieval systems:

### Lexical Index

SQLite FTS5 indexes:

- `name`
- `qualname`
- `signature`
- `summary`
- `concepts`
- `unit_type`
- `file_path`

Lexical queries are made safe for FTS5 by splitting and quoting user terms.
BM25 ranks the matching units.

### Dense Index

The configured embedder creates one vector per unit. Vectors are stored in
the sqlite-vec `vec_units` virtual table with the unit id, language, kind, and
embedding dimensions.

New embeddings are generated in batches of 64. The local default is a 384
dimensional FastEmbed/ONNX model. HTTP-compatible providers use the same
storage path after returning vectors.

### SQLite Runtime

The project index is a single `.urag/index.db` file. urag loads sqlite-vec,
enables foreign keys, and uses SQLite WAL mode. FTS5 triggers keep the lexical
index synchronized when units are inserted, updated, or deleted.

## 6. Search From An Agent

An agent can call the MCP `search` tool like this:

```json
{
  "query": "where is validate_token defined",
  "top_k": 3,
  "mode": "hybrid",
  "language": "python",
  "include_evidence": false
}
```

The query classifier identifies this as a local repository question. It
selects a 2,000-token evidence budget and a default top-k of 5 unless the
caller overrides `top_k`.

The MCP tool returns a JSON string with this shape:

```json
{
  "query": "where is validate_token defined",
  "mode": "hybrid",
  "class": "local",
  "budget_tokens": 2000,
  "count": 1,
  "results": [
    {
      "id": 12,
      "name": "validate_token",
      "qualname": "validate_token",
      "type": "function",
      "signature": "def validate_token(token: str) -> bool:",
      "summary": "Validate an access token.",
      "file": "auth.py",
      "lines": [1, 3],
      "score": 0.0323,
      "ranks": {
        "lexical": 1,
        "dense": 2
      },
      "commit": "a1b2c3d4e5f6789012345678901234567890abcd",
      "stale": false
    }
  ]
}
```

The packet gives the agent enough to identify the definition and cite
`auth.py:1-3`. It does not include the complete function body. The id is the
handle for the next step.

The packet fields mean:

| Field | Meaning |
| --- | --- |
| `id` | Unit id used by `fetch_unit` or `urag get` |
| `name`, `qualname` | Simple and qualified symbol names |
| `type` | Function, method, class, import, or document chunk type |
| `signature` | Compact declaration text |
| `summary` | Docstring, comment, or document summary |
| `file`, `lines` | Repository location and source span |
| `score` | Fused retrieval score or call-graph score |
| `ranks` | Lexical and dense rank, when applicable |
| `commit` | Revision used when the file was indexed |
| `stale` | Whether the file differs from the indexed revision |

For a normal search, evidence is absent by design. This is the main token
efficiency boundary: the agent receives retrieval metadata first, not whole
source files.

## 7. Fetch Exact Evidence

After selecting a result, the agent calls:

```json
{
  "unit_id": 12
}
```

The MCP `fetch_unit` response is shaped like this:

```json
{
  "unit_id": 12,
  "file": "auth.py",
  "lines": [1, 3],
  "language": "py",
  "commit": "a1b2c3d4e5f6789012345678901234567890abcd",
  "span": "def validate_token(token: str) -> bool:\n    \"\"\"Validate an access token.\"\"\"\n    return token != \"\"",
  "stale": false
}
```

The database uses the stored file path and line range, then reads the current
file from disk. That means the returned span is current source, while
`commit` identifies the revision captured by the index. In a Git project,
`stale: true` means the file differs from that revision.

The CLI provides the same operation:

```bash
urag get 12 --root /path/to/demo-project --json
```

When search evidence is requested directly, urag trims the span to the query's
budget. The full span remains available through `fetch_unit` or `urag get`.

## 8. Ask An Impact Question

For a direct caller query, the agent calls:

```json
{
  "name": "validate_token",
  "limit": 20,
  "depth": 1
}
```

The response uses the same compact packet style, with call-specific fields:

```json
{
  "query": "validate_token",
  "mode": "calls",
  "depth": 1,
  "count": 1,
  "results": [
    {
      "id": 14,
      "name": "login",
      "qualname": "login",
      "type": "function",
      "signature": "def login(token: str) -> bool:",
      "summary": "",
      "file": "app.py",
      "lines": [3, 4],
      "score": 1.0,
      "ranks": {
        "lexical": null,
        "dense": null
      },
      "commit": "a1b2c3d4e5f6789012345678901234567890abcd",
      "stale": false,
      "calls": "validate_token",
      "call_line": 4
    }
  ]
}
```

`calls` identifies the recorded callee and `call_line` identifies the call
site. The agent can then call `fetch_unit(14)` to inspect the complete `login`
implementation.

For callers-of-callers, set `depth` above 1:

```json
{
  "name": "validate_token",
  "limit": 20,
  "depth": 2
}
```

The second result is representative of the additional hop:

```json
{
  "id": 15,
  "name": "handle_request",
  "qualname": "handle_request",
  "type": "function",
  "signature": "def handle_request(token: str) -> bool:",
  "summary": "",
  "file": "app.py",
  "lines": [6, 7],
  "score": 1.0,
  "ranks": {
    "lexical": null,
    "dense": null
  },
  "commit": "a1b2c3d4e5f6789012345678901234567890abcd",
  "stale": false,
  "calls": "login",
  "call_line": 7,
  "hop": 2
}
```

The traversal is breadth-first, cycle-safe, and reports the shortest hop for a
unit. It is a static call graph, so it does not guarantee runtime-complete
dispatch or reflection analysis.

## 9. Alias-Aware Queries

The example imports `validate_token` without writing the module path at the
call site. The index stores the binding:

```text
alias:  validate_token
target: auth.validate_token
```

Therefore this fully qualified query can find `login`:

```bash
urag callers auth.validate_token --root /path/to/demo-project --json
```

An alias-resolved packet can include:

```json
{
  "calls": "validate_token",
  "call_line": 4,
  "resolved_to": "auth.validate_token"
}
```

Alias resolution is implemented for Python, TypeScript, Go, Rust, and C#.
Local symbols can shadow aliases, and dynamic or reflective bindings remain
outside the static analysis model.

## 10. Update The Index

Suppose `auth.py` changes. An agent or developer can run:

```bash
urag index --root /path/to/demo-project
```

The indexer discovers the file, compares its metadata with the `files` row,
re-extracts only the changed file, replaces its units and call edges, embeds
new retrieval keys, and leaves unchanged files alone.

The MCP equivalent is:

```json
{}
```

sent to `index_now`, which returns:

```json
{
  "files": 2,
  "changed": 1,
  "deleted": 0,
  "units": 4,
  "embedded": 4,
  "last_indexed": "2026-08-05T23:00:00+00:00"
}
```

If a query runs before re-indexing, Git-aware search may mark the old result
as stale. Run `index_now` or `urag index` before using it as current evidence.

## What The Agent Actually Receives

The normal agent interaction is intentionally two-stage:

```text
Question
   |
   v
search or callers
   |
   +--> compact metadata, file, lines, scores, provenance
   |
   v
fetch_unit for selected ids
   |
   +--> exact current source spans
   |
   v
Agent answer with file:line evidence
```

The agent does not receive:

- Every file in the repository.
- Full source bodies in normal search packets.
- A generated natural-language answer from urag.
- A guaranteed complete runtime dependency graph.

The agent does receive enough information to choose evidence deliberately:
symbol identity, declaration metadata, summaries, repository locations,
retrieval ranks, Git provenance, stale state, and call-site details.

## Other MCP Responses

`status` returns index health and configuration:

```json
{
  "root": "/workspace/demo-project",
  "files": 2,
  "units": 4,
  "embedded": 4,
  "by_language": {
    "python": 2
  },
  "last_indexed": "2026-08-05T23:00:00+00:00",
  "provider": "local",
  "model": "BAAI/bge-small-en-v1.5"
}
```

`init_project` performs the first full index and returns initialization and
index statistics:

```json
{
  "initialized": true,
  "root": "/workspace/demo-project",
  "files": 2,
  "changed": 2,
  "deleted": 0,
  "units": 4,
  "embedded": 4
}
```

## Implementation Map

| File | Responsibility |
| --- | --- |
| `src/urag/config.py` | Project discovery, configuration, languages, and exclusions |
| `src/urag/extractors/` | Language and Markdown unit, call, and alias extraction |
| `src/urag/indexer.py` | Discovery, incremental indexing, call mapping, and embeddings |
| `src/urag/db.py` | SQLite schema, FTS5, sqlite-vec, calls, aliases, and evidence |
| `src/urag/classify.py` | Rule-based query class and budget selection |
| `src/urag/retrieve.py` | Lexical, dense, hybrid, call-graph, and evidence retrieval |
| `src/urag/mcp_server.py` | Agent-facing MCP tools and JSON packets |
| `src/urag/cli.py` | Human-facing command-line interface |

The short version is: urag indexes structure and relationships once, retrieves
small searchable records quickly, and makes exact source a deliberate second
step.
