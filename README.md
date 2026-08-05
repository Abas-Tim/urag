# urag — Universal Repository Agent Graph

Structure-aware, token-efficient RAG for software projects. Indexes symbols,
signatures, relationships and docs into a single portable SQLite database,
then answers queries with small, cited evidence packets instead of whole files.

**Key design (per 2026 RAG research):**
- **Retrieval keys ≠ evidence payloads** — compact L0/L1 records (name, signature,
  summary, concepts) are embedded and lexically indexed; exact source spans (L2)
  are loaded lazily on demand.
- **Hybrid retrieval** — FTS5 lexical (finds names/identifiers) + sqlite-vec dense
  (finds concepts), fused with Reciprocal Rank Fusion.
- **Structure-aware units** — tree-sitter extracts functions, methods, classes,
  interfaces, imports with line/byte spans, docstrings and relationships.
- **Incremental + continuous** — mtime-based re-indexing; `watch` daemon keeps
  the index fresh automatically.
- **Model-agnostic embeddings** — local ONNX (offline, no API key) by default,
  or any OpenAI-compatible `/embeddings` endpoint.

## Install

urag is a Python CLI that installs anywhere uv or Python runs (Windows,
macOS, Linux). Requires only the `urag` command plus `git` (for git-aware
provenance); the embedding model (~25 MB) downloads on first use.

**One-liner from the internet (needs uv; auto-installs uv if missing):**

```bash
# macOS / Linux
curl -LsSf https://raw.githubusercontent.com/Abas-Tim/urag/main/bootstrap/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/Abas-Tim/urag/main/bootstrap/install.ps1 | iex
```

**From PyPI (once published) — the standard install:**

```bash
uv tool install urag-cli       # global `urag` on PATH (uv)
pipx install urag-cli          # same, via pipx
# or in a project venv:     uv add urag-cli

urag --version
```

**Zero-install for agent harnesses (MCP)** — `uvx` downloads and caches on
first use, no setup on new machines:

```json
{ "mcpServers": { "urag": { "command": ["uvx", "--from", "urag-cli", "urag", "mcp", "--root", "/path/to/project"] } } }
```

`uvx` supports any install source, so until urag is on PyPI you can point
it at the repo or a wheel: `uvx --from git+https://github.com/Abas-Tim/urag urag mcp --root .`

**From source (developers):**

```bash
git clone https://github.com/Abas-Tim/urag
cd urag && uv sync && uv run urag --help
```

## Usage

```bash
urag init               # set up .urag/ config + .gitignore, initial index
urag watch              # continuously re-index on file changes (Ctrl+C to stop)
urag search "how does auth work"        # hybrid search
urag search "TokenValidator" --mode lexical --top-k 5
urag search "RRF" --mode lexical --json --evidence
urag get 190            # fetch exact source span for a unit
urag status             # index stats
urag doctor             # health check
```

Project config lives in `.urag/urag.toml` (auto-created):

```toml
[embedding]
provider = "local"        # local | http | none
model = "BAAI/bge-small-en-v1.5"
dimension = 384
# http_url = "http://localhost:11434/v1"
# http_api_key = ""
# http_model = "nomic-embed-text"

[index]
languages = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "markdown"]
exclude = [".urag", ".git", "node_modules", "dist", "build", ".venv"]
```

## Architecture

```
sources ──► tree-sitter extractors ──► units (L0/L1 records)
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
               FTS5 (lexical)    vec0 (dense)      files/units tables
                    └──────────────┬──────────────────┘
                                   ▼
                     RRF fusion → ranked evidence packets
                                   ▼
                    L2 source spans loaded on demand (`get`)
```

| Layer | Contents | Size |
|---|---|---|
| L0 | qualname, signature, summary, concepts (retrieval key, embedded) | ~100–300 tokens |
| L1 | relationships, file/line/byte pointers | ~50 tokens |
| L2 | exact source span, loaded on demand | full unit |

## Notes

- Per-project single-file index: `.urag/index.db` (SQLite + WAL). Gitignored.
- Embeddings default to `BAAI/bge-small-en-v1.5` (384 dims, ~25 MB, cached in
  `%LOCALAPPDATA%\urag` or `~/.cache/urag`).
- `tree-sitter` is pinned `<0.26` — 0.26.0 has an unfixed use-after-free
  segfault during repo-scale traversals (py-tree-sitter issue #472).

## Git-aware invalidation

Every indexed file records the commit it was indexed at. Search results and
evidence packets carry `commit` (8-hex) and `stale` flags:

- `stale=true` means the file changed since it was indexed (uncommitted edit,
  branch switch, pull) — treat the evidence as outdated and re-index.
- Re-indexing uses `git status`/`git diff` to find changed files (O(changed)
  instead of stat-ing the whole repo) and works even when file mtimes lie.
- Non-git projects fall back to mtime-based checks automatically.

```bash
urag search "parse_token" --mode lexical --json   # results include commit/stale
urag get 190                                      # span + commit + stale flag
```

## Adaptive query classification

Queries are auto-classified into budget tiers before retrieval — context is a
per-query decision, not a fixed constant:

| Class | top_k | budget | behavior |
|---|---|---|---|
| `symbol` | 3 | 800 tok | exact names/identifiers: lexical-only |
| `local` | 5 | 2,000 tok | single-module implementation questions |
| `debugging` | 8 | 4,000 tok | errors, crashes, cross-module flow |
| `impact` | 10 | 6,000 tok | callers, dependencies, "what breaks if X" |

Evidence spans returned with `--evidence` are trimmed to the class budget
(head + truncation note; the full span stays available via `urag get`).

```bash
urag classify "what calls this method"   # -> impact (top_k=10, 6000 tok)
urag search "why does it crash" --top-k 2
```

## Evaluating retrieval quality (`urag eval`)

Compares urag against the non-urag context baselines on the same questions:

```bash
# auto-generate provable-gold questions (definitions + callers), compare default systems
urag eval --root . --autogen 10 --top-k 5

# hand-written questions.jsonl: {"query": "...", "gold_file": "src/a.py"}
urag eval --root . --questions questions.jsonl

# add the LLM answer-quality tier (OpenAI-compatible endpoint)
urag eval --root . --autogen 10 --judge-url https://api.openai.com/v1 --judge-model gpt-4o-mini --judge-key $KEY

# machine-readable report
urag eval --root . --autogen 10 --json --report eval.json
```

Systems compared per question: `urag-hybrid`, `urag-lexical`, `rg` (grep
baseline), `chunk` (structure-free fixed-size chunk embeddings), and optional
`oracle` (whole gold files). Metrics: **recall@k, MRR, tokens/retrieval,
p50/p95 latency**. Definition and call questions get provable gold from the
index itself (the unit's file / the actual `call_edges`), so evaluation works
without hand labeling; doc/conceptual questions need a hand-written
`--questions` file.

## Call graph (impact analysis)

Every function body is scanned for call sites (python `call`, ts/js/go/rust/c/cpp
`call_expression`, java `method_invocation`, c# `invocation_expression`), stored
as `call_edges(caller_unit, callee, callee_full, line)`. Impact questions
("what calls X", "what breaks if X changes") are answered from the graph
instead of text similarity:

```bash
urag search "what calls index_all"     # -> mode=calls, real callers + line numbers
urag callers fit_evidence              # dedicated command
```

```json
{"query": "what calls index_all", "mode": "calls", "results": [
  {"qualname": "init", "file": "src/urag/cli.py", "lines": [57, 82], "calls": "indexer.index_all", "call_line": 81}
]}
```

Matching normalizes the last segment (`self.validate`, `os.path.exists`,
`TokenValidator::validate` all match `validate`); constructor calls are
included (who instantiates X is an impact question). Alias resolution is
approximate by design — dynamic/reflective calls are invisible to static
analysis.

## MCP server (agent harnesses)

```bash
urag mcp --root /path/to/project      # stdio MCP server
# root also auto-discovers from cwd; set URAG_ROOT env var to override
```

Tools: `search`, `fetch_unit`, `index_now`, `status`, `init_project`.
Results are compact evidence packets (signature + summary + file:line + ranks);
exact source spans are fetched only for the units you need.

### opencode

```json
// opencode.json
{
  "mcp": {
    "urag": {
      "type": "local",
      "command": ["urag", "mcp", "--root", "/path/to/project"],
      "enabled": true
    }
  }
}
```

### Claude Code

```json
// .mcp.json
{
  "mcpServers": {
    "urag": {
      "command": "urag",
      "args": ["mcp", "--root", "/path/to/project"]
    }
  }
}
```

### Cursor / generic

```json
{
  "mcpServers": {
    "urag": {
      "command": "urag",
      "args": ["mcp", "--root", "/path/to/project"],
      "type": "stdio"
    }
  }
}
```

## Skill files (harness guidance)

`skills/urag/SKILL.md` teaches agents the token-efficient workflow (search
small → fetch evidence only for what you use). Install per harness:

```bash
# opencode (global)
cp -r skills/urag ~/.config/opencode/skills/urag

# Claude Code
cp -r skills/urag ~/.claude/skills/urag
```

## Roadmap

- [x] MCP server (`urag mcp`) for Claude Code / opencode / Cursor
- [x] Skill files (`skills/urag/SKILL.md`) for agent harnesses
- [x] Git-aware invalidation (per-commit provenance, stale-evidence flags)
- [x] More languages (go, rust, java, c, c++, c#)
- [x] Adaptive query-classifier → per-query token budgets
- [x] Call-graph pass — impact queries answered from real call edges
- [ ] Multi-hop traversal (callers-of-callers) on demand
- [ ] Import-alias resolution for fully-qualified callee chains
- [ ] More languages (kotlin, swift, ruby, php, zig)
- [ ] Train the query router from production traces
