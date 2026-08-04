---
name: urag
description: Retrieves project knowledge (symbols, architecture, docs) from the urag structure-aware RAG index. Use when you need to find where a function/class/type is defined, how a component works, what calls or depends on something, or what config/docs say — instead of grepping or reading whole files. Trigger on questions about project code structure, symbol locations, or "how does X work" in a urag-indexed repository.
---

# urag — project knowledge retrieval

Use this skill whenever you need to find project knowledge: where a symbol is
defined, how a component works, what calls something, or what a config or doc
says. urag is a structure-aware RAG index — it answers with small, cited
evidence packets instead of whole files, so it is far more token-efficient
than grepping or reading files blind.

## Setup (one time)

```bash
urag init                       # creates .urag/ config + first index
urag watch                      # optional: keep index fresh continuously
```

If the index is missing (`urag status` fails), run `urag init` first.
If the MCP server is configured, you can also call `init_project` / `index_now`
directly.

## Usage

### Search for knowledge

```bash
urag search "how does authentication work" --top-k 5
urag search "TokenValidator.validate" --mode lexical --top-k 3
urag search "cache invalidation" --mode dense --top-k 3
urag search "RRF" --mode lexical --language python --json
```

Modes:
- `hybrid` (default): lexical + dense fused with RRF. Use for most questions.
- `lexical`: exact identifiers, config keys, error strings, symbol names.
- `dense`: conceptual / natural-language questions.

### Fetch exact source (only for hits you will actually use)

```bash
urag get 190                    # exact source span for unit 190
```

### Status / health

```bash
urag status
urag doctor
```

## Token-efficient workflow (important)

1. Search with `--top-k 3-5` first. Results are compact records.
2. Read `file:lines` from the results; fetch evidence (`urag get <id>` or
   `search --evidence`) **only** for the 1-3 units that matter.
3. Never dump whole files into context — urag exists so you don't have to.
4. Use `--mode lexical` when you know the exact name — it is cheap and precise.
5. Re-run `urag index` (or let `urag watch` run) after big changes to keep
   evidence fresh.
6. Check the `stale` flag on results: if a hit is stale (file changed since
   it was indexed), re-index before relying on it. `urag classify "<query>"`
   shows the budget tier a query will get.

## Output format

Search results include: `qualname`, `type` (function/class/method/interface/
struct/enum/trait/import/doc_chunk), `signature`, `summary`, `file`,
`lines [start, end]`, ranks (`lexical#N dense#N`), plus `commit` and `stale`
provenance when the project is a git repo. With `--json` you get the same as
JSON for scripting, including the query `class` and `budget_tokens`.
