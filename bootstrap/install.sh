#!/usr/bin/env sh
# urag installer for macOS / Linux (POSIX sh).
#
# Usage:
#   curl -LsSf https://Abas-Tim.github.io/urag/install.sh | sh
#   # or from the repo:
#   curl -LsSf https://raw.githubusercontent.com/Abas-Tim/urag/main/bootstrap/install.sh | sh
#
# Env overrides:
#   URAG_PACKAGE  package spec to install (default: urag -> PyPI)
#                 e.g. "urag==0.1.0" or a git URL / wheel path
#   URAG_BIN      install directory (default: uv tool bin dir, i.e. ~/.local/bin)
set -e

URAG_PACKAGE="${URAG_PACKAGE:-urag}"

say() { printf '%s\n' "$*"; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  say "uv not found - installing it (https://astral.sh/uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

main() {
  ensure_uv
  say "installing urag ($URAG_PACKAGE) via uv tool..."
  uv tool install "$URAG_PACKAGE"
  if ! command -v urag >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
  say ""
  say "urag installed: $(urag --version)"
  say ""
  say "Next steps:"
  say "  1. urag init --root /path/to/project   # create .urag/ + first index"
  say "  2. urag watch --root /path/to/project  # keep it fresh (optional)"
  say "  3. urag search 'how does auth work' --root /path/to/project"
  say "  MCP: add {\"urag\": {\"command\": [\"urag\", \"mcp\", \"--root\", \"/path/to/project\"]}} to your agent harness"
}

main "$@"
