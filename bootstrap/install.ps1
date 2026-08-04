# urag installer for Windows (PowerShell 5.1+).
#
# Usage:
#   irm https://YOUR-ORG.github.io/urag/install.ps1 | iex
#   # or from the repo:
#   irm https://raw.githubusercontent.com/YOUR-ORG/urag/main/bootstrap/install.ps1 | iex
#
# Env overrides:
#   URAG_PACKAGE  package spec to install (default: urag -> PyPI)
#                 e.g. "urag==0.1.0" or a git URL / wheel path
$ErrorActionPreference = "Stop"
$Package = if ($env:URAG_PACKAGE) { $env:URAG_PACKAGE } else { "urag" }

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) { return }
    Write-Host "uv not found - installing it (https://astral.sh/uv)..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
}

function Main {
    Ensure-Uv
    Write-Host "installing urag ($Package) via uv tool..."
    uv tool install $Package
    if (-not (Get-Command urag -ErrorAction SilentlyContinue)) {
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
    }
    Write-Host ""
    Write-Host "urag installed: $(urag --version)"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Green
    Write-Host "  1. urag init --root J:\path\to\project"
    Write-Host "  2. urag watch --root J:\path\to\project  (optional)"
    Write-Host "  3. urag search 'how does auth work' --root J:\path\to\project"
    Write-Host "  MCP: add `"urag`" = { command: ['urag', 'mcp'] } to your agent harness config"
}

Main
