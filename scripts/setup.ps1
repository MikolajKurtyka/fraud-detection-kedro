# One-command project setup for Windows (PowerShell).
#   .\scripts\setup.ps1             # environment + git hooks + credential check
#   .\scripts\setup.ps1 -Download   # ...and download the Kaggle data
param([switch]$Download)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "==> Checking uv" -ForegroundColor Cyan
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found, installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "==> Installing dependencies (uv sync)" -ForegroundColor Cyan
uv sync

Write-Host "==> Installing pre-commit hooks" -ForegroundColor Cyan
if (-not (Test-Path .git)) { git init | Out-Null }
uv run pre-commit install

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example"
}

Write-Host "==> Checking Kaggle credentials and competition access" -ForegroundColor Cyan
uv run python scripts/download_data.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Download) {
    Write-Host "==> Downloading data" -ForegroundColor Cyan
    uv run python scripts/download_data.py
}

Write-Host "`nSetup complete. Next: uv run kedro run --pipelines data_ingestion" -ForegroundColor Green
