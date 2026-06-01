# Start the Self-Healing RAG API (use python -m uvicorn — works even if uvicorn.exe has a stale path).
Set-Location $PSScriptRoot\..

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Create .venv first: python -m venv .venv"
    exit 1
}

Write-Host "Starting http://127.0.0.1:8000/app/"
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
