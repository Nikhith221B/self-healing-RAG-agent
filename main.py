from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.schemas import HealthResponse

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(
    title="Self-Healing RAG Agent",
    description="Agentic self-healing RAG with critic-based feedback and retry.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if FRONTEND_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="app")


@app.get("/")
def root():
    if FRONTEND_DIR.is_dir():
        return RedirectResponse(url="/app/")
    return {
        "service": "self-healing-rag-agent",
        "docs": "/docs",
        "health": "/health",
        "ingest": "POST /ingest",
        "ask": "POST /ask",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="self-healing-rag-agent")
