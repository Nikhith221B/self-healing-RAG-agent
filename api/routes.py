from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    AskRequest,
    AskResponse,
    CriticFeedback,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    SourceChunk,
)
from config import settings
from graph import run_self_healing_rag
from ingest import IngestionError, ingest_documents
from learning import bandit
from run_logger import log_ask_run, log_feedback, run_exists

router = APIRouter()

UPLOAD_DIR = Path("data/sample_docs")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
UNSUPPORTED_FILE_MESSAGE = (
    "Unsupported file type. Supported types are PDF, TXT, MD, and Markdown."
)
VALID_RATINGS = frozenset({"helpful", "not_helpful"})


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] | None = File(default=None),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for upload in files:
        filename = Path(upload.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a name.")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=UNSUPPORTED_FILE_MESSAGE)

        content = await upload.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"Uploaded file is empty: {filename}",
            )

        destination = UPLOAD_DIR / filename
        destination.write_bytes(content)

    try:
        summary = ingest_documents(str(UPLOAD_DIR))
    except IngestionError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return IngestResponse(**summary)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    bandit_arm: str | None = None
    top_k_used = request.top_k
    if settings.enable_bandit:
        bandit_arm, top_k_used = bandit.select_top_k()

    started = time.perf_counter()
    try:
        result = run_self_healing_rag(
            question=request.question,
            top_k=top_k_used,
            max_retries=request.max_retries,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log_ask_run(
            question=request.question,
            top_k=top_k_used,
            max_retries=request.max_retries,
            result={},
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
            bandit_arm=bandit_arm,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Graph execution failed: {exc}",
        ) from exc

    if settings.enable_bandit and bandit_arm:
        bandit.update(bandit_arm, bandit.reward_from_result(result))

    run_id = log_ask_run(
        question=request.question,
        top_k=top_k_used,
        max_retries=request.max_retries,
        result=result,
        latency_ms=(time.perf_counter() - started) * 1000,
        bandit_arm=bandit_arm,
    )

    critic_feedback = CriticFeedback.model_validate(
        result.get("critic_feedback") or {}
    )
    critic_history = [
        CriticFeedback.model_validate(item)
        for item in result.get("critic_history") or []
    ]
    sources = [SourceChunk.model_validate(item) for item in result.get("sources") or []]

    return AskResponse(
        run_id=run_id,
        answer=result["answer"],
        final_status=result["final_status"],
        retry_count=result["retry_count"],
        original_query=result["original_query"],
        query_history=result["query_history"],
        critic_feedback=critic_feedback,
        critic_history=critic_history,
        sources=sources,
        top_k_used=top_k_used,
        bandit_arm=bandit_arm,
    )


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    rating = request.rating.strip().lower()
    if rating not in VALID_RATINGS:
        raise HTTPException(
            status_code=400,
            detail="rating must be 'helpful' or 'not_helpful'",
        )
    if not request.run_id.strip():
        raise HTTPException(status_code=400, detail="run_id is required")
    if not run_exists(request.run_id.strip()):
        raise HTTPException(status_code=404, detail="run_id not found in logs")

    log_feedback(
        run_id=request.run_id.strip(),
        rating=rating,
        comment=request.comment,
    )
    return FeedbackResponse(
        status="ok",
        run_id=request.run_id.strip(),
        message="Feedback recorded for offline learning.",
    )
