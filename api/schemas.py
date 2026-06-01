from __future__ import annotations

from pydantic import BaseModel, Field

from config import settings


class HealthResponse(BaseModel):
    status: str
    service: str


class CriticFeedback(BaseModel):
    grounded: bool = False
    score: float = 0.0
    decision: str = "retry"
    supported_chunks: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    chunk_index: int | None = None
    text: str


class IngestResponse(BaseModel):
    status: str
    processed_files: int
    chunks_created: int
    vectorstore_path: str


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default_factory=lambda: settings.default_top_k, ge=1, le=20)
    max_retries: int = Field(
        default_factory=lambda: settings.max_retries,
        ge=0,
        le=5,
    )


class AskResponse(BaseModel):
    run_id: str
    answer: str
    final_status: str
    retry_count: int
    original_query: str
    query_history: list[str]
    critic_feedback: CriticFeedback
    critic_history: list[CriticFeedback] = Field(default_factory=list)
    sources: list[SourceChunk] = Field(default_factory=list)
    top_k_used: int | None = None
    bandit_arm: str | None = None


class FeedbackRequest(BaseModel):
    run_id: str
    rating: str = Field(description="helpful or not_helpful")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    run_id: str
    message: str
