from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    chroma_persist_dir: str
    chroma_collection_name: str
    log_dir: str
    enable_bandit: bool
    default_top_k: int
    max_retries: int


settings = Settings(
    gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "vectorstore"),
    chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    enable_bandit=_get_bool("ENABLE_BANDIT", False),
    default_top_k=_get_int("DEFAULT_TOP_K", 5),
    max_retries=_get_int("MAX_RETRIES", 2),
)

