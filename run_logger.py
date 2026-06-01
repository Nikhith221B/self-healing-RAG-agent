from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings


def _log_path() -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "runs.jsonl"


def _feedback_path() -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "feedback.jsonl"


def log_ask_run(
    *,
    question: str,
    top_k: int,
    max_retries: int,
    result: dict[str, Any],
    latency_ms: float,
    error: str | None = None,
    bandit_arm: str | None = None,
) -> str:
    """Append one /ask execution record for offline learning analysis."""
    run_id = str(uuid.uuid4())
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "top_k": top_k,
        "bandit_arm": bandit_arm,
        "max_retries": max_retries,
        "latency_ms": round(latency_ms, 2),
        "error": error,
        "final_status": result.get("final_status"),
        "retry_count": result.get("retry_count"),
        "query_history": result.get("query_history"),
        "critic_feedback": result.get("critic_feedback"),
        "critic_history": result.get("critic_history"),
        "answer_preview": (result.get("answer") or "")[:500],
    }
    with _log_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_id


def log_feedback(*, run_id: str, rating: str, comment: str | None = None) -> None:
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "comment": comment or "",
    }
    with _feedback_path().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_exists(run_id: str) -> bool:
    path = _log_path()
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_id") == run_id:
                return True
    return False
