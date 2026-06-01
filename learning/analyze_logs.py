from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from config import settings


def analyze(log_file: Path | None = None) -> dict:
    path = log_file or Path(settings.log_dir) / "runs.jsonl"
    if not path.exists():
        return {"total_runs": 0, "message": f"No log file at {path}"}

    statuses: Counter[str] = Counter()
    retries: list[int] = []
    scores: list[float] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            statuses[record.get("final_status") or "unknown"] += 1
            retries.append(int(record.get("retry_count") or 0))
            feedback = record.get("critic_feedback") or {}
            if isinstance(feedback, dict) and feedback.get("score") is not None:
                scores.append(float(feedback["score"]))

    feedback_counts: Counter[str] = Counter()
    feedback_path = path.parent / "feedback.jsonl"
    if feedback_path.exists():
        with feedback_path.open(encoding="utf-8") as fb:
            for line in fb:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                feedback_counts[entry.get("rating", "unknown")] += 1

    total = sum(statuses.values())
    report = {
        "total_runs": total,
        "status_counts": dict(statuses),
        "avg_retry_count": round(sum(retries) / len(retries), 2) if retries else 0,
        "avg_critic_score": round(sum(scores) / len(scores), 3) if scores else None,
        "accept_rate": round(statuses.get("accepted", 0) / total, 3) if total else 0,
        "refuse_rate": round(statuses.get("refused", 0) / total, 3) if total else 0,
        "feedback_counts": dict(feedback_counts),
    }
    try:
        from learning.bandit import summary as bandit_summary

        report["bandit"] = bandit_summary()
    except Exception:
        report["bandit"] = {}
    return report


if __name__ == "__main__":
    summary = analyze()
    print(json.dumps(summary, indent=2))
    if summary.get("total_runs", 0) == 0:
        sys.exit(1)
