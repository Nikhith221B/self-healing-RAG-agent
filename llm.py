from __future__ import annotations

import json
import re
from typing import Any

from google import genai

from config import settings
from prompts import CRITIC_PROMPT, GENERATOR_PROMPT, QUERY_REWRITER_PROMPT

VALID_DECISIONS = frozenset({"accept", "retry", "refuse"})
_client: genai.Client | None = None


class LLMError(Exception):
    """Raised when the LLM client cannot run."""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise LLMError(
                "GEMINI_API_KEY is required. Set it in your .env file before calling the LLM."
            )
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _generate(prompt: str) -> str:
    response = _get_client().models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise LLMError("Gemini returned an empty response.")
    return text.strip()


def _format_context(context: list[dict]) -> str:
    if not context:
        return "No context provided."

    parts: list[str] = []
    for index, chunk in enumerate(context):
        chunk_id = chunk.get("chunk_id", f"chunk_{index}")
        source = chunk.get("source", "unknown")
        content = chunk.get("content") or chunk.get("text", "")
        parts.append(f"[{chunk_id}] (source: {source})\n{content}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _critic_fallback(reason: str) -> dict[str, Any]:
    return {
        "grounded": False,
        "score": 0.0,
        "decision": "retry",
        "supported_chunks": [],
        "unsupported_claims": [],
        "reason": reason,
    }


def _normalize_critic_payload(data: dict[str, Any]) -> dict[str, Any]:
    decision = str(data.get("decision", "retry")).lower().strip()
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Invalid critic decision: {decision}")

    score_raw = data.get("score", 0.0)
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        raise ValueError("Critic score must be a number.") from None
    score = max(0.0, min(1.0, score))

    grounded = data.get("grounded", False)
    if not isinstance(grounded, bool):
        grounded = str(grounded).lower() in {"true", "1", "yes"}

    supported_chunks = data.get("supported_chunks", [])
    if not isinstance(supported_chunks, list):
        supported_chunks = []
    supported_chunks = [str(item) for item in supported_chunks]

    unsupported_claims = data.get("unsupported_claims", [])
    if not isinstance(unsupported_claims, list):
        unsupported_claims = []
    unsupported_claims = [str(item) for item in unsupported_claims]

    reason = str(data.get("reason", "")).strip() or "No reason provided."

    return {
        "grounded": grounded,
        "score": score,
        "decision": decision,
        "supported_chunks": supported_chunks,
        "unsupported_claims": unsupported_claims,
        "reason": reason,
    }


def _parse_critic_response(raw: str) -> dict[str, Any]:
    data = _extract_json(raw)
    if data is None:
        return _critic_fallback("Failed to parse critic JSON; defaulting to retry.")
    try:
        return _normalize_critic_payload(data)
    except ValueError as exc:
        return _critic_fallback(f"Invalid critic payload: {exc}")


def generate_answer(query: str, context: list[dict]) -> str:
    prompt = GENERATOR_PROMPT.format(
        query=query.strip(),
        context=_format_context(context),
    )
    return _generate(prompt)


def critique_answer(query: str, context: list[dict], answer: str) -> dict[str, Any]:
    prompt = CRITIC_PROMPT.format(
        query=query.strip(),
        context=_format_context(context),
        answer=answer.strip(),
    )
    raw = _generate(prompt)
    return _parse_critic_response(raw)


def rewrite_query(
    original_query: str,
    current_query: str,
    critic_feedback: dict,
    context: list[dict],
) -> str:
    prompt = QUERY_REWRITER_PROMPT.format(
        original_query=original_query.strip(),
        current_query=current_query.strip(),
        critic_feedback=json.dumps(critic_feedback, indent=2),
        context=_format_context(context),
    )
    rewritten = _generate(prompt)
    return rewritten.strip().strip('"').strip("'")


if __name__ == "__main__":
    import sys

    mock_context = [
        {
            "chunk_id": "hr_policy.txt_0",
            "source": "hr_policy.txt",
            "content": (
                "Full-time employees are eligible for 18 paid leaves per year. "
                "Part-time employees receive pro-rated leave based on hours worked."
            ),
            "chunk_index": 0,
        }
    ]
    question = "How many paid leaves are available for full-time employees?"

    try:
        print("=== generate_answer ===")
        answer = generate_answer(question, mock_context)
        print(answer)

        print("\n=== critique_answer ===")
        feedback = critique_answer(question, mock_context, answer)
        print(json.dumps(feedback, indent=2))

        print("\n=== rewrite_query ===")
        new_query = rewrite_query(
            original_query=question,
            current_query=question,
            critic_feedback=feedback,
            context=mock_context,
        )
        print(new_query)
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(
            f"Gemini API error: {exc}\n"
            "Tip: set GEMINI_MODEL in .env (e.g. gemini-2.5-flash) if you hit quota limits.",
            file=sys.stderr,
        )
        sys.exit(1)
