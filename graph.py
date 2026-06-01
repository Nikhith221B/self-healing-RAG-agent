from __future__ import annotations

import json
import operator
import sys
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import settings
from llm import critique_answer, generate_answer, rewrite_query
from retriever import RetrievalError, retrieve_context

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents to answer this reliably."
)


class GraphState(TypedDict):
    original_query: str
    current_query: str
    query_history: Annotated[list[str], operator.add]
    retrieved_context: list[dict]
    source_documents: list[dict]
    generated_answer: str
    critic_feedback: dict
    critic_history: Annotated[list[dict], operator.add]
    retry_count: int
    max_retries: int
    top_k: int
    final_answer: str
    final_status: str


def _context_to_sources(context: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for chunk in context:
        sources.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("content") or chunk.get("text", ""),
            }
        )
    return sources


def retrieve_context_node(state: GraphState) -> dict[str, Any]:
    try:
        context = retrieve_context(state["current_query"], top_k=state["top_k"])
    except RetrievalError as exc:
        raise RuntimeError(str(exc)) from exc
    return {
        "retrieved_context": context,
        "source_documents": _context_to_sources(context),
    }


def generate_answer_node(state: GraphState) -> dict[str, Any]:
    answer = generate_answer(state["current_query"], state["retrieved_context"])
    return {"generated_answer": answer}


def critic_check_node(state: GraphState) -> dict[str, Any]:
    feedback = critique_answer(
        state["current_query"],
        state["retrieved_context"],
        state["generated_answer"],
    )
    update: dict[str, Any] = {"critic_feedback": feedback}
    if feedback.get("decision") != "accept":
        update["critic_history"] = [feedback]
    return update


def rewrite_query_node(state: GraphState) -> dict[str, Any]:
    new_query = rewrite_query(
        original_query=state["original_query"],
        current_query=state["current_query"],
        critic_feedback=state["critic_feedback"],
        context=state["retrieved_context"],
    )
    return {
        "current_query": new_query,
        "query_history": [new_query],
        "retry_count": state["retry_count"] + 1,
    }


def final_answer_node(state: GraphState) -> dict[str, Any]:
    return {
        "final_answer": state["generated_answer"],
        "final_status": "accepted",
    }


def refusal_answer_node(state: GraphState) -> dict[str, Any]:
    return {
        "final_answer": REFUSAL_MESSAGE,
        "final_status": "refused",
    }


def route_after_critic(
    state: GraphState,
) -> Literal["final_answer", "rewrite_query", "refusal_answer"]:
    decision = state["critic_feedback"].get("decision", "retry")

    if decision == "accept":
        return "final_answer"
    if decision == "refuse":
        return "refusal_answer"
    if state["retry_count"] >= state["max_retries"]:
        return "refusal_answer"
    return "rewrite_query"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("critic_check", critic_check_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("final_answer", final_answer_node)
    workflow.add_node("refusal_answer", refusal_answer_node)

    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", "critic_check")
    workflow.add_conditional_edges(
        "critic_check",
        route_after_critic,
        {
            "final_answer": "final_answer",
            "rewrite_query": "rewrite_query",
            "refusal_answer": "refusal_answer",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve_context")
    workflow.add_edge("final_answer", END)
    workflow.add_edge("refusal_answer", END)

    return workflow.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_self_healing_rag(
    question: str,
    top_k: int | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """
    Run the self-healing RAG workflow and return the final response payload.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    k = top_k if top_k is not None else settings.default_top_k
    retries = max_retries if max_retries is not None else settings.max_retries

    initial_state: GraphState = {
        "original_query": question.strip(),
        "current_query": question.strip(),
        "query_history": [question.strip()],
        "retrieved_context": [],
        "source_documents": [],
        "generated_answer": "",
        "critic_feedback": {},
        "critic_history": [],
        "retry_count": 0,
        "max_retries": retries,
        "top_k": k,
        "final_answer": "",
        "final_status": "",
    }

    final_state = _get_graph().invoke(initial_state)

    return {
        "answer": final_state["final_answer"],
        "final_status": final_state["final_status"],
        "retry_count": final_state["retry_count"],
        "original_query": final_state["original_query"],
        "query_history": final_state["query_history"],
        "critic_feedback": final_state["critic_feedback"],
        "critic_history": final_state["critic_history"],
        "sources": final_state["source_documents"],
    }


if __name__ == "__main__":
    sample_question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "How many paid leaves are available for full-time employees?"
    )

    try:
        result = run_self_healing_rag(sample_question)
        print(json.dumps(result, indent=2))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Graph execution failed: {exc}", file=sys.stderr)
        sys.exit(1)
