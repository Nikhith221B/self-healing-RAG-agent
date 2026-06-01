from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

from config import settings


class RetrievalError(Exception):
    """Raised when retrieval cannot proceed."""


def _get_query_embedding_function() -> GoogleGeminiEmbeddingFunction:
    if not settings.gemini_api_key:
        raise RetrievalError(
            "GEMINI_API_KEY is required for retrieval. "
            "Set it in your .env file before querying."
        )
    return GoogleGeminiEmbeddingFunction(
        task_type="RETRIEVAL_QUERY",
        api_key_env_var="GEMINI_API_KEY",
    )


def _get_collection():
    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        return None

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        return client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=_get_query_embedding_function(),
        )
    except Exception:
        return None


def _distance_to_similarity(distance: float | None) -> float | None:
    if distance is None:
        return None
    # Chroma cosine distance is typically 1 - cosine_similarity for normalized vectors.
    return max(0.0, min(1.0, 1.0 - float(distance)))


def retrieve_context(query: str, top_k: int | None = None) -> list[dict]:
    """
    Query the ChromaDB vectorstore and return top-k relevant chunks.

    Returns an empty list if the vectorstore is missing or has no documents.
    """
    if not query.strip():
        raise RetrievalError("Query must not be empty.")

    k = top_k if top_k is not None else settings.default_top_k
    if k < 1:
        raise RetrievalError("top_k must be at least 1.")

    collection = _get_collection()
    if collection is None:
        return []

    doc_count = collection.count()
    if doc_count == 0:
        return []

    results = collection.query(
        query_texts=[query.strip()],
        n_results=min(k, doc_count),
        include=["documents", "metadatas", "distances"],
    )

    ids_batch = results.get("ids") or []
    if not ids_batch or not ids_batch[0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = (results.get("distances") or [[]])[0]

    chunks: list[dict] = []
    for index, chunk_id in enumerate(ids_batch[0]):
        metadata = metadatas[index] or {}
        content = documents[index] or ""
        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            chunk_index = int(chunk_index)

        chunks.append(
            {
                "chunk_id": metadata.get("chunk_id", chunk_id),
                "content": content,
                "source": metadata.get("source", ""),
                "chunk_index": chunk_index,
                "file_type": metadata.get("file_type"),
                "similarity_score": _distance_to_similarity(
                    distances[index] if index < len(distances) else None
                ),
            }
        )

    return chunks


if __name__ == "__main__":
    import json
    import sys

    sample_query = "How many paid leaves are available for full-time employees?"
    query = sys.argv[1] if len(sys.argv) > 1 else sample_query
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else settings.default_top_k

    try:
        results = retrieve_context(query, top_k=top_k)
        if not results:
            print(
                "No results. Run ingestion first: python ingest.py",
                file=sys.stderr,
            )
            sys.exit(1)
        print(json.dumps(results, indent=2))
    except RetrievalError as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        sys.exit(1)
