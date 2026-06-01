from __future__ import annotations

from pathlib import Path

import chromadb
import pdfplumber
from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

from config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class IngestionError(Exception):
    """Raised when ingestion cannot proceed."""


def _load_pdf(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages).strip()


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_document(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path), "pdf"
    if suffix in {".txt", ".md", ".markdown"}:
        file_type = "markdown" if suffix in {".md", ".markdown"} else "txt"
        return _load_text_file(path), file_type
    raise IngestionError(
        f"Unsupported file type '{suffix}' for {path.name}. "
        f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def _chunk_text(text: str, source_filename: str, file_type: str) -> list[dict]:
    if not text:
        return []

    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end]
        chunk_id = f"{source_filename}_{chunk_index}"
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_filename,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "file_type": file_type,
                },
            }
        )
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        chunk_index += 1

    return chunks


def _get_embedding_function() -> GoogleGeminiEmbeddingFunction:
    if not settings.gemini_api_key:
        raise IngestionError(
            "GEMINI_API_KEY is required to generate embeddings. "
            "Set it in your .env file before running ingestion."
        )
    return GoogleGeminiEmbeddingFunction(
        task_type="RETRIEVAL_DOCUMENT",
        api_key_env_var="GEMINI_API_KEY",
    )


def _get_collection():
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=_get_embedding_function(),
    )


def ingest_documents(input_dir: str = "data/sample_docs") -> dict:
    """
    Load supported documents, chunk them, embed, and store in ChromaDB.

    Returns a summary with processed_files, chunks_created, and vectorstore_path.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise IngestionError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise IngestionError(
            f"No supported documents found in {input_dir}. "
            f"Add PDF, TXT, or Markdown files."
        )

    collection = _get_collection()
    processed_files = 0
    chunks_created = 0

    for file_path in files:
        try:
            text, file_type = _load_document(file_path)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Failed to read {file_path.name}: {exc}") from exc

        if not text:
            raise IngestionError(f"Document is empty: {file_path.name}")

        doc_chunks = _chunk_text(text, file_path.name, file_type)
        if not doc_chunks:
            raise IngestionError(f"No chunks created for {file_path.name}")

        existing = collection.get(where={"source": file_path.name})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        metadatas = []
        for chunk in doc_chunks:
            meta = dict(chunk["metadata"])
            # Chroma metadata values must be str, int, float, or bool.
            meta["chunk_index"] = int(meta["chunk_index"])
            metadatas.append(meta)

        collection.upsert(
            ids=[chunk["id"] for chunk in doc_chunks],
            documents=[chunk["text"] for chunk in doc_chunks],
            metadatas=metadatas,
        )

        processed_files += 1
        chunks_created += len(doc_chunks)

    return {
        "status": "success",
        "processed_files": processed_files,
        "chunks_created": chunks_created,
        "vectorstore_path": settings.chroma_persist_dir,
    }


if __name__ == "__main__":
    import json
    import sys

    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs"
    try:
        summary = ingest_documents(target_dir)
        print(json.dumps(summary, indent=2))
    except IngestionError as exc:
        print(f"Ingestion error: {exc}", file=sys.stderr)
        sys.exit(1)
