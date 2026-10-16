from pathlib import Path

from app.pipeline import ingest_repository
from .embedder import Embedder
from .vector_store import VectorStore


def index_repository(
    repo_url: str,
    repo_path: Path
):
    # 1. Ingest repository
    result = ingest_repository(
        repo_url,
        repo_path
    )

    chunks = result["chunks"]

    if not chunks:
        raise ValueError(
            "No chunks were generated from the repository."
        )

    # 2. Create embeddings
    embedder = Embedder()

    vectors = embedder.embed_chunks(chunks)

    # 3. Create vector store
    dimension = len(vectors[0])

    store = VectorStore(
        dimension=dimension
    )

    # 4. Store vectors + metadata
    store.add(
        vectors,
        chunks
    )

    return store, result