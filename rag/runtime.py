"""Construct the existing retrieval and generation pipeline for app entry points."""

from embeddings.bm25_store import BM25Store
from embeddings.embedder import Embedder
from embeddings.hybrid_retriever import HybridRetriever
from embeddings.reranker import Reranker
from embeddings.retriever import Retriever
from embeddings.vector_store import VectorStore
from rag.generator import Generator
from settings import Settings, configure_logging


def load_runtime(vector_dir: str = "data/vectorstore") -> tuple[HybridRetriever, Generator]:
    """Load persisted chunks and initialize semantic, lexical, rerank, and LLM stages."""
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    store = VectorStore.load(vector_dir)
    if not store.chunks:
        raise ValueError(f"Vector store contains no chunks: {vector_dir}")

    generator = Generator()
    embedder = Embedder(settings.embedding_model)
    semantic_retriever = Retriever(vector_store=store, embedder=embedder)
    bm25_store = BM25Store()
    bm25_store.add(store.chunks)
    reranker = Reranker(settings.reranker_model)
    hybrid = HybridRetriever(
        semantic_retriever,
        bm25_store,
        reranker,
        confidence_threshold=settings.confidence_threshold,
    )
    return hybrid, generator