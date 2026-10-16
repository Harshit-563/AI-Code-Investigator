
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from settings import Settings


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.score_threshold = Settings.from_env().semantic_min_score

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float | None = None,
        language: str | None = None,
        chunk_type: str | None = None
    ):
        query_vector = self.embedder.embed(query)
        threshold = self.score_threshold if score_threshold is None else score_threshold

        results = self.vector_store.search(
            query_vector,
            k=k
        )

        filtered_results = []

        for result in results:

            if result["score"] < threshold:
                continue

            chunk = result["chunk"]

            if language is not None:
                if chunk.language.lower() != language.lower():
                    continue

            if chunk_type is not None:
                if chunk.chunk_type.lower() != chunk_type.lower():
                    continue

            filtered_results.append(result)

        return filtered_results


