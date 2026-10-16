import logging
from collections.abc import Mapping
from typing import Any

from embeddings.retriever import Retriever
from embeddings.bm25_store import BM25Store
from embeddings.reranker import Reranker
from settings import Settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuse semantic and lexical candidates before cross-encoder reranking."""

    def __init__(
        self,
        semantic_retriever: Retriever,
        bm25_store: BM25Store,
        reranker: Reranker,
        confidence_threshold: float | None = None,
    ):
        self.semantic_retriever = semantic_retriever
        self.bm25_store = bm25_store
        self.reranker = reranker
        settings = Settings.from_env()
        self.candidate_k = settings.retrieval_candidates
        self.rrf_k = settings.rrf_k
        self.semantic_weight = settings.semantic_weight
        self.keyword_weight = settings.keyword_weight
        self.confidence_threshold = (
            settings.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        candidate_k: int | None = None,
        rrf_k: int | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        candidate_k = self.candidate_k if candidate_k is None else candidate_k
        rrf_k = self.rrf_k if rrf_k is None else rrf_k
        semantic_weight = self.semantic_weight if semantic_weight is None else semantic_weight
        keyword_weight = self.keyword_weight if keyword_weight is None else keyword_weight
        if k <= 0 or candidate_k <= 0:
            return []

        # 1. Semantic retrieval
        semantic_results = self.semantic_retriever.retrieve(
            query,
            k=candidate_k
        )

        # 2. Keyword retrieval
        keyword_results = self.bm25_store.search(
            query,
            k=candidate_k
        )
        self._validate_results(semantic_results, "semantic")
        self._validate_results(keyword_results, "BM25")
        logger.info(
            "Retrieval candidates semantic=%d bm25=%d",
            len(semantic_results),
            len(keyword_results),
        )

        scores = {}
        chunks = {}
        semantic_scores = {}
        keyword_scores = {}

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):
            chunk = result["chunk"]
            key = self._chunk_key(chunk)

            scores[key] = scores.get(key, 0) + (
                semantic_weight / (rrf_k + rank)
            )
            chunks[key] = chunk
            semantic_scores[key] = max(semantic_scores.get(key, float("-inf")), result["score"])

        for rank, result in enumerate(
            keyword_results,
            start=1
        ):
            chunk = result["chunk"]
            key = self._chunk_key(chunk)

            scores[key] = scores.get(key, 0) + (
                keyword_weight / (rrf_k + rank)
            )
            chunks[key] = chunk
            keyword_scores[key] = max(keyword_scores.get(key, float("-inf")), result["score"])

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        candidates = [
            {
                "score": score,
                "rrf_score": score,
                "semantic_score": semantic_scores.get(key, 0.0),
                "bm25_score": keyword_scores.get(key, 0.0),
                "chunk": chunks[key],
            }
            for key, score in ranked[:candidate_k]
        ]
        # Reranking receives the fused candidate list, not either raw list.
        reranked = self.reranker.rerank(
            query,
            candidates,
            k=k
        )
        self._validate_results(reranked, "reranker")
        candidate_metadata = {self._chunk_key(item["chunk"]): item for item in candidates}
        scored_results = []
        for result in reranked:
            key = self._chunk_key(result["chunk"])
            if key not in candidate_metadata:
                raise ValueError("Reranker returned a chunk outside the fused candidate set")
            metadata = candidate_metadata[key]
            reranker_score = result.get("reranker_score", result.get("score"))
            if not isinstance(reranker_score, (int, float)):
                raise ValueError("Malformed reranker result: score must be numeric")
            confidence = self._confidence(
                float(reranker_score),
                float(metadata["semantic_score"]),
                float(metadata["bm25_score"]),
                float(metadata["rrf_score"]),
            )
            scored_results.append({**result, "confidence": confidence})

        top_confidence = scored_results[0]["confidence"] if scored_results else 0.0
        accepted = (
            scored_results
            if top_confidence >= self.confidence_threshold
            else []
        )

        logger.info(
            "Reranked=%d accepted=%d selected_files=%s confidence_threshold=%.3f",
            len(reranked),
            len(accepted),
            sorted({result["chunk"].file_path for result in accepted}),
            self.confidence_threshold,
        )
        if scored_results:
            logger.info(
                "Confidence decision=%s score=%.3f",
                "accept" if top_confidence >= self.confidence_threshold else "abstain",
                top_confidence,
            )
        return accepted

    @staticmethod
    def _validate_results(results: Any, source: str) -> None:
        if not isinstance(results, list):
            raise ValueError(f"Malformed {source} retrieval results: expected a list")
        for index, result in enumerate(results):
            if not isinstance(result, Mapping) or "chunk" not in result:
                raise ValueError(
                    f"Malformed {source} result at position {index}: expected a chunk field"
                )
            if not isinstance(result.get("score"), (int, float)):
                raise ValueError(
                    f"Malformed {source} result at position {index}: score must be numeric"
                )
            chunk = result["chunk"]
            if not all(hasattr(chunk, field) for field in ("file_path", "start_line", "end_line", "symbol")):
                raise ValueError(
                    f"Malformed {source} result at position {index}: chunk metadata is incomplete"
                )

    @staticmethod
    def _confidence(
        reranker_score: float,
        semantic_score: float,
        bm25_score: float,
        rrf_score: float,
    ) -> float:
        """Combine bounded signals; high lexical/RRF scores cannot override weak relevance."""
        reranker_signal = min(1.0, max(0.0, (reranker_score + 12.0) / 20.0))
        semantic_signal = min(1.0, max(0.0, (semantic_score - 0.45) / 0.40))
        bm25_signal = min(1.0, max(0.0, bm25_score / (bm25_score + 3.0))) if bm25_score > 0 else 0.0
        rrf_signal = min(1.0, max(0.0, rrf_score / (rrf_score + 0.05))) if rrf_score > 0 else 0.0
        return (
            0.65 * reranker_signal
            + 0.15 * semantic_signal
            + 0.05 * bm25_signal
            + 0.15 * rrf_signal
        )

    @staticmethod
    def _chunk_key(chunk):
        return (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.symbol,
            chunk.chunk_type,
            chunk.content,
        )
