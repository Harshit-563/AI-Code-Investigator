import logging

from sentence_transformers import CrossEncoder
from settings import Settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker with a preference for distinct source files."""

    def __init__(
        self,
        model_name: str | None = None
    ):
        self.model_name = model_name or Settings.from_env().reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        results: list,
        k: int = 5
    ):

        if not results or k <= 0:
            return []

        pairs = [
            (
                query,
                f"""
        File: {result["chunk"].file_path}
        Symbol: {result["chunk"].symbol}
        Type: {result["chunk"].chunk_type}

        {result["chunk"].content}
        """
            )
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):

            reranked.append({
                **result,
                "score": float(score),
                "reranker_score": float(score),
                "chunk": result["chunk"]
            })

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )
        logger.debug(
            "Reranked candidates=%d top_score=%.4f",
            len(reranked),
            reranked[0]["score"],
        )

        # Prefer distinct files before taking additional chunks from a file.
        # A codebase question often needs both implementation and documentation;
        # without this, several README sections can crowd out a relevant code
        # file even when it was retrieved as a candidate.
        selected = []
        selected_files = set()

        for result in reranked:
            file_path = result["chunk"].file_path
            if file_path in selected_files:
                continue

            selected.append(result)
            selected_files.add(file_path)

            if len(selected) == k:
                return selected

        for result in reranked:
            if result in selected:
                continue

            selected.append(result)
            if len(selected) == k:
                break

        return selected
