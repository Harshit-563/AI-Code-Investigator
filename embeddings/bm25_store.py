import re
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Store:
    """Lexical index over chunk content and retrieval metadata."""

    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def add(self, chunks: list[Any]) -> None:
        """Replace the indexed corpus with the supplied code chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.bm25 = None
            return

        documents = [self._tokenize(self._build_document(chunk)) for chunk in self.chunks]
        self.bm25 = BM25Okapi(documents)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return the highest-scoring chunks for content and metadata terms."""
        if not self.chunks or self.bm25 is None or k <= 0:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        if not any(score > 0 for score in scores):
            query_terms = set(tokens)
            overlaps = [
                len(query_terms.intersection(self._tokenize(self._build_document(chunk))))
                for chunk in self.chunks
            ]
            ranked_indices = sorted(
                range(len(overlaps)),
                key=lambda index: (-overlaps[index], index),
            )[:k]
            return [
                {"score": float(overlaps[index]), "chunk": self.chunks[index]}
                for index in ranked_indices
                if overlaps[index] > 0
            ]

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index)
        )[:k]

        return [
            {
                "score": float(scores[i]),
                "chunk": self.chunks[i]
            }
            for i in ranked_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _tokenize(text: str):
        """Keep identifiers intact while also indexing path/snake-case parts."""
        tokens = re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", text.casefold())
        expanded = []
        for token in tokens:
            expanded.append(token)
            if "_" in token:
                expanded.extend(token.split("_"))
        return expanded

    @staticmethod
    def _build_document(chunk: Any) -> str:
        """Include retrieval metadata so file and symbol queries find code chunks."""
        return "\n".join(
            [
                f"File: {chunk.file_path}",
                f"Symbol: {chunk.symbol}",
                f"Type: {chunk.chunk_type}",
                chunk.content,
            ]
        )
