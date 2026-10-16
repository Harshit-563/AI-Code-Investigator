from sentence_transformers import SentenceTransformer
from settings import Settings

class Embedder:

    def __init__(
        self,
        model_name: str | None = None
    ):
        self.model_name = model_name or Settings.from_env().embedding_model
        self.model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()

    def build_embedding_text(self, chunk) -> str:
        return (
            f"Repository: {chunk.repository}\n"
            f"File: {chunk.file_path}\n"
            f"Language: {chunk.language}\n"
            f"Symbol: {chunk.symbol}\n"
            f"Type: {chunk.chunk_type}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n\n"
            f"{chunk.content}"
        )

    def embed_chunk(self, chunk) -> list[float]:
        text = self.build_embedding_text(chunk)

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()

    def embed_chunks(self, chunks) -> list[list[float]]:
        texts = [
            self.build_embedding_text(chunk)
            for chunk in chunks
        ]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return vectors.tolist()