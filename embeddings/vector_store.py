from pathlib import Path
import pickle

import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(
        self,
        vectors: list[list[float]],
        chunks: list
    ) -> None:

        if len(vectors) != len(chunks):
            raise ValueError(
                "Number of vectors must match number of chunks"
            )

        array = np.asarray(
            vectors,
            dtype="float32"
        )

        self.index.add(array)
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: list[float],
        k: int = 5
    ):

        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)

        query = np.asarray(
            [query_vector],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query,
            k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            results.append({
                "score": float(score),
                "chunk": self.chunks[index]
            })

        return results

    def save(self, directory: str | Path) -> None:

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.pkl",
            "wb"
        ) as file:

            pickle.dump(
                self.chunks,
                file
            )

    @classmethod
    def load(
        cls,
        directory: str | Path
    ):

        directory = Path(directory)
        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.pkl"
        missing_files = [
            str(path)
            for path in (index_path, metadata_path)
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                "Vector store is incomplete; missing: " + ", ".join(missing_files)
            )

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "rb"
        ) as file:

            chunks = pickle.load(file)

        store = cls(
            dimension=index.d
        )

        store.index = index
        store.chunks = chunks

        return store

    @property
    def size(self) -> int:
        return self.index.ntotal