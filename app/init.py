from pathlib import Path

from .pipeline import ingest_repository
from embeddings.embedder import Embedder


repo_url = "https://github.com/Harshit-563/Agent_Coco"

result = ingest_repository(
    repo_url,
    Path("data/repos/test-repo")
)

chunks = result["chunks"]

print("Chunks:", len(chunks))


embedder = Embedder()

vectors = embedder.embed_chunks(chunks)

print("Vectors:", len(vectors))

if vectors:
    print("Vector dimensions:", len(vectors[0]))

    print("\nFirst chunk:")
    print("File:", chunks[0].file_path)
    print("Symbol:", chunks[0].symbol)

    print("\nFirst 10 vector values:")
    print(vectors[0][:10])