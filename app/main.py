from pathlib import Path

from .pipeline import ingest_repository


# GitHub repository
repo_url = "https://github.com/Harshit-563/Agent_Coco"

# Local destination
repo_path = Path("data/repos/test-repo")


# Run ingestion
result = ingest_repository(
    repo_url,
    repo_path
)


# ============================================================
# 1. Repository information
# ============================================================

print("\n" + "=" * 70)
print("REPOSITORY INFORMATION")
print("=" * 70)

print("Repository:", result["repository"])
print("Commit:", result["commit"])


# ============================================================
# 2. Files found
# ============================================================

print("\n" + "=" * 70)
print("FILES FOUND")
print("=" * 70)

print("Total files:", len(result["files"]))

for i, file_path in enumerate(result["files"], start=1):
    print(f"{i}. {file_path}")


# ============================================================
# 3. Generated chunks
# ============================================================

print("\n" + "=" * 70)
print("CODE CHUNKS")
print("=" * 70)

print("Total chunks:", len(result["chunks"]))


for i, chunk in enumerate(result["chunks"], start=1):

    print("\n" + "-" * 70)

    print(f"Chunk #{i}")
    print("File:", chunk.file_path)
    print("Language:", chunk.language)
    print("Symbol:", chunk.symbol)
    print("Type:", chunk.chunk_type)

    print(
        "Lines:",
        chunk.start_line,
        "-",
        chunk.end_line
    )

    print("\nContent:")
    print(chunk.content)


# ============================================================
# 4. Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Files:", len(result["files"]))
print("Chunks:", len(result["chunks"]))
print("Commit:", result["commit"])
