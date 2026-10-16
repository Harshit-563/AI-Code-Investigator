from pathlib import Path

repo = Path("")

# Find all files recursively
# Keep only .py files
# Print their paths
for item in repo.rglob("*"):
    if item.is_file() and item.suffix == ".py":
        print(item)