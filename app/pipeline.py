from pathlib import Path

from .ingestion import clone_repository, get_current_commit
from .scanner import scan_repository, detect_language, read_file
from .chunker import create_python_chunks
from .md_chunker import create_markdown_chunks
from .config_chunker import create_config_chunk


def ingest_repository(
    repo_url: str,
    destination: Path
):
    repo_path = clone_repository(
        repo_url,
        destination
    )

    commit = get_current_commit(repo_path)

    files = scan_repository(repo_path)

    chunks = []

    for file_path in files:

        language = detect_language(file_path)

        source = read_file(file_path)

        if language == "Python":

            file_chunks = create_python_chunks(
                source=source,
                file_path=str(
                    file_path.relative_to(repo_path)
                ),
                language=language,
                repository=str(repo_path),
                commit=commit
                
            )

            chunks.extend(file_chunks)

        elif language == "Markdown":

            file_chunks = create_markdown_chunks(
                source=source,
                file_path=str(file_path.relative_to(repo_path)),
                language=language,
                repository=repo_url,
                commit=commit
            )

            chunks.extend(file_chunks)  
             
        elif language in ("JSON", "YAML"):

            file_chunks = create_config_chunk(
                source=source,
                file_path=str(file_path.relative_to(repo_path)),
                language=language,
                repository=repo_url,
                commit=commit
            )

            chunks.extend(file_chunks)     

    return {
        "repository": str(repo_path),
        "commit": commit,
        "files": files,
        "chunks": chunks
    }
