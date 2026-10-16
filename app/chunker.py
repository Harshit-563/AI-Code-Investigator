from .models import CodeChunk
from .parser import parse_python_file


def create_python_chunks(
    source: str,
    file_path: str,
    repository: str,
    commit: str,
    language: str = "python",
) -> list[CodeChunk]:
    """
    Create code chunks from a Python source file.
    """

    parsed_data = parse_python_file(source)

    chunks = []

    # Create chunks for classes
    for cls in parsed_data["classes"]:
        chunk = CodeChunk(
            file_path=file_path,
            language=language,
            symbol=cls["name"],
            chunk_type="class",
            start_line=cls["start_line"],
            end_line=cls["end_line"],
            content=cls["source"],
            repository=repository,
            commit=commit
        )

        chunks.append(chunk)

    # Create chunks for functions and methods
    for func in parsed_data["functions"]:

        chunk_type = (
            "method"
            if "." in func["name"]
            else "function"
        )

        chunk = CodeChunk(
            file_path=file_path,
            language=language,
            symbol=func["name"],
            chunk_type=chunk_type,
            start_line=func["start_line"],
            end_line=func["end_line"],
            content=func["source"],
            repository=repository,
            commit=commit
        )

        chunks.append(chunk)

    return chunks
