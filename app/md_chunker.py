from .models import CodeChunk


def create_markdown_chunks(
    source: str,
    file_path: str,
    repository: str,
    commit: str,
    language: str = "Markdown"
) -> list[CodeChunk]:

    chunks = []

    lines = source.splitlines()

    current_title = None
    current_start = None
    current_lines = []

    for line_number, line in enumerate(lines, start=1):

        # Markdown heading
        if line.startswith("#"):

            # Save previous section
            if current_title is not None:

                chunks.append(
                    CodeChunk(
                        repository=repository,
                        commit=commit,
                        file_path=file_path,
                        language=language,
                        symbol=current_title,
                        chunk_type="markdown_section",
                        start_line=current_start,
                        end_line=line_number - 1,
                        content="\n".join(current_lines).strip()
                    )
                )

            current_title = line.lstrip("#").strip()
            current_start = line_number
            current_lines = [line]

        else:
            current_lines.append(line)

    # Save final section
    if current_title is not None:

        chunks.append(
            CodeChunk(
                repository=repository,
                commit=commit,
                file_path=file_path,
                language=language,
                symbol=current_title,
                chunk_type="markdown_section",
                start_line=current_start,
                end_line=len(lines),
                content="\n".join(current_lines).strip()
            )
        )

    return chunks