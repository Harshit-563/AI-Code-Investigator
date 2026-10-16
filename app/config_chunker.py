import json
import yaml

from .models import CodeChunk


def create_config_chunk(
    source: str,
    file_path: str,
    repository: str,
    commit: str,
    language: str
) -> list[CodeChunk]:

    try:
        if language == "JSON":
            data = json.loads(source)

        else:
            data = yaml.safe_load(source)

        content = json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )

    except Exception:
        content = source

    return [
        CodeChunk(
            repository=repository,
            commit=commit,
            file_path=file_path,
            language=language,
            symbol=file_path,
            chunk_type="configuration",
            start_line=1,
            end_line=len(source.splitlines()),
            content=content
        )
    ]