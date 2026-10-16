"""The canonical formatter for retrieved code chunks in model context."""

from collections.abc import Iterable, Mapping
from typing import Any


def _chunk_from_result(item: Any) -> Any:
    """Accept either a CodeChunk or the result mapping used by retrievers."""
    if isinstance(item, Mapping):
        if "chunk" not in item:
            raise ValueError("Malformed retrieval result: expected a 'chunk' field")
        return item["chunk"]
    return item


def _field(chunk: Any, name: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(name)
    return getattr(chunk, name, None)


def normalize_chunks(results: Iterable[Any] | None) -> list[Any]:
    """Validate retrieval results and return their underlying chunks."""
    if results is None:
        raise ValueError("Retrieval results cannot be None; pass an empty list instead")

    try:
        chunks = [_chunk_from_result(item) for item in results]
    except TypeError as error:
        raise ValueError("Retrieval results must be an iterable of chunks") from error

    required_fields = ("file_path", "symbol", "start_line", "end_line", "content")
    for index, chunk in enumerate(chunks):
        missing = [name for name in required_fields if _field(chunk, name) is None]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"Malformed chunk at retrieval position {index}: missing {names}")
        if not isinstance(_field(chunk, "content"), str):
            raise ValueError(f"Malformed chunk at retrieval position {index}: content must be text")

    return chunks


class ContextBuilder:
    """Format chunks without retrieval scores or wrapper metadata."""

    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars

    def build(self, results: Iterable[Any]) -> str:
        if self.max_chars <= 0:
            return ""

        sections = []
        total_chars = 0

        for chunk in normalize_chunks(results):
            section = (
                f"Source: {_field(chunk, 'file_path')}\n"
                f"Symbol: {_field(chunk, 'symbol')}\n"
                f"Lines: {_field(chunk, 'start_line')}-{_field(chunk, 'end_line')}\n"
                f"\n"
                f"{_field(chunk, 'content')}"
            )

            if total_chars + len(section) > self.max_chars:
                break

            sections.append(section)
            total_chars += len(section)

        return "\n\n".join(sections)


def build_context(chunks: Iterable[Any], max_chars: int = 12000) -> str:
    """Compatibility function backed by the canonical ContextBuilder."""
    return ContextBuilder(max_chars=max_chars).build(chunks)
