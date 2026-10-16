"""Streamlit-independent helpers for safe RAG result presentation."""

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace
from pathlib import PurePosixPath
import re
from typing import Any

from rag.context_builder import normalize_chunks

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:GOOGLE_API_KEY|API_KEY|SECRET_KEY|ACCESS_TOKEN|PASSWORD)\s*[=:]\s*)[^\s,;]+"
)
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)


def is_sensitive_path(file_path: str) -> bool:
    """Return true for dotenv files, including nested and Windows-style paths."""
    normalized = file_path.replace("\\", "/")
    return any(part.casefold().startswith(".env") for part in PurePosixPath(normalized).parts)


def redact_sensitive_text(content: str) -> str:
    """Mask common credential assignments and recognizable token formats."""
    safe_content = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", content)
    for pattern in _SECRET_PATTERNS:
        safe_content = pattern.sub("[REDACTED]", safe_content)
    return safe_content


def sanitize_retrieval_results(results: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize wrapper results, omit dotenv chunks, and redact displayed content."""
    chunks = normalize_chunks(results)
    safe_results: list[dict[str, Any]] = []
    for index, (result, chunk) in enumerate(zip(results, chunks)):
        if is_sensitive_path(str(chunk.file_path)):
            continue

        content = redact_sensitive_text(chunk.content)
        if is_dataclass(chunk):
            safe_chunk = replace(chunk, content=content)
        elif isinstance(chunk, Mapping):
            safe_chunk = {**chunk, "content": content}
        else:
            raise ValueError(f"Unsupported chunk type at result {index}")

        safe_result = dict(result) if isinstance(result, Mapping) else {"chunk": chunk}
        safe_result["chunk"] = safe_chunk
        safe_results.append(safe_result)
    return safe_results


def split_answer_and_sources(answer: str, deterministic_sources: str) -> tuple[str, str]:
    """Separate the exact backend-generated Sources suffix from answer prose."""
    if not deterministic_sources or not answer.endswith(deterministic_sources):
        return answer.strip(), ""
    prose = answer[: -len(deterministic_sources)].rstrip()
    return prose, deterministic_sources
