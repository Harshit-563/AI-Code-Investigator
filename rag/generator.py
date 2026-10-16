import os
import re
import logging
import time
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

from google import genai
from rag.context_builder import ContextBuilder, normalize_chunks
from settings import Settings

logger = logging.getLogger(__name__)


class Generator:
    """Generate codebase answers and append citations from retrieved chunks."""

    def __init__(
        self,
        model_name: str | None = None,
        retries: int | None = None,
        backoff_seconds: float | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        settings = Settings.from_env()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or settings.gemini_model
        self.retries = settings.generation_retries if retries is None else retries
        self.backoff_seconds = (
            settings.generation_backoff_seconds
            if backoff_seconds is None
            else backoff_seconds
        )
        if self.retries < 0 or self.backoff_seconds < 0:
            raise ValueError("Generation retries and backoff must be non-negative")
        self.sleep_fn = sleep_fn

    def generate(self, query: str, chunks: list) -> str:
        normalized_chunks = normalize_chunks(chunks)
        if not normalized_chunks:
            return self._empty_answer()

        context = ContextBuilder().build(normalized_chunks)

        prompt = f"""
You are a codebase investigation assistant.

Answer the user's question using ONLY the provided codebase context.

If the context does not contain enough information, say:
"I could not find enough information in the indexed codebase."

Be precise and mention relevant file names and symbols.
Do not create citations, a Sources section, or line references. Source citations
are added separately from the retrieved metadata.

USER QUESTION:
{query}

CODEBASE CONTEXT:
{context}
"""

        try:
            response = self._generate_with_retries(prompt)
        except Exception as error:
            logger.error("Gemini generation failed (%s)", type(error).__name__)
            raise RuntimeError("Gemini generation failed; check service availability and configuration.") from error

        # Citations are deliberately generated from the retrieved chunks after
        # the model responds.  This keeps source attribution grounded in the
        # retriever output rather than asking the model to invent references.
        sources = self._format_sources(normalized_chunks)
        answer = self._remove_model_sources(getattr(response, "text", None) or "")
        answer = answer.strip() or "I could not find enough information in the indexed codebase."
        return f"{answer}\n\n{sources}"

    @staticmethod
    def _format_sources(chunks: list) -> str:
        """Return deterministic, de-duplicated citations for retrieved chunks."""
        chunks = normalize_chunks(chunks)
        citations = []
        seen = set()

        for chunk in chunks:
            get = chunk.get if isinstance(chunk, dict) else lambda field: getattr(chunk, field)
            citation = (
                get("file_path"),
                get("symbol"),
                get("start_line"),
                get("end_line"),
            )

            if citation in seen:
                continue

            seen.add(citation)
            file_path, symbol, start_line, end_line = citation
            citations.append(
                f"- `{file_path}` - `{symbol}` (lines {start_line}-{end_line})"
            )

        return "Sources:\n" + "\n".join(citations)

    @staticmethod
    def _remove_model_sources(answer: str) -> str:
        """Remove a model-authored Sources block before canonical citations."""
        return re.sub(r"(?im)^\s*Sources:\s*$[\s\S]*\Z", "", answer).rstrip()

    @staticmethod
    def _empty_answer() -> str:
        return "I could not find enough information in the indexed codebase."

    def _generate_with_retries(self, prompt: str):
        for attempt in range(self.retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
            except Exception as error:
                status = self._status_code(error)
                if status not in {429, 500, 502, 503, 504}:
                    logger.error("Gemini request failed with non-retryable error (%s)", type(error).__name__)
                    raise
                if attempt >= self.retries:
                    logger.error("Gemini transient error exhausted retries status=%s", status)
                    raise
                delay = self.backoff_seconds * (2 ** attempt)
                logger.warning(
                    "Gemini transient status=%s retry=%d/%d backoff_seconds=%.2f",
                    status,
                    attempt + 1,
                    self.retries,
                    delay,
                )
                self.sleep_fn(delay)

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        for attribute in ("status_code", "code"):
            value = getattr(error, attribute, None)
            if isinstance(value, int):
                return value
            candidate = getattr(value, "value", None)
            if isinstance(candidate, int):
                return candidate
        response = getattr(error, "response", None)
        value = getattr(response, "status_code", None)
        return value if isinstance(value, int) else None
