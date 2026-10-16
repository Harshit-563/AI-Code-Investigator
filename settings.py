"""Environment-backed configuration for indexing, retrieval, and generation."""

from dataclasses import dataclass
import logging
import os


def _float_setting(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return default if raw is None else float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def _int_setting(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return default if raw is None else int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    reranker_model: str
    retrieval_candidates: int
    semantic_weight: float
    keyword_weight: float
    rrf_k: int
    semantic_min_score: float
    confidence_threshold: float
    gemini_model: str
    generation_retries: int
    generation_backoff_seconds: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            reranker_model=os.getenv(
                "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            retrieval_candidates=_int_setting("RAG_RETRIEVAL_CANDIDATES", 20),
            semantic_weight=_float_setting("RAG_SEMANTIC_WEIGHT", 1.0),
            keyword_weight=_float_setting("RAG_KEYWORD_WEIGHT", 2.0),
            rrf_k=_int_setting("RAG_RRF_K", 60),
            semantic_min_score=_float_setting("RAG_SEMANTIC_MIN_SCORE", 0.57),
            confidence_threshold=_float_setting("RAG_CONFIDENCE_THRESHOLD", 0.40),
            gemini_model=os.getenv("RAG_GEMINI_MODEL", "gemini-3.6-flash"),
            generation_retries=_int_setting("RAG_GENERATION_RETRIES", 3),
            generation_backoff_seconds=_float_setting(
                "RAG_GENERATION_BACKOFF_SECONDS", 0.5
            ),
            log_level=os.getenv("RAG_LOG_LEVEL", "INFO").upper(),
        )
        if settings.retrieval_candidates <= 0:
            raise ValueError("RAG_RETRIEVAL_CANDIDATES must be positive")
        if settings.rrf_k <= 0:
            raise ValueError("RAG_RRF_K must be positive")
        if settings.generation_retries < 0:
            raise ValueError("RAG_GENERATION_RETRIES cannot be negative")
        if settings.generation_backoff_seconds < 0:
            raise ValueError("RAG_GENERATION_BACKOFF_SECONDS cannot be negative")
        if not 0 <= settings.confidence_threshold <= 1:
            raise ValueError("RAG_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if settings.semantic_weight < 0 or settings.keyword_weight < 0:
            raise ValueError("RAG retrieval weights cannot be negative")
        if not hasattr(logging, settings.log_level):
            raise ValueError("RAG_LOG_LEVEL must be a valid Python logging level")
        return settings


def configure_logging(level: str | None = None) -> None:
    """Configure concise process-wide logging without exposing retrieved text."""
    selected_level = (level or Settings.from_env().log_level).upper()
    numeric_level = getattr(logging, selected_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {selected_level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )