"""Offline tests for presentation helpers; no Streamlit/backend models needed."""

import unittest

from app.models import CodeChunk
from rag.streamlit_helpers import (
    is_sensitive_path,
    redact_sensitive_text,
    sanitize_retrieval_results,
    split_answer_and_sources,
)


def chunk(file_path: str, content: str) -> CodeChunk:
    return CodeChunk(
        file_path=file_path,
        language="python",
        symbol="load_config",
        chunk_type="function",
        start_line=1,
        end_line=3,
        content=content,
        repository="test",
        commit="test",
    )


class StreamlitHelperTests(unittest.TestCase):
    def test_sensitive_paths_are_detected_across_platform_separators(self):
        self.assertTrue(is_sensitive_path("secrets/.env"))
        self.assertTrue(is_sensitive_path(r"secrets\.env.local"))
        self.assertFalse(is_sensitive_path("app/config.py"))

    def test_display_results_omit_env_chunks_and_redact_tokens(self):
        safe = chunk("config.py", "GOOGLE_API_KEY=AIza" + "A" * 30)
        env_chunk = chunk(".env", "GOOGLE_API_KEY=do-not-display")

        displayed = sanitize_retrieval_results([
            {"score": 0.5, "confidence": 0.7, "chunk": safe},
            {"score": 0.4, "confidence": 0.6, "chunk": env_chunk},
        ])

        self.assertEqual(len(displayed), 1)
        self.assertIn("[REDACTED]", displayed[0]["chunk"].content)
        self.assertNotIn("do-not-display", repr(displayed))

    def test_answer_split_only_accepts_exact_backend_sources_suffix(self):
        source_block = "Sources:\n- `tools.py` - `scan` (lines 1-5)"
        self.assertEqual(
            split_answer_and_sources(f"Grounded response.\n\n{source_block}", source_block),
            ("Grounded response.", source_block),
        )
        self.assertEqual(
            split_answer_and_sources("Plain response", source_block),
            ("Plain response", ""),
        )

    def test_redactor_masks_common_secret_assignments(self):
        self.assertEqual(
            redact_sensitive_text("API_KEY=not-a-real-key"),
            "API_KEY=[REDACTED]",
        )


if __name__ == "__main__":
    unittest.main()