import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call
from unittest.mock import Mock
from unittest.mock import patch

from app.models import CodeChunk
from embeddings.bm25_store import BM25Store
from embeddings.hybrid_retriever import HybridRetriever
from rag.context_builder import ContextBuilder
from rag.generator import Generator
from rag.cli import run_cli
from embeddings.vector_store import VectorStore


def make_chunk(
    file_path: str,
    symbol: str,
    content: str,
    start_line: int = 1,
    end_line: int = 5,
    chunk_type: str = "function",
) -> CodeChunk:
    return CodeChunk(
        file_path=file_path,
        language="python",
        symbol=symbol,
        chunk_type=chunk_type,
        start_line=start_line,
        end_line=end_line,
        content=content,
        repository="test-repo",
        commit="test",
    )


class ContextAndCitationTests(unittest.TestCase):
    def test_context_uses_clean_chunk_format(self):
        chunk = make_chunk("tools.py", "get_file_structure", "Returns the file tree.")

        self.assertEqual(
            ContextBuilder().build([{"score": 0.8, "chunk": chunk}]),
            "Source: tools.py\n"
            "Symbol: get_file_structure\n"
            "Lines: 1-5\n"
            "\n"
            "Returns the file tree.",
        )

    def test_context_builder_rejects_malformed_retrieval_result(self):
        with self.assertRaisesRegex(ValueError, "expected a 'chunk' field"):
            ContextBuilder().build([{"score": 0.8}])

    def test_context_builder_rejects_chunk_missing_required_metadata(self):
        with self.assertRaisesRegex(ValueError, "missing symbol"):
            ContextBuilder().build([{"chunk": {"file_path": "tools.py"}}])

    def test_citations_are_deduplicated_in_retrieval_order(self):
        first = make_chunk("tools.py", "get_file_structure", "tree", 10, 25)
        repeated = make_chunk("tools.py", "get_file_structure", "tree", 10, 25)
        second = make_chunk("main.py", "main", "entry point", 30, 40)

        self.assertEqual(
            Generator._format_sources([first, repeated, second]),
            "Sources:\n"
            "- `tools.py` - `get_file_structure` (lines 10-25)\n"
            "- `main.py` - `main` (lines 30-40)",
        )


class RetrievalTests(unittest.TestCase):
    def test_bm25_finds_tool_from_metadata(self):
        tools_chunk = make_chunk(
            "tools.py",
            "get_file_structure",
            "Scans a folder and returns its file tree.",
        )
        unrelated_chunk = make_chunk(
            "parser.py",
            "parse_config",
            "Parses configuration values.",
        )
        store = BM25Store()
        store.add([tools_chunk, unrelated_chunk])

        results = store.search("What tools does Agent Coco use?", k=2)

        self.assertTrue(results)
        self.assertIs(results[0]["chunk"], tools_chunk)

    def test_hybrid_retriever_passes_fused_candidates_to_reranker(self):
        semantic_chunk = make_chunk("semantic.py", "semantic", "vector result")
        keyword_chunk = make_chunk("tools.py", "get_file_structure", "keyword result")
        semantic = Mock()
        semantic.retrieve.return_value = [{"score": 0.9, "chunk": semantic_chunk}]
        bm25 = Mock()
        bm25.search.return_value = [{"score": 1.2, "chunk": keyword_chunk}]
        reranker = Mock()
        reranker.rerank.return_value = [{"score": 0.7, "chunk": keyword_chunk}]
        hybrid = HybridRetriever(semantic, bm25, reranker)

        results = hybrid.retrieve("question", k=1, candidate_k=5)

        reranker.rerank.assert_called_once()
        args, kwargs = reranker.rerank.call_args
        self.assertEqual(args[0], "question")
        self.assertEqual(
            {item["chunk"].file_path for item in args[1]},
            {"semantic.py", "tools.py"},
        )
        self.assertEqual(kwargs["k"], 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk"], keyword_chunk)
        self.assertEqual(results[0]["score"], 0.7)
        self.assertGreaterEqual(results[0]["confidence"], hybrid.confidence_threshold)

    def test_hybrid_abstains_for_unrelated_signals_but_keeps_ambiguous_relevant(self):
        chunk = make_chunk("README.md", "overview", "Agent Coco documentation")
        semantic = Mock()
        bm25 = Mock()
        reranker = Mock()
        hybrid = HybridRetriever(semantic, bm25, reranker, confidence_threshold=0.4)

        weak_confidence = hybrid._confidence(-10.8, 0.59, 3.0, 0.04)
        useful_confidence = hybrid._confidence(0.8, 0.72, 2.0, 0.04)
        self.assertLess(weak_confidence, hybrid.confidence_threshold)
        self.assertGreaterEqual(useful_confidence, hybrid.confidence_threshold)

        weak_chunk = make_chunk("other.py", "unrelated", "irrelevant result")
        semantic.retrieve.return_value = [
            {"score": 0.59, "chunk": chunk},
            {"score": 0.58, "chunk": weak_chunk},
        ]
        bm25.search.return_value = [{"score": 3.0, "chunk": chunk}]
        reranker.rerank.return_value = [
            {"score": -10.8, "chunk": chunk},
            {"score": -11.0, "chunk": weak_chunk},
        ]
        self.assertEqual(hybrid.retrieve("unrelated", k=1), [])

        reranker.rerank.return_value = [
            {"score": 0.8, "chunk": chunk},
            {"score": -11.0, "chunk": weak_chunk},
        ]
        accepted = hybrid.retrieve("ambiguous relevant", k=2)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted[0]["chunk"], chunk)

    def test_hybrid_rejects_malformed_retrieval_dictionaries(self):
        semantic = Mock()
        semantic.retrieve.return_value = [{"score": 0.7}]
        hybrid = HybridRetriever(semantic, Mock(), Mock())

        with self.assertRaisesRegex(ValueError, "expected a chunk field"):
            hybrid.retrieve("query")


class RuntimeSafetyTests(unittest.TestCase):
    def test_empty_retrieval_does_not_call_model(self):
        generator = object.__new__(Generator)

        self.assertEqual(
            generator.generate("question", []),
            "I could not find enough information in the indexed codebase.",
        )

    def test_missing_api_key_fails_clearly(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY is not set"):
                Generator()

    def test_empty_model_response_gets_fallback_and_citations(self):
        generator = object.__new__(Generator)
        generator.retries = 0
        generator.backoff_seconds = 0
        generator.sleep_fn = Mock()
        generator.client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(text=" ")
            )
        )
        generator.model_name = "fake-model"
        chunk = make_chunk("tools.py", "get_file_structure", "Returns a tree.")

        answer = generator.generate("question", [chunk])

        self.assertTrue(answer.startswith("I could not find enough information"))
        self.assertTrue(answer.endswith(Generator._format_sources([chunk])))

    def test_model_sources_are_replaced_with_metadata_citations(self):
        generator = object.__new__(Generator)
        generator.retries = 0
        generator.backoff_seconds = 0
        generator.sleep_fn = Mock()
        generator.client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(
                    text="Answer.\n\nSources:\n- invented.py (lines 1-2)"
                )
            )
        )
        generator.model_name = "fake-model"
        chunk = make_chunk("tools.py", "get_file_structure", "Returns a tree.")

        answer = generator.generate("question", [chunk])

        self.assertNotIn("invented.py", answer)
        self.assertEqual(answer.count("Sources:"), 1)
        self.assertTrue(answer.endswith(Generator._format_sources([chunk])))

    def test_transient_generation_error_retries_with_exponential_backoff(self):
        generator = object.__new__(Generator)
        generator.retries = 2
        generator.backoff_seconds = 0.25
        generator.sleep_fn = Mock()
        generator.model_name = "fake-model"
        responses = [SimpleNamespace(status_code=503), SimpleNamespace(status_code=503), "ok"]

        def generate_content(**_kwargs):
            response = responses.pop(0)
            if response == "ok":
                return SimpleNamespace(text="answer")
            error = RuntimeError("temporary")
            error.status_code = response.status_code
            raise error

        generator.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content)
        )
        chunk = make_chunk("tools.py", "get_file_structure", "Returns a tree.")

        answer = generator.generate("question", [chunk])

        self.assertTrue(answer.startswith("answer"))
        self.assertEqual(generator.sleep_fn.call_args_list, [call(0.25), call(0.5)])

    def test_non_transient_generation_error_is_not_retried(self):
        generator = object.__new__(Generator)
        generator.retries = 3
        generator.backoff_seconds = 0.1
        generator.sleep_fn = Mock()
        generator.model_name = "fake-model"
        request = Mock(side_effect=RuntimeError("bad request"))
        generator.client = SimpleNamespace(models=SimpleNamespace(generate_content=request))
        chunk = make_chunk("tools.py", "get_file_structure", "Returns a tree.")

        with self.assertRaisesRegex(RuntimeError, "Gemini generation failed"):
            generator.generate("question", [chunk])
        request.assert_called_once()
        generator.sleep_fn.assert_not_called()

    def test_transient_generation_exhaustion_has_no_sources_or_low_level_details(self):
        generator = object.__new__(Generator)
        generator.retries = 1
        generator.backoff_seconds = 0
        generator.sleep_fn = Mock()
        generator.model_name = "fake-model"
        error = RuntimeError("sensitive server details")
        error.status_code = 503
        request = Mock(side_effect=error)
        generator.client = SimpleNamespace(models=SimpleNamespace(generate_content=request))
        chunk = make_chunk("tools.py", "get_file_structure", "Returns a tree.")

        with self.assertRaisesRegex(RuntimeError, "Gemini generation failed") as context:
            generator.generate("question", [chunk])
        self.assertNotIn("Sources:", str(context.exception))
        self.assertNotIn("sensitive server details", str(context.exception))
        self.assertEqual(request.call_count, 2)

    def test_settings_environment_overrides_and_validation(self):
        from settings import Settings

        with patch.dict(os.environ, {
            "RAG_CONFIDENCE_THRESHOLD": "0.65",
            "RAG_GENERATION_RETRIES": "4",
            "RAG_LOG_LEVEL": "DEBUG",
        }):
            settings = Settings.from_env()
        self.assertEqual(settings.confidence_threshold, 0.65)
        self.assertEqual(settings.generation_retries, 4)
        self.assertEqual(settings.log_level, "DEBUG")

        with patch.dict(os.environ, {"RAG_CONFIDENCE_THRESHOLD": "2.0"}):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                Settings.from_env()

    def test_generation_service_failure_is_actionable_and_does_not_echo_error(self):
        generator = object.__new__(Generator)

        def fail_generation(**_kwargs):
            raise RuntimeError("secret-bearing low-level response")

        generator.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=fail_generation)
        )
        generator.model_name = "fake-model"
        chunk = make_chunk("tools.py", "get_file_structure", "Returns a tree.")

        with self.assertRaisesRegex(RuntimeError, "Gemini generation failed") as context:
            generator.generate("question", [chunk])
        self.assertNotIn("secret-bearing", str(context.exception))

    def test_missing_vector_store_files_raise_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "index.faiss.*metadata.pkl"):
                VectorStore.load(directory)

    def test_cli_repeats_questions_and_exits_cleanly(self):
        inputs = iter(["first question", "second question", "quit"])
        outputs = []
        answers = []

        run_cli(
            lambda query: answers.append(query) or f"answer: {query}",
            input_fn=lambda _prompt: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(answers, ["first question", "second question"])
        self.assertEqual(outputs[-1], "Goodbye.")


if __name__ == "__main__":
    unittest.main()