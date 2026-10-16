"""End-to-end smoke test for retrieval, context construction, and generation."""

from rag.context_builder import ContextBuilder, normalize_chunks
from rag.generator import Generator
from rag.runtime import load_runtime


def run_smoke_test(query: str = "What tools does Agent Coco use?") -> str:
    """Run the full local pipeline and verify citations match retrieved chunks."""
    hybrid, generator = load_runtime()
    results = hybrid.retrieve(query, k=5)
    chunks = normalize_chunks(results)
    context = ContextBuilder().build(results)
    if chunks and not context:
        raise AssertionError("Retrieved chunks produced empty prompt context")

    answer = generator.generate(query, chunks)
    expected_sources = Generator._format_sources(chunks)
    if chunks and not answer.endswith(expected_sources):
        raise AssertionError("Generated Sources section does not match retrieved chunks")
    if not chunks and answer != "I could not find enough information in the indexed codebase.":
        raise AssertionError("Empty retrieval did not produce the abstention response")

    print(f"QUERY:\n{query}\n")
    print(f"PROMPT CONTEXT:\n{context or '(no retrieved context)'}\n")
    print(f"ANSWER:\n{answer}")
    return answer


if __name__ == "__main__":
    try:
        run_smoke_test()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"RAG smoke test failed: {error}") from error