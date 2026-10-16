"""Interactive command-line interface for codebase questions."""

from collections.abc import Callable

from rag.runtime import load_runtime


def run_cli(
    answer_question: Callable[[str], str],
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Ask repeated questions until the user enters exit or quit."""
    while True:
        try:
            query = input_fn("Question (exit to quit): ").strip()
        except EOFError:
            output_fn("Goodbye.")
            return

        if query.casefold() in {"exit", "quit"}:
            output_fn("Goodbye.")
            return
        if not query:
            continue

        try:
            output_fn(answer_question(query))
        except Exception as error:
            output_fn(f"Request failed: {error}")


def main() -> None:
    try:
        hybrid, generator = load_runtime()
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Unable to start RAG CLI: {error}") from error

    def answer_question(query: str) -> str:
        results = hybrid.retrieve(query, k=5)
        return generator.generate(query, results)

    run_cli(answer_question)


if __name__ == "__main__":
    main()