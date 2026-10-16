"""Streamlit frontend for the existing Codebase Investigator RAG runtime."""

from pathlib import Path
import sys
from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None


def _streamlit_missing() -> None:
    print(
        "Streamlit is not installed. Install project dependencies with "
        "'.venv\\Scripts\\python.exe -m pip install -r requirements.txt', "
        "then run '.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py'.",
        file=sys.stderr,
    )


if st is not None:
    from rag.streamlit_helpers import (
        is_sensitive_path,
        sanitize_retrieval_results,
        split_answer_and_sources,
    )
    from settings import Settings

    @st.cache_resource(show_spinner=False)
    def _load_cached_runtime():
        from rag.runtime import load_runtime

        return load_runtime()

    @st.cache_data(ttl=30, max_entries=2, show_spinner=False)
    def _cached_vector_status(
        index_dir: str,
        index_mtime: float,
        metadata_mtime: float,
    ) -> tuple[bool, str]:
        del index_dir, index_mtime, metadata_mtime
        return True, "Index files present"

    def _vector_status() -> tuple[bool, str]:
        index_dir = Path("data/vectorstore")
        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "metadata.pkl"
        missing = [path.name for path in (index_path, metadata_path) if not path.is_file()]
        if missing:
            return False, "Missing " + " and ".join(missing)
        return _cached_vector_status(
            str(index_dir),
            index_path.stat().st_mtime,
            metadata_path.stat().st_mtime,
        )

    def _safe_display_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sanitize_retrieval_results(results)

    def _render_sources(source_section: str, results: list[dict[str, Any]]) -> None:
        if not source_section:
            st.caption("No retrieved sources.")
            return
        st.markdown("#### Sources")
        for result in results:
            chunk = result["chunk"]
            path = str(chunk.file_path)
            if is_sensitive_path(path):
                continue
            st.markdown(
                f"**`{path}`** · `{chunk.symbol}` · lines {chunk.start_line}-{chunk.end_line}"
            )
            with st.expander(f"View {path} · {chunk.symbol}"):
                st.code(chunk.content, language=getattr(chunk, "language", None) or "text")

    def _render_message(message: dict[str, Any]) -> None:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                return

            status = message.get("status", "")
            if status:
                st.caption(status)
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"], message.get("results", []))
            if message.get("context"):
                with st.expander("Retrieved context"):
                    st.code(message["context"], language="markdown")

    def main() -> None:
        st.set_page_config(
            page_title="AI Code Investigator",
            page_icon=":material/code_blocks:",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        settings = Settings.from_env()
        index_ready, index_message = _vector_status()

        with st.sidebar:
            st.markdown("### Workspace")
            if index_ready:
                st.success(index_message, icon=":material/check_circle:")
            else:
                st.error(index_message, icon=":material/warning:")
            st.markdown("#### Runtime configuration")
            st.caption(f"Embedding · `{settings.embedding_model}`")
            st.caption(f"Reranker · `{settings.reranker_model}`")
            st.caption(f"Generator · `{settings.gemini_model}`")
            st.caption(f"Retrieval top-k · `{5}`")
            st.caption(f"Confidence threshold · `{settings.confidence_threshold:.2f}`")
            with st.expander("How it works"):
                st.markdown(
                    "**Index** → **Hybrid retrieval** → **Reranking** → "
                    "**Grounded answer** → **Sources**"
                )
                st.caption(
                    "Semantic and BM25 results are fused using weighted RRF. "
                    "Low-confidence questions are answered with an abstention."
                )
            if st.button("Clear conversation", icon=":material/delete_outline:", width="stretch"):
                st.session_state.messages = []
                st.rerun()

        st.caption("REPOSITORY INTELLIGENCE")
        st.title("AI Code Investigator")
        st.markdown(
            "Ask questions about the indexed codebase. Answers are grounded in "
            "retrieved files, symbols, and line ranges."
        )
        st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            _render_message(message)

        prompt = st.chat_input(
            "Ask about a function, file, workflow, or configuration…",
            disabled=not index_ready,
        )
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching and checking evidence…"):
                        hybrid, generator = _load_cached_runtime()
                        from rag.context_builder import ContextBuilder

                        retrieved = hybrid.retrieve(prompt, k=5)
                        safe_results = _safe_display_results(retrieved)
                        threshold = settings.confidence_threshold
                        if not safe_results:
                            answer = "I could not find enough information in the indexed codebase."
                            sources = ""
                            context = ""
                            status = "Abstained · evidence below confidence threshold or no matching chunks"
                        else:
                            context = ContextBuilder().build(safe_results)
                            generated = generator.generate(prompt, safe_results)
                            deterministic_sources = generator._format_sources(safe_results)
                            answer, sources = split_answer_and_sources(
                                generated, deterministic_sources
                            )
                            top_confidence = max(
                                (float(item.get("confidence", 0.0)) for item in safe_results),
                                default=0.0,
                            )
                            status = f"Grounded · confidence {top_confidence:.2f} · {len(safe_results)} chunks"
                    st.caption(status)
                    st.markdown(answer)
                    if sources:
                        _render_sources(sources, safe_results)
                    if context:
                        with st.expander("Retrieved context"):
                            st.code(context, language="markdown")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "results": safe_results,
                        "context": context,
                        "status": status,
                    })
                except ValueError as error:
                    message = str(error)
                    if "GOOGLE_API_KEY" in message:
                        friendly = "Gemini is not configured. Add GOOGLE_API_KEY to rag/.env, then reload the app."
                    else:
                        friendly = f"The request could not be processed: {message}"
                    st.error(friendly)
                    st.session_state.messages.append({"role": "assistant", "content": friendly})
                except FileNotFoundError:
                    friendly = "The vector index is missing. Build the repository index before asking questions."
                    st.error(friendly)
                    st.session_state.messages.append({"role": "assistant", "content": friendly})
                except RuntimeError as error:
                    friendly = str(error)
                    st.error(friendly)
                    st.session_state.messages.append({"role": "assistant", "content": friendly})
                except Exception:
                    friendly = "Something went wrong while searching the index. Check application logs for details."
                    st.error(friendly)
                    st.session_state.messages.append({"role": "assistant", "content": friendly})

else:
    def main() -> None:
        _streamlit_missing()


if __name__ == "__main__":
    main()