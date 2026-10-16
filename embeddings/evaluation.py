TEST_CASES = [
    {
        "query": "How does Agent Coco explore the codebase?",
        "expected_files": ["README.md", "main.py", "tools.py"],
        "category": "workflow",
    },
    {
        "query": "How does Agent Coco generate documentation?",
        "expected_files": ["README.md", "main.py"],
        "category": "documentation",
    },
    {
        "query": "How does the agent interact with the user?",
        "expected_files": ["main.py"],
        "category": "workflow",
    },
    {
        "query": "What tools does Agent Coco use?",
        "expected_files": ["tools.py", "README.md"],
        "category": "tools",
    },
    {
        "query": "What does get_file_structure return?",
        "expected_files": ["tools.py"],
        "category": "symbol",
    },
    {
        "query": "How does read_file_content protect sensitive files?",
        "expected_files": ["tools.py"],
        "category": "symbol",
    },
    {
        "query": "Which function writes documentation to disk?",
        "expected_files": ["tools.py"],
        "category": "tool function",
    },
    {
        "query": "What happens in start_agent when the user types quit?",
        "expected_files": ["main.py"],
        "category": "symbol",
    },
    {
        "query": "How does print_response handle function calls?",
        "expected_files": ["main.py"],
        "category": "symbol",
    },
    {
        "query": "Show the implementation in tools.py",
        "expected_files": ["tools.py"],
        "category": "file path",
    },
    {
        "query": "What does the README say about installing dependencies?",
        "expected_files": ["README.md"],
        "category": "documentation",
    },
    {
        "query": "Where should the Google API key be configured?",
        "expected_files": ["README.md", "main.py"],
        "category": "configuration",
    },
    {
        "query": "What are the EXPLORE READ WRITE steps?",
        "expected_files": ["README.md", "main.py"],
        "category": "documentation",
    },
    {
        "query": "How can I inspect the project folder structure?",
        "expected_files": ["tools.py", "README.md"],
        "category": "tool function",
    },
    {
        "query": "How does Agent Coco work?",
        "expected_files": ["README.md", "main.py", "tools.py"],
        "category": "ambiguous",
    },
    {
        "query": "What happens when I ask the agent to create a README?",
        "expected_files": ["README.md", "main.py", "tools.py"],
        "category": "ambiguous",
    },
    {
        "query": "Which files explain how the agent reads and writes files?",
        "expected_files": ["tools.py", "README.md"],
        "category": "ambiguous",
    },
    {
        "query": "How do I configure Kubernetes GPU autoscaling for this project?",
        "expected_files": [],
        "category": "unanswerable",
    },
    {
        "query": "Which Rust compiler flags enable WebAssembly SIMD here?",
        "expected_files": [],
        "category": "unanswerable",
    },
]


def _build_hybrid(vector_dir: str):
    """Initialize retrieval models only when evaluation is explicitly run."""
    from embeddings.bm25_store import BM25Store
    from embeddings.embedder import Embedder
    from embeddings.hybrid_retriever import HybridRetriever
    from embeddings.reranker import Reranker
    from embeddings.retriever import Retriever
    from embeddings.vector_store import VectorStore

    store = VectorStore.load(vector_dir)
    embedder = Embedder()
    semantic_retriever = Retriever(vector_store=store, embedder=embedder)
    bm25 = BM25Store()
    bm25.add(store.chunks)
    return HybridRetriever(semantic_retriever, bm25, Reranker())


def evaluate(k: int = 5, hybrid=None, vector_dir: str = "data/vectorstore") -> float:
    """Evaluate file-level recall on answerable queries and abstention on negatives."""
    if hybrid is None:
        hybrid = _build_hybrid(vector_dir)

    relevant_recalls = []
    negative_cases = 0
    correct_abstentions = 0
    case_scores = []

    for test in TEST_CASES:
        results = hybrid.retrieve(test["query"], k=k, candidate_k=20)
        retrieved_files = {
            result["chunk"].file_path
            for result in results
            if isinstance(result, dict) and hasattr(result.get("chunk"), "file_path")
        }
        expected_files = set(test["expected_files"])

        if expected_files:
            recall = len(retrieved_files & expected_files) / len(expected_files)
            relevant_recalls.append(recall)
            case_scores.append(recall)
            outcome = f"Recall: {recall:.2f}"
        else:
            negative_cases += 1
            abstained = not retrieved_files
            correct_abstentions += int(abstained)
            case_scores.append(float(abstained))
            outcome = f"Abstained: {abstained}"

        print(f"[{test['category']}] {test['query']}")
        print(f"  Expected: {sorted(expected_files)}")
        print(f"  Retrieved: {sorted(retrieved_files)}")
        print(f"  {outcome}")

    average_recall = sum(relevant_recalls) / len(relevant_recalls) if relevant_recalls else 0.0
    abstention_rate = correct_abstentions / negative_cases if negative_cases else 0.0
    overall_score = sum(case_scores) / len(case_scores) if case_scores else 0.0
    print(f"\nRecall@{k} over {len(relevant_recalls)} answerable queries: {average_recall:.3f}")
    print(f"No-answer accuracy: {correct_abstentions}/{negative_cases} ({abstention_rate:.3f})")
    print(f"Overall case score: {sum(case_scores):.1f}/{len(case_scores)} ({overall_score:.3f})")
    return average_recall


if __name__ == "__main__":
    evaluate(k=5)