from pathlib import Path


repo = Path("data\\repos\\test-repo")

suffix={".py", ".ipynb", ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".tsv", ".xml", ".html", ".htm", ".css", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h", ".hpp", ".java"}

language_map = {
        ".py": "Python",
        ".ipynb": "Jupyter Notebook",
        ".txt": "Text",
        ".md": "Markdown",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".csv": "CSV",
        ".tsv": "TSV",
        ".xml": "XML",
        ".html": "HTML",
        ".htm": "HTML",
        ".css": "CSS",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "React JSX",
        ".tsx": "React TSX",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C Header",
        ".hpp": "C++ Header",
        ".java": "Java",
    }

def scan_repository(repo_path: Path) -> list[Path]:
    """
    Scan the repository for files with specific suffixes.
    """
    found_files = []
    for item in repo_path.rglob("*"):
        if (item.stat().st_size >100 * 1024 * 1024 or item.stat().st_size == 0):  # Skip files larger than 100MB or empty files
            continue
        if item.is_file() and item.suffix in suffix:
            found_files.append(item)
    return found_files

def detect_language(file_path: Path) -> str:
    """
    Detect the programming language of a file based on its suffix.
    """
    
    return language_map.get(file_path.suffix.lower(), "unknown")

def read_file(file_path: Path) -> str:
    """
    Read the contents of a file.
    """

    try:
        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return ""
