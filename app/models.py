from dataclasses import dataclass


@dataclass
class FileMetadata:
    path: str
    language: str
    size: int
    
@dataclass
class CodeChunk:
    file_path: str
    language: str
    symbol: str
    chunk_type: str
    start_line: int
    end_line: int
    content: str
    repository: str
    commit: str
   