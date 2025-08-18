import re
from typing import List

def simple_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return [c for c in chunks if c]
