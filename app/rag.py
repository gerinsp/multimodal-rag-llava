import pickle
from typing import List, Tuple
import faiss
import numpy as np
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import FAISS_PATH, META_PATH, EMBED_MODEL_NAME, GOOGLE_API_KEY

class Retriever:
    def __init__(self):
        self.index = faiss.read_index(str(FAISS_PATH))
        with open(META_PATH, "rb") as f:
            self.meta = pickle.load(f)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.embedder = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL_NAME,
            api_key=GOOGLE_API_KEY,
        )

    def embed_query(self, q: str) -> np.ndarray:
        vec = self.embedder.embed_query(q)   # -> List[float]
        return np.array([vec], dtype="float32")

    def search(self, q: str, k: int = 4) -> List[Tuple[float, dict]]:
        qv = self.embed_query(q)
        scores, idx = self.index.search(qv, k)
        results = []
        for s, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            results.append((float(s), self.meta[i]))
        return results
