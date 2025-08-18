import argparse
import pickle
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import DOCS_DIR, FAISS_PATH, META_PATH, EMBED_MODEL_NAME, GOOGLE_API_KEY
from .utils import simple_chunk

def read_pdf(path: Path) -> List[Dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        if txt.strip():
            pages.append({"page": i+1, "text": txt})
    return pages

def build_corpus(docs_dir: Path) -> List[Dict]:
    corpus = []
    for pdf in docs_dir.glob("**/*.pdf"):
        try:
            pages = read_pdf(pdf)
            for page in pages:
                for ch in simple_chunk(page["text"], 800, 120):
                    corpus.append({
                        "source": str(pdf.relative_to(docs_dir)),
                        "page": page["page"],
                        "text": ch
                    })
        except Exception as e:
            print(f"[WARN] gagal baca {pdf}: {e}")
    return corpus

def embed_passages(passages: List[str], model) -> np.ndarray:
    texts = [f"passage: {p}" for p in passages]
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def write_faiss(emb: np.ndarray):
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(FAISS_PATH))

def main(docs_dir: Path):
    print(f"[INGEST] scan PDF di {docs_dir}")
    corpus = build_corpus(docs_dir)
    if not corpus:
        print("Tidak ada teks yang terindeks. Pastikan PDF berisi teks (bukan image-only).")
        return
    passages = [c["text"] for c in corpus]

    embedding = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL_NAME,
        api_key=GOOGLE_API_KEY,
    )

    emb = np.array(embedding.embed_documents(passages))
    print("[INGEST] embeddings:", emb.shape)

    print("[INGEST] build & save FAISS")
    write_faiss(emb)

    print("[INGEST] save metadata")
    with open(META_PATH, "wb") as f:
        pickle.dump(corpus, f)

    print("[DONE] index tersimpan:", FAISS_PATH, META_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default=str(DOCS_DIR))
    args = parser.parse_args()
    main(Path(args.docs))
