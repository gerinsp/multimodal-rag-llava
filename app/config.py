from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# RAG
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.pkl"
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))

# Generation
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "384"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# LLaVA modes
LLAVA_MODE = os.getenv("LLAVA_MODE", "local").lower()  # local | api

# Local

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLAVA_LOCAL_PATH = PROJECT_ROOT / "model"
LLAVA_ADAPTER_PATH = PROJECT_ROOT / "adapter"

LLAVA_MODEL_NAME = str(LLAVA_LOCAL_PATH)

HF_TOKEN = os.getenv("HF_TOKEN", "")

# HF API
HF_API_URL = os.getenv("HF_API_URL", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

