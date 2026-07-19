from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

ARXIV_CSV_PATH = RAW_DATA_DIR / "arxiv_papers.csv"
PAPERS_PKL_PATH = PROCESSED_DATA_DIR / "papers.pkl"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_index.bin"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.pkl"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

TOP_K_DEFAULT = 10
TOP_K_MAX = 50

MIN_ABSTRACT_LENGTH = 50
MAX_ABSTRACT_LENGTH = 3000

SUPPORTED_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR",
    "cs.NE", "cs.RO", "stat.ML", "eess.AS", "eess.IV"
]

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"