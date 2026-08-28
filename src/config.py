from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INDEX_DIR = DATA_DIR / 'index'
INDEX_FILE = INDEX_DIR / 'faiss.index'
METADATA_FILE = INDEX_DIR / 'metadata.json'
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
TOP_K = int(os.getenv('TOP_K', '5'))
MIN_RETRIEVAL_SCORE = float(os.getenv('MIN_RETRIEVAL_SCORE', '0.10'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '80'))
HASH_DIM = int(os.getenv('HASH_DIM', '4096'))
