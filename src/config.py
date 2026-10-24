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
RERANK_MODEL = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
TOP_K = int(os.getenv('TOP_K', '5'))
# How many chunks the vector search hands to the reranker. Wider than TOP_K so the
# reranker can promote a chunk the bi-encoder ranked low.
RERANK_CANDIDATES = int(os.getenv('RERANK_CANDIDATES', '15'))
# Tuned against evaluation/evaluation_set.json with all-MiniLM-L6-v2 embeddings.
# A sweep over the 42 labelled cases put the best balance at 0.55: 28/32 answerable
# questions still answered, 9/10 out-of-corpus questions correctly refused.
# Kept for reference and for running the pipeline without reranking.
MIN_RETRIEVAL_SCORE = float(os.getenv('MIN_RETRIEVAL_SCORE', '0.55'))
# Cross-encoder logit, not a 0-1 similarity. Swept against the 42 labelled cases:
# answerable questions scored 2.28 to 9.94, out-of-corpus scored -11.21 to 2.19.
# 2.0 keeps all 32 answerable questions and refuses 9 of 10 out-of-corpus ones.
# A threshold of 2.24 scores a perfect 42/42 on this set, but it only separates two
# adjacent points 0.08 apart, so it is overfitted to this sample and not used here.
MIN_RERANK_SCORE = float(os.getenv('MIN_RERANK_SCORE', '2.0'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '80'))
HASH_DIM = int(os.getenv('HASH_DIM', '4096'))
