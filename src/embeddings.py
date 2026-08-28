import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from .config import HASH_DIM

# Offline-safe deterministic embedding fallback. If sentence-transformers is installed,
# build_index.py can be switched to it with USE_SENTENCE_TRANSFORMERS=1.
VECTORIZER = HashingVectorizer(
    n_features=HASH_DIM,
    alternate_sign=False,
    norm=None,
    lowercase=True,
    stop_words='english',
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b',
    ngram_range=(1,2),
)

def embed(texts):
    x = VECTORIZER.transform(texts).astype('float32').toarray()
    # A query containing only unseen tokens has a zero vector. Convert the
    # resulting non-finite normalized values to zero so retrieval can refuse it.
    return np.nan_to_num(normalize(x, norm='l2'), nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
