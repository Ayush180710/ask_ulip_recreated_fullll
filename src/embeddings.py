from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

# Loaded once and reused. Downloading/loading the model on every call would be slow;
# this keeps a single copy in memory for the lifetime of the process.
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed(texts):
    model = _get_model()
    # normalize_embeddings=True makes every vector unit length, so a plain dot
    # product between two vectors equals their cosine similarity. That lets the
    # FAISS index use a simple inner-product search instead of a slower,
    # separate cosine computation.
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype('float32')
