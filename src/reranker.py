from sentence_transformers import CrossEncoder
from .config import RERANK_MODEL

# Loaded once and reused, same pattern as the embedding model.
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model

class Reranker:
    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        model = _get_model()
        # The bi-encoder embedded the query and each chunk separately, so its score
        # only reflects how similar they look in vector space. The cross-encoder reads
        # the pair together and scores how well this chunk answers this question.
        pairs = [(query, c['chunk'].text) for c in candidates]
        scores = model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)
        return sorted(candidates, key=lambda c: -c['rerank_score'])[:top_k]
