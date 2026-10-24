import json
import faiss
from .embeddings import embed
from .models import Chunk

class VectorIndex:
    def __init__(self, index_file, metadata_file):
        with open(metadata_file, encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.index = faiss.read_index(str(index_file))

    def search(self, query, k=5):
        q = embed([query])
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        return [
            {'chunk': Chunk(**self.metadata[int(i)]), 'score': float(s)}
            for s, i in zip(scores[0], ids[0]) if i != -1
        ]
