import json
import numpy as np
from pathlib import Path
from .embeddings import embed
from .models import Chunk

class VectorIndex:
    def __init__(self,index_file,metadata_file):
        with open(metadata_file,encoding='utf-8') as f:
            self.metadata=json.load(f)
        # The portable demo index is stored as NumPy data inside data/index/faiss.index.
        with open(index_file,'rb') as f:
            self.vectors=np.load(f)['vectors'].astype('float32')

    def search(self,query,k=5):
        q=embed([query])[0]
        # Elementwise summation avoids an Accelerate/NumPy matmul issue observed
        # with this portable index on older macOS Python builds.
        scores=np.nan_to_num(np.sum(self.vectors * q, axis=1), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        k=min(k,len(scores))
        ids=np.argsort(-scores)[:k]
        return [
            {'chunk':Chunk(**self.metadata[int(i)]),'score':float(scores[int(i)])}
            for i in ids
        ]
