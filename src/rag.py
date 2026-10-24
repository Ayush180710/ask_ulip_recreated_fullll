from .config import INDEX_FILE, METADATA_FILE, TOP_K, RERANK_CANDIDATES, MIN_RERANK_SCORE
from .index_store import VectorIndex
from .reranker import Reranker
from .llm import Generator

class RAGPipeline:
    def __init__(self):
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError('Prebuilt index is missing. Run python scripts/build_index.py')
        self.retriever = VectorIndex(INDEX_FILE, METADATA_FILE)
        self.reranker = Reranker()
        self.generator = Generator()

    def retrieve(self, question, top_k=TOP_K):
        # Fetch a wider candidate pool than we need, then let the cross-encoder
        # decide the final ordering. Recall comes from the vector search, precision
        # comes from the reranker.
        candidates = self.retriever.search(question, RERANK_CANDIDATES)
        return self.reranker.rerank(question, candidates, top_k)

    def ask(self, question, top_k=TOP_K):
        results = self.retrieve(question, top_k)
        # The reranker judges answerability directly, so it gates the response
        # instead of the raw vector-similarity score.
        if not results or results[0]['rerank_score'] < MIN_RERANK_SCORE:
            return ('I do not have enough evidence in the indexed public ULIP corpus to answer that question. '
                    'Please try a question about the documents shown in this demo.'), []
        return self.generator.answer(question, results), results
