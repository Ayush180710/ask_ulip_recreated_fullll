from .config import INDEX_FILE,METADATA_FILE,TOP_K,MIN_RETRIEVAL_SCORE
from .index_store import VectorIndex
from .llm import Generator

class RAGPipeline:
    def __init__(self):
        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError('Prebuilt index is missing. Run python scripts/build_index.py')
        self.retriever=VectorIndex(INDEX_FILE,METADATA_FILE)
        self.generator=Generator()
    def ask(self,question,top_k=TOP_K):
        results=self.retriever.search(question,top_k)
        # Retrieval always returns nearest neighbours. A low score is not evidence that
        # the corpus answers the question, so refuse instead of grounding an answer in it.
        if not results or results[0]['score'] < MIN_RETRIEVAL_SCORE:
            return ('I do not have enough evidence in the indexed public ULIP corpus to answer that question. '
                    'Please try a question about the documents shown in this demo.'), []
        return self.generator.answer(question,results),results
