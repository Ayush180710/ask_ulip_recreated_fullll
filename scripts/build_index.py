import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.chunking import build_chunks
from src.embeddings import embed
from src.config import RAW_DIR,INDEX_DIR,INDEX_FILE,METADATA_FILE

chunks=build_chunks(RAW_DIR)
texts=[f'{c.title}\n{c.source}\n{c.text}' for c in chunks]
vectors=embed(texts)
INDEX_DIR.mkdir(parents=True,exist_ok=True)
with open(INDEX_FILE,'wb') as f: np.savez(f,vectors=vectors)
METADATA_FILE.write_text(json.dumps([c.__dict__ for c in chunks],ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Indexed {len(chunks)} chunks from {len(list(RAW_DIR.glob("*.txt")))} corpus files.')
print(f'Index: {INDEX_FILE}')
