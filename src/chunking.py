from .models import Chunk
from .config import CHUNK_SIZE, CHUNK_OVERLAP
import re

def clean(text):
    text = text.replace('\x00',' ')
    text = re.sub(r'[ \t]+',' ',text)
    text = re.sub(r'\n{3,}','\n\n',text)
    return text.strip()

def parse_file(path):
    raw = path.read_text(encoding='utf-8', errors='ignore')
    lines = raw.splitlines()
    meta = {}
    body=[]
    for line in lines:
        if ':' in line and line.split(':',1)[0] in {'TITLE','SOURCE','URL','DATE'}:
            k,v=line.split(':',1); meta[k]=v.strip()
        else:
            body.append(line)
    return meta.get('TITLE',path.stem), meta.get('SOURCE',path.name), meta.get('URL',''), meta.get('DATE',''), clean('\n'.join(body))

def build_chunks(raw_dir):
    chunks=[]
    for path in sorted(raw_dir.glob('*.txt')):
        title,source,url,date,text=parse_file(path)
        start=0; i=0; step=max(1,CHUNK_SIZE-CHUNK_OVERLAP)
        while start < len(text):
            end=min(len(text),start+CHUNK_SIZE)
            part=text[start:end].strip()
            if part:
                chunks.append(Chunk(f'{path.stem}-{i}',title,part,source,url,date))
            if end>=len(text): break
            start += step; i += 1
    return chunks
