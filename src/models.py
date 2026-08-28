from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    source: str
    url: str = ''
    date: str = ''
