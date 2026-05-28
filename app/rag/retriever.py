from __future__ import annotations

from app.rag.vector_store import VectorStore
from app.utils.helpers import RetrievedChunk


class Retriever:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self.vector_store.search(query, k=k)
