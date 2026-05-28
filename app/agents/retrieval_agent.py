from __future__ import annotations

from app.rag.retriever import Retriever
from app.utils.helpers import RetrievedChunk


class RetrievalAgent:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, k=k)
