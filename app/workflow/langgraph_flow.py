from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from app.agents import QueryAgent, ResearchAgent, RetrievalAgent, ValidationAgent
from app.rag.document_loader import DocumentLoader
from app.rag.retriever import Retriever
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    query: str
    refined_query: str
    answer: str
    citations: list[str]
    warnings: list[str]
    grounded: bool


class ResearchWorkflow:
    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.query_agent = QueryAgent()
        self.retrieval_agent = RetrievalAgent(Retriever(self.vector_store))
        self.research_agent = ResearchAgent()
        self.validation_agent = ValidationAgent()
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        logger.info(
            "ResearchWorkflow initialized | indexed_chunks=%s | vector_store=%s",
            self.vector_store.count(),
            self.vector_store.index_file,
        )

    def ingest_document(self, path: str | Path) -> int:
        logger.info("Loading document: %s", path)
        documents = self.loader.load(path)
        chunks = self.chunker.split(documents)
        logger.info("Document loaded | path=%s | documents=%s | chunks=%s", path, len(documents), len(chunks))
        count = self.vector_store.add_chunks(chunks)
        logger.info("Indexed %s chunks from %s", count, path)
        return count

    def run(self, query: str, k: int = 5) -> ResearchState:
        logger.info("Research query started | chars=%s | k=%s", len(query), k)
        refined_query = self.query_agent.refine(query)
        logger.info("Query refined | original=%r | refined=%r", query, refined_query)
        chunks = self.retrieval_agent.retrieve(refined_query, k=k)
        logger.info(
            "Retrieved %s chunks | top_sources=%s",
            len(chunks),
            [f"{chunk.metadata.get('source', 'unknown')}#{chunk.chunk_id}" for chunk in chunks[:3]],
        )
        answer = self.research_agent.answer(refined_query, chunks)
        validation = self.validation_agent.validate(answer, chunks)
        logger.info(
            "Research query finished | grounded=%s | answer_chars=%s | citations=%s",
            validation["grounded"],
            len(answer),
            len(validation["citations"]),
        )
        return {
            "query": query,
            "refined_query": refined_query,
            "answer": answer,
            "citations": validation["citations"],
            "warnings": validation["warnings"],
            "grounded": validation["grounded"],
        }
