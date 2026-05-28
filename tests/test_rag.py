from __future__ import annotations

from app.rag.embeddings import EmbeddingModel
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import VectorStore
from app.utils.helpers import Document


def test_chunker_splits_document():
    chunker = TextChunker(chunk_size=20, overlap=5)
    chunks = chunker.split([Document(text="alpha beta gamma delta epsilon", metadata={"source": "sample"})])
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "sample"


def test_vector_store_retrieves_relevant_chunk(tmp_path):
    store = VectorStore(index_dir=tmp_path, embedding_model=EmbeddingModel())
    chunks = TextChunker(chunk_size=80, overlap=10).split(
        [
            Document(text="Python is used for data automation and analysis.", metadata={"source": "a.txt"}),
            Document(text="Gardening requires sunlight and soil health.", metadata={"source": "b.txt"}),
        ]
    )
    store.add_chunks(chunks)
    results = store.search("data automation with python", k=1)
    assert results
    assert results[0].metadata["source"] == "a.txt"
