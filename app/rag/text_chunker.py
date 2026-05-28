from __future__ import annotations

from app.utils.helpers import Document, TextChunk


class TextChunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 150) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for document in documents:
            text = " ".join(document.text.split())
            if not text:
                continue
            start = 0
            index = 0
            while start < len(text):
                end = self._find_chunk_end(text, start)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            metadata={**document.metadata, "chunk_index": index},
                        )
                    )
                if end == len(text):
                    break
                start = max(0, end - self.overlap)
                index += 1
        return chunks

    def _find_chunk_end(self, text: str, start: int) -> int:
        hard_end = min(start + self.chunk_size, len(text))
        if hard_end == len(text):
            return hard_end
        sentence_end = max(text.rfind(".", start, hard_end), text.rfind("?", start, hard_end), text.rfind("!", start, hard_end))
        if sentence_end > start + int(self.chunk_size * 0.55):
            return sentence_end + 1
        space_end = text.rfind(" ", start, hard_end)
        if space_end > start + int(self.chunk_size * 0.55):
            return space_end
        return hard_end
