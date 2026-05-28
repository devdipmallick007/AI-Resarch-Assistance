from __future__ import annotations

import hashlib
import math

from app.llm.openai_client import OpenAIClient


class EmbeddingModel:
    def __init__(self, dimensions: int = 384, client: OpenAIClient | None = None) -> None:
        self.dimensions = dimensions
        self.client = client or OpenAIClient()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.client.embeddings(texts)
        if vectors:
            return vectors
        return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
