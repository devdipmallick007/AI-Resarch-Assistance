from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from app.rag.embeddings import EmbeddingModel
from app.utils.helpers import RetrievedChunk, TextChunk


class VectorStore:
    def __init__(self, index_dir: str | Path | None = None, embedding_model: EmbeddingModel | None = None) -> None:
        self.index_dir = Path(index_dir or os.getenv("VECTOR_STORE_PATH", "data/faiss_index"))
        self.index_file = self.index_dir / "store.json"
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        vectors = self.embedding_model.embed([chunk.text for chunk in chunks])
        start_id = len(self._records)
        for offset, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self._records.append(
                {
                    "id": str(start_id + offset),
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": vector,
                }
            )
        self._save()
        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not self._records:
            return []
        query_vector = self.embedding_model.embed([query])[0]
        query_terms = self._tokenize(query)
        definition_entity = self._definition_entity(query)
        phrase_score_query = self._normalized_text(query)
        scored = []
        for record in self._records:
            text = record["text"]
            vector_score = self._cosine(query_vector, record["embedding"])
            keyword_score = self._keyword_score(query_terms, text)
            coverage_score = self._coverage_score(query_terms, text)
            phrase_score = self._phrase_score(phrase_score_query, definition_entity, text)
            definition_score = self._definition_score(query_terms, text)
            entity_definition_score = self._entity_definition_score(definition_entity, text)
            if query_terms and len(query_terms) <= 3 and coverage_score < 0.67:
                continue
            if definition_entity:
                score = (
                    (1.45 * entity_definition_score)
                    + (0.35 * phrase_score)
                    + (0.25 * keyword_score)
                    + (0.20 * coverage_score)
                    + (0.10 * definition_score)
                    + (0.05 * vector_score)
                )
            else:
                score = (
                    (0.35 * vector_score)
                    + (0.35 * keyword_score)
                    + (0.20 * phrase_score)
                    + (0.15 * coverage_score)
                    + (0.10 * definition_score)
                )
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievedChunk] = []
        seen_texts: set[str] = set()
        for score, record in scored:
            text_key = self._dedupe_key(record["text"])
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            results.append(
                RetrievedChunk(
                    text=record["text"],
                    metadata=record["metadata"],
                    score=score,
                    chunk_id=record["id"],
                )
            )
            if len(results) == k:
                break
        return results

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records = []
        self._save()

    def _load(self) -> list[dict]:
        if not self.index_file.exists():
            return []
        return json.loads(self.index_file.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.index_file.write_text(json.dumps(self._records, indent=2), encoding="utf-8")

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            size = min(len(left), len(right))
            left = left[:size]
            right = right[:size]
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)

    def _tokenize(self, text: str) -> set[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "for",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
            "with",
        }
        tokens = set()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
            if token in stop_words:
                continue
            tokens.add(token)
            if len(token) > 3 and token.endswith("s"):
                tokens.add(token[:-1])
        return tokens

    def _keyword_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = self._tokenize(text)
        if not text_terms:
            return 0.0
        overlap = query_terms & text_terms
        return len(overlap) / len(query_terms)

    def _coverage_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        text_terms = self._tokenize(text)
        if not text_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    def _definition_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        lowered = text.lower()
        score = 0.0
        definition_markers = (" is ", " are ", " refers to ", " defined as ", " framework", " library")
        if any(marker in lowered for marker in definition_markers):
            score += 0.4
        first_window = lowered[:450]
        if any(term in first_window for term in query_terms):
            score += 0.4
        if lowered.strip().startswith(tuple(query_terms)):
            score += 0.2
        return min(score, 1.0)

    def _phrase_score(self, query: str, entity: str, text: str) -> float:
        lowered = self._normalized_text(text)
        score = 0.0
        if entity and entity in lowered:
            score += 0.7
        query_without_question_words = re.sub(
            r"\b(?:what|who|where|when|why|how|is|are|was|were|the|a|an)\b",
            " ",
            query,
        )
        important_phrase = " ".join(query_without_question_words.split())
        if important_phrase and important_phrase in lowered:
            score += 0.3
        return min(score, 1.0)

    def _definition_entity(self, query: str) -> str:
        match = re.search(r"\b(?:what|who)\s+(?:is|are|was|were)\s+(.+?)[?!.]*$", query.strip(), re.IGNORECASE)
        if not match:
            return ""
        entity = re.sub(r"\b(?:the|a|an)\b", "", match.group(1), flags=re.IGNORECASE)
        entity = re.sub(r"\b(?:in|of|from|using|with)\s+[a-zA-Z][a-zA-Z0-9_ -]*$", "", entity, flags=re.IGNORECASE)
        return " ".join(entity.split()).lower()

    def _entity_definition_score(self, entity: str, text: str) -> float:
        if not entity:
            return 0.0
        lowered = text.lower()
        escaped = re.escape(entity)
        score = 0.0
        patterns = (
            rf"\b{escaped}\s+is\s+(?:a|an|the)?\b",
            rf"\b{escaped}\s+are\s+(?:a|an|the)?\b",
            rf"\b{escaped}\s+(?:a|an|the)\s+",
            rf"\b{escaped}\s+(?:built-in|standard|mutable|immutable)\b",
            rf"\b{escaped}\s+refers\s+to\b",
            rf"\b{escaped}\s+is\s+defined\s+as\b",
        )
        if any(re.search(pattern, lowered) for pattern in patterns):
            score += 1.0
        if re.search(rf"\b{escaped}\b", lowered[:600]):
            score += 0.35
        if lowered.strip().startswith(entity):
            score += 0.25
        return min(score, 1.0)

    def _dedupe_key(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        return normalized[:500]

    def _normalized_text(self, text: str) -> str:
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text.lower())
        return re.sub(r"\s+", " ", text).strip()
