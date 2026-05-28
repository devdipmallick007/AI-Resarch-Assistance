from __future__ import annotations

import re

from app.utils.helpers import RetrievedChunk


class ValidationAgent:
    """Checks that an answer is grounded in retrieved source chunks."""

    def validate(self, answer: str, chunks: list[RetrievedChunk]) -> dict:
        retrieved_citations = [f"{chunk.metadata.get('source', 'unknown')}#{chunk.chunk_id}" for chunk in chunks]
        answer_citations = re.findall(r"\[([^\[\]]+#\d+)\]", answer)
        citations = self._dedupe(answer_citations + retrieved_citations)
        has_context = bool(chunks)
        grounded = has_context and answer.strip() != "" and all(
            citation in retrieved_citations for citation in answer_citations
        )
        return {
            "grounded": grounded,
            "citations": citations,
            "warnings": [] if grounded else ["No supporting context was retrieved."],
        }

    def _dedupe(self, citations: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                ordered.append(citation)
        return ordered
