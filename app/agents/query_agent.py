from __future__ import annotations

import re


class QueryAgent:
    """Normalizes user questions before retrieval."""

    def refine(self, query: str) -> str:
        query = re.sub(r"\s+", " ", query or "").strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        query = re.sub(r"^what is (.+s)\??$", r"what are \1?", query, flags=re.IGNORECASE)
        return query
