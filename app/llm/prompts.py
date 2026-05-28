from __future__ import annotations

from app.utils.helpers import RetrievedChunk

RESEARCH_SYSTEM_PROMPT = (
    "You are a professional multi-agent research assistant. Produce accurate, concise, "
    "evidence-grounded answers using only the supplied context. Lead with the answer, "
    "not with process narration. Synthesize instead of copying long passages. Cite every "
    "factual claim with the provided source label, for example [report.pdf#3]. Prefer "
    "the most relevant source over many weak sources. If evidence is incomplete, say so "
    "plainly. Avoid speculation, filler, generic textbook explanations, and unrelated "
    "retrieved context."
)


def build_research_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"Source: [{chunk.metadata.get('source', 'unknown')}#{chunk.chunk_id}]\n"
        f"Relevance score: {chunk.score:.3f}\n"
        f"Excerpt:\n{chunk.text}"
        for chunk in chunks
    )
    return (
        f"Research question:\n{query}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Write the final answer in this exact Markdown structure:\n"
        "**Answer**\n"
        "Give the direct answer in 1-3 sentences with citations.\n\n"
        "**Evidence**\n"
        "- Add 1-3 short bullets only when they improve the answer.\n"
        "- Each bullet must cite a source.\n\n"
        "**Limitations**\n"
        "Include this section only if the retrieved evidence is weak, incomplete, or conflicting.\n\n"
        "Final answer:"
    )
