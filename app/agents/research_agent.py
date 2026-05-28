from __future__ import annotations

import re

from app.llm.openai_client import OpenAIClient
from app.llm.prompts import RESEARCH_SYSTEM_PROMPT, build_research_prompt
from app.utils.helpers import RetrievedChunk


class ResearchAgent:
    def __init__(self, client: OpenAIClient | None = None) -> None:
        self.client = client or OpenAIClient()

    def answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I could not find relevant information in the indexed documents."

        prompt = build_research_prompt(query, chunks)
        response = self.client.chat(RESEARCH_SYSTEM_PROMPT, prompt)
        if response:
            return response
        return self._extractive_answer(query, chunks)

    def _extractive_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        query_terms = self._query_terms(query)
        analytical_answer = self._analytical_answer(query, chunks)
        if analytical_answer:
            return analytical_answer

        definition_entity = self._definition_entity(query)
        definition_answer = self._definition_answer(definition_entity, chunks)
        if definition_answer:
            return definition_answer

        source_lines = []
        seen_sentences: set[str] = set()
        for chunk in chunks[:6]:
            label = chunk.metadata.get("source", "unknown source")
            sentence = self._best_sentence(chunk.text, query_terms)
            sentence_key = sentence.lower()
            if sentence and sentence_key not in seen_sentences:
                seen_sentences.add(sentence_key)
                source_lines.append(f"- {sentence} [{label}#{chunk.chunk_id}]")
            if len(source_lines) == 3:
                break
        if not source_lines:
            source_lines = [
                f"- {self._clean_pdf_text(chunks[0].text)[:500]} [{chunks[0].metadata.get('source', 'unknown source')}#{chunks[0].chunk_id}]"
            ]
        return "**Answer**\nBased on the strongest retrieved evidence, the answer is:\n\n**Evidence**\n" + "\n".join(source_lines)

    def _analytical_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        lowered = query.lower()
        analytical_terms = ("compare", "contrast", "why", "how", "explain", "discuss", "evaluate")
        if not any(term in lowered for term in analytical_terms):
            return ""

        query_terms = self._query_terms(query)
        candidates = self._rank_evidence_sentences(chunks, query_terms)
        if len(candidates) < 2:
            return ""

        top = candidates[:3]
        answer_points = [
            f"{sentence} {self._citation(chunk)}"
            for _, sentence, chunk in top[:2]
        ]
        evidence_points = [
            f"- {sentence} {self._citation(chunk)}"
            for _, sentence, chunk in top
        ]
        return "**Answer**\n" + " ".join(answer_points) + "\n\n**Evidence**\n" + "\n".join(evidence_points)

    def _query_terms(self, query: str) -> set[str]:
        stop_words = {"a", "an", "and", "are", "is", "of", "the", "to", "what"}
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", query.lower())
            if token not in stop_words
        }

    def _best_sentence(self, text: str, query_terms: set[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return ""

        def score(sentence: str) -> float:
            lowered = sentence.lower()
            terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", lowered))
            overlap = len(query_terms & terms)
            definition_bonus = 2 if any(marker in lowered for marker in (" is ", " are ", " refers to ", " framework", " library", " system", " capability")) else 0
            return overlap + definition_bonus

        best = max(sentences, key=score).strip()
        return self._clean_pdf_text(best[:700])

    def _find_sentence(self, chunks: list[RetrievedChunk], keywords: tuple[str, ...]) -> tuple[str, RetrievedChunk] | None:
        best: tuple[float, str, RetrievedChunk] | None = None
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text.strip()):
                cleaned = self._clean_pdf_text(sentence)
                lowered = cleaned.lower()
                score = sum(1 for keyword in keywords if keyword in lowered)
                if score == 0:
                    continue
                candidate = (float(score), cleaned, chunk)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return None
        return best[1], best[2]

    def _rank_evidence_sentences(
        self,
        chunks: list[RetrievedChunk],
        query_terms: set[str],
    ) -> list[tuple[float, str, RetrievedChunk]]:
        ranked: list[tuple[float, str, RetrievedChunk]] = []
        seen: set[str] = set()
        reasoning_terms = {
            "because",
            "therefore",
            "however",
            "whereas",
            "while",
            "coordination",
            "coordinated",
            "consistent",
            "strategic",
            "impact",
            "effect",
            "important",
            "requires",
            "enables",
            "aims",
        }
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text.strip()):
                cleaned = self._clean_pdf_text(sentence)
                if len(cleaned) < 40:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                sentence_terms = self._query_terms(cleaned)
                overlap = len(query_terms & sentence_terms)
                reasoning_bonus = sum(1 for term in reasoning_terms if term in key)
                score = overlap + (0.5 * reasoning_bonus) + min(chunk.score, 1.0)
                if overlap:
                    ranked.append((score, cleaned[:550], chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _citation(self, chunk: RetrievedChunk) -> str:
        return f"[{chunk.metadata.get('source', 'unknown source')}#{chunk.chunk_id}]"

    def _join_citations(self, *citations: str) -> str:
        seen: set[str] = set()
        ordered: list[str] = []
        for citation in citations:
            if citation and citation not in seen:
                seen.add(citation)
                ordered.append(citation)
        return " ".join(ordered)

    def _definition_entity(self, query: str) -> str:
        match = re.search(r"\b(?:what|who)\s+(?:is|are|was|were)\s+(.+?)[?!.]*$", query.strip(), re.IGNORECASE)
        if not match:
            return ""
        entity = re.sub(r"\b(?:the|a|an)\b", "", match.group(1), flags=re.IGNORECASE)
        entity = re.sub(r"\b(?:in|of|from|using|with)\s+[a-zA-Z][a-zA-Z0-9_ -]*$", "", entity, flags=re.IGNORECASE)
        return " ".join(entity.split()).lower()

    def _definition_answer(self, entity: str, chunks: list[RetrievedChunk]) -> str:
        if not entity:
            return ""
        escaped = re.escape(entity)
        patterns = [
            re.compile(rf"\b{escaped}\s+(?:is|are|refers to|is defined as)\b.*?(?:\.|$)", re.IGNORECASE),
            re.compile(rf"\b{escaped}\s+(?:a|an|the|built-in|standard|mutable|immutable)\b.*?(?:\.|$)", re.IGNORECASE),
        ]
        candidates: list[tuple[float, str, RetrievedChunk]] = []
        for chunk in chunks:
            for pattern in patterns:
                match = pattern.search(chunk.text)
                if match:
                    sentence = self._clean_pdf_text(match.group(0))
                    sentence = self._normalize_glossary_definition(entity, sentence)
                    candidates.append((self._definition_sentence_score(sentence), sentence, chunk))
                    break
        if not candidates:
            return ""
        _, sentence, chunk = max(candidates, key=lambda item: item[0])
        label = chunk.metadata.get("source", "unknown source")
        return f"**Answer**\n{sentence} [{label}#{chunk.chunk_id}]"

    def _definition_sentence_score(self, sentence: str) -> float:
        lowered = sentence.lower()
        score = 0.0
        strong_definition_terms = (
            "built on",
            "foundation",
            "capabilities",
            "memory",
            "planning",
            "tool-use",
            "tools",
            "system",
            "framework",
            "designed to",
            "consist",
            "built-in",
            "sequence",
            "array",
            "mutable",
        )
        weak_context_terms = (
            "no longer",
            "reshaping",
            "case study",
            "paper",
            "profile",
            "platform",
            "visibility",
        )
        score += sum(1.0 for term in strong_definition_terms if term in lowered)
        score -= sum(0.75 for term in weak_context_terms if term in lowered)
        score += max(0.0, 2.0 - (len(sentence) / 220))
        return score

    def _clean_pdf_text(self, text: str) -> str:
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        return " ".join(text.split())

    def _normalize_glossary_definition(self, entity: str, sentence: str) -> str:
        match = re.match(rf"({re.escape(entity)})\s+([Aa]n?|[Tt]he)\s+(.+)", sentence, flags=re.IGNORECASE)
        if not match:
            return sentence
        subject = match.group(1)
        article = match.group(2).lower()
        rest = match.group(3)
        return f"{article.capitalize()} {subject} is {article} {rest}"
