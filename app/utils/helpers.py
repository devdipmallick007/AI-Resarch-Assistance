from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class TextChunk:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float
    chunk_id: str
