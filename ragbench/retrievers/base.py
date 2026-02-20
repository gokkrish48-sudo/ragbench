"""Abstract base retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragbench.ingest import Chunk


@dataclass
class RetrievalResult:
    query: str
    retrieved: list[Chunk]
    scores: list[float]
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return [c.id for c in self.retrieved]


class BaseRetriever(ABC):
    """Abstract retriever — all retriever backends implement this."""

    def __init__(self, top_k: int = 10, **kwargs):
        self.top_k = top_k

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build index from chunks."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve top-k chunks for a query."""
        ...

    def batch_retrieve(self, queries: list[str], top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve for multiple queries."""
        return [self.retrieve(q, top_k) for q in queries]

    @property
    @abstractmethod
    def name(self) -> str:
        ...
