"""BM25 sparse retriever using rank_bm25."""

from __future__ import annotations

import time
from rank_bm25 import BM25Okapi

from ragbench.ingest import Chunk
from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """Okapi BM25 sparse keyword retriever."""

    def __init__(self, top_k: int = 10, **kwargs):
        super().__init__(top_k=top_k)
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return "bm25"

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self._index = BM25Okapi(tokenized)
        log.info(f"BM25 index built: [bold]{len(chunks)}[/] chunks")

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if self._index is None:
            raise RuntimeError("Index not built. Call .index() first.")

        k = top_k or self.top_k
        start = time.perf_counter()

        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:k]
        top_scores = scores[top_indices].tolist()
        top_chunks = [self._chunks[i] for i in top_indices]

        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            retrieved=top_chunks,
            scores=top_scores,
            latency_ms=latency,
        )
