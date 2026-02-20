"""Hybrid retriever — Reciprocal Rank Fusion of BM25 + Dense."""

from __future__ import annotations

import time
from collections import defaultdict

from ragbench.ingest import Chunk
from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.retrievers.bm25_retriever import BM25Retriever
from ragbench.retrievers.dense_retriever import DenseRetriever
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion (RRF) of sparse + dense retrievers."""

    def __init__(
        self,
        top_k: int = 10,
        alpha: float = 0.6,
        rrf_k: int = 60,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "faiss_flat",
        **kwargs,
    ):
        super().__init__(top_k=top_k)
        self.alpha = alpha  # weight for dense (1-alpha for sparse)
        self.rrf_k = rrf_k
        self.sparse = BM25Retriever(top_k=top_k * 2)
        self.dense = DenseRetriever(
            top_k=top_k * 2,
            index_type=index_type,
            embedding_model=embedding_model,
        )

    @property
    def name(self) -> str:
        return f"hybrid_alpha{self.alpha}"

    def index(self, chunks: list[Chunk]) -> None:
        self.sparse.index(chunks)
        self.dense.index(chunks)
        self._chunks_map = {c.id: c for c in chunks}
        log.info(f"Hybrid index built (alpha={self.alpha})")

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        start = time.perf_counter()

        sparse_result = self.sparse.retrieve(query, top_k=k * 2)
        dense_result = self.dense.retrieve(query, top_k=k * 2)

        # RRF fusion
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, chunk in enumerate(sparse_result.retrieved):
            rrf_scores[chunk.id] += (1 - self.alpha) / (self.rrf_k + rank + 1)

        for rank, chunk in enumerate(dense_result.retrieved):
            rrf_scores[chunk.id] += self.alpha / (self.rrf_k + rank + 1)

        # Sort by fused score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

        top_chunks = [self._chunks_map[cid] for cid in sorted_ids if cid in self._chunks_map]
        top_scores = [rrf_scores[cid] for cid in sorted_ids if cid in self._chunks_map]

        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            retrieved=top_chunks,
            scores=top_scores,
            latency_ms=latency,
            metadata={
                "sparse_latency_ms": sparse_result.latency_ms,
                "dense_latency_ms": dense_result.latency_ms,
            },
        )
