"""Dense retriever using FAISS vector index."""

from __future__ import annotations

import time
import numpy as np

from ragbench.ingest import Chunk
from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """FAISS-backed dense vector retriever."""

    def __init__(
        self,
        top_k: int = 10,
        index_type: str = "faiss_flat",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs,
    ):
        super().__init__(top_k=top_k)
        self.index_type = index_type
        self.embedding_model_name = embedding_model
        self._index = None
        self._chunks: list[Chunk] = []
        self._encoder = None

    @property
    def name(self) -> str:
        return f"dense_{self.index_type}"

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def index(self, chunks: list[Chunk]) -> None:
        import faiss

        self._chunks = chunks

        # Stack embeddings — compute if missing
        embeddings = []
        for c in chunks:
            if c.embedding is not None:
                embeddings.append(c.embedding)
            else:
                emb = self.encoder.encode(c.text, normalize_embeddings=True)
                c.embedding = emb
                embeddings.append(emb)

        matrix = np.stack(embeddings).astype("float32")
        dim = matrix.shape[1]

        # Build FAISS index
        if self.index_type == "faiss_flat":
            self._index = faiss.IndexFlatIP(dim)
        elif self.index_type == "faiss_ivf":
            nlist = min(100, len(chunks) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantizer, dim, max(1, nlist))
            self._index.train(matrix)
        elif self.index_type == "faiss_hnsw":
            self._index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(matrix)
        log.info(
            f"FAISS index built ({self.index_type}): "
            f"[bold]{len(chunks)}[/] vectors, dim={dim}"
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if self._index is None:
            raise RuntimeError("Index not built. Call .index() first.")

        k = top_k or self.top_k
        start = time.perf_counter()

        q_emb = self.encoder.encode(query, normalize_embeddings=True)
        q_emb = np.array([q_emb]).astype("float32")

        scores, indices = self._index.search(q_emb, k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        top_chunks = [self._chunks[i] for i in indices if i >= 0]
        top_scores = [s for s, i in zip(scores, indices) if i >= 0]

        latency = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            retrieved=top_chunks,
            scores=top_scores,
            latency_ms=latency,
        )
