"""Retriever registry and factory."""

from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.retrievers.bm25_retriever import BM25Retriever
from ragbench.retrievers.dense_retriever import DenseRetriever
from ragbench.retrievers.hybrid_retriever import HybridRetriever
from ragbench.retrievers.graph_retriever import GraphRetriever

RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
    "graph": GraphRetriever,
}


def create_retriever(type: str, **kwargs) -> BaseRetriever:
    """Factory to create a retriever by type name."""
    if type not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown retriever: {type}. Available: {list(RETRIEVER_REGISTRY)}")
    return RETRIEVER_REGISTRY[type](**kwargs)
