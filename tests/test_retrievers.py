"""Tests for retriever implementations."""

import pytest
import numpy as np
from ragbench.ingest import Chunk
from ragbench.retrievers import create_retriever
from ragbench.retrievers.bm25_retriever import BM25Retriever
from ragbench.retrievers.dense_retriever import DenseRetriever
from ragbench.retrievers.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_chunks():
    texts = [
        "Retrieval augmented generation combines search with language models",
        "BM25 is a bag-of-words retrieval function based on term frequency",
        "Dense retrieval uses neural embeddings for semantic similarity search",
        "Hybrid retrieval combines sparse and dense methods using rank fusion",
        "Knowledge graphs store entities and relationships for structured retrieval",
    ]
    return [
        Chunk(id=f"chunk_{i}", text=t, embedding=np.random.randn(384).astype("float32"))
        for i, t in enumerate(texts)
    ]


class TestBM25Retriever:
    def test_index_and_retrieve(self, sample_chunks):
        r = BM25Retriever(top_k=3)
        r.index(sample_chunks)
        result = r.retrieve("BM25 term frequency retrieval")
        assert len(result.retrieved) == 3
        assert result.latency_ms > 0
        # BM25 chunk should be ranked high
        ids = [c.id for c in result.retrieved]
        assert "chunk_1" in ids

    def test_empty_query(self, sample_chunks):
        r = BM25Retriever(top_k=2)
        r.index(sample_chunks)
        result = r.retrieve("")
        assert len(result.retrieved) == 2

    def test_top_k_override(self, sample_chunks):
        r = BM25Retriever(top_k=5)
        r.index(sample_chunks)
        result = r.retrieve("retrieval", top_k=2)
        assert len(result.retrieved) == 2


class TestRetrieverFactory:
    def test_create_bm25(self):
        r = create_retriever("bm25", top_k=5)
        assert isinstance(r, BM25Retriever)
        assert r.top_k == 5

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown retriever"):
            create_retriever("unknown_type")


class TestRetrievalResult:
    def test_doc_ids(self, sample_chunks):
        r = BM25Retriever(top_k=3)
        r.index(sample_chunks)
        result = r.retrieve("retrieval")
        assert len(result.doc_ids) == 3
        assert all(isinstance(d, str) for d in result.doc_ids)
