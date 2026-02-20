"""Tests for evaluation metrics."""

import pytest
from ragbench.ingest import Chunk
from ragbench.evaluators.retrieval_metrics import RetrievalEvaluator, EvalQuery
from ragbench.retrievers.base import RetrievalResult


@pytest.fixture
def perfect_retrieval():
    """Retrieval where all relevant docs are in top-k."""
    chunks = [Chunk(id=f"doc_{i}", text=f"text {i}") for i in range(5)]
    result = RetrievalResult(query="test", retrieved=chunks[:3], scores=[0.9, 0.8, 0.7])
    gt = EvalQuery(query="test", relevant_ids=["doc_0", "doc_1", "doc_2"])
    return result, gt


@pytest.fixture
def partial_retrieval():
    """Retrieval where only some relevant docs are found."""
    chunks = [Chunk(id=f"doc_{i}", text=f"text {i}") for i in range(5)]
    result = RetrievalResult(query="test", retrieved=chunks[:5], scores=[0.9, 0.8, 0.7, 0.6, 0.5])
    gt = EvalQuery(query="test", relevant_ids=["doc_0", "doc_3"])
    return result, gt


class TestPrecision:
    def test_perfect_precision(self, perfect_retrieval):
        result, gt = perfect_retrieval
        ev = RetrievalEvaluator(metrics=["precision"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["precision@3"].value == 1.0

    def test_partial_precision(self, partial_retrieval):
        result, gt = partial_retrieval
        ev = RetrievalEvaluator(metrics=["precision"], k_values=[5])
        metrics = ev.evaluate([result], [gt])
        assert metrics["precision@5"].value == pytest.approx(0.4)


class TestRecall:
    def test_perfect_recall(self, perfect_retrieval):
        result, gt = perfect_retrieval
        ev = RetrievalEvaluator(metrics=["recall"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["recall@3"].value == 1.0


class TestNDCG:
    def test_perfect_ndcg(self, perfect_retrieval):
        result, gt = perfect_retrieval
        ev = RetrievalEvaluator(metrics=["ndcg"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["ndcg@3"].value == pytest.approx(1.0)


class TestMRR:
    def test_first_position(self, perfect_retrieval):
        result, gt = perfect_retrieval
        ev = RetrievalEvaluator(metrics=["mrr"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["mrr@3"].value == 1.0

    def test_second_position(self):
        chunks = [Chunk(id=f"doc_{i}", text=f"text {i}") for i in range(3)]
        result = RetrievalResult(query="test", retrieved=chunks, scores=[0.9, 0.8, 0.7])
        gt = EvalQuery(query="test", relevant_ids=["doc_1"])
        ev = RetrievalEvaluator(metrics=["mrr"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["mrr@3"].value == pytest.approx(0.5)


class TestHitRate:
    def test_hit(self, perfect_retrieval):
        result, gt = perfect_retrieval
        ev = RetrievalEvaluator(metrics=["hit_rate"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["hit_rate@3"].value == 1.0

    def test_miss(self):
        chunks = [Chunk(id=f"doc_{i}", text=f"text {i}") for i in range(3)]
        result = RetrievalResult(query="test", retrieved=chunks, scores=[0.9, 0.8, 0.7])
        gt = EvalQuery(query="test", relevant_ids=["doc_99"])
        ev = RetrievalEvaluator(metrics=["hit_rate"], k_values=[3])
        metrics = ev.evaluate([result], [gt])
        assert metrics["hit_rate@3"].value == 0.0
