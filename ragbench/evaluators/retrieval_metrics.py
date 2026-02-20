"""Retrieval quality metrics: Precision@K, Recall@K, nDCG@K, MRR, Hit Rate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ragbench.retrievers.base import RetrievalResult


@dataclass
class EvalQuery:
    query: str
    relevant_ids: list[str]  # ground-truth relevant doc IDs


@dataclass
class MetricResult:
    metric: str
    k: int
    value: float
    per_query: list[float] = field(default_factory=list)


class RetrievalEvaluator:
    """Compute standard IR metrics over retrieval results."""

    SUPPORTED = {"precision", "recall", "ndcg", "mrr", "hit_rate"}

    def __init__(self, metrics: list[str] | None = None, k_values: list[int] | None = None):
        self.metrics = metrics or ["precision", "recall", "ndcg", "mrr"]
        self.k_values = k_values or [1, 3, 5, 10]

        invalid = set(self.metrics) - self.SUPPORTED
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}. Supported: {self.SUPPORTED}")

    def evaluate(
        self, results: list[RetrievalResult], ground_truth: list[EvalQuery]
    ) -> dict[str, MetricResult]:
        """Evaluate retrieval results against ground truth."""
        assert len(results) == len(ground_truth), "Results and ground truth must align"

        all_metrics: dict[str, MetricResult] = {}

        for metric_name in self.metrics:
            for k in self.k_values:
                per_query = []
                for result, gt in zip(results, ground_truth):
                    retrieved_ids = [c.id for c in result.retrieved[:k]]
                    relevant = set(gt.relevant_ids)

                    if metric_name == "precision":
                        score = self._precision_at_k(retrieved_ids, relevant)
                    elif metric_name == "recall":
                        score = self._recall_at_k(retrieved_ids, relevant)
                    elif metric_name == "ndcg":
                        score = self._ndcg_at_k(retrieved_ids, relevant, k)
                    elif metric_name == "mrr":
                        score = self._mrr(retrieved_ids, relevant)
                    elif metric_name == "hit_rate":
                        score = self._hit_rate(retrieved_ids, relevant)
                    else:
                        score = 0.0

                    per_query.append(score)

                key = f"{metric_name}@{k}"
                all_metrics[key] = MetricResult(
                    metric=metric_name,
                    k=k,
                    value=float(np.mean(per_query)),
                    per_query=per_query,
                )

        return all_metrics

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
        if not retrieved:
            return 0.0
        hits = sum(1 for r in retrieved if r in relevant)
        return hits / len(retrieved)

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for r in retrieved if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        ideal_rels = sorted([1.0] * min(len(relevant), k), reverse=True)
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _mrr(retrieved: list[str], relevant: set[str]) -> float:
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _hit_rate(retrieved: list[str], relevant: set[str]) -> float:
        return 1.0 if any(r in relevant for r in retrieved) else 0.0

    def summary_table(self, metrics: dict[str, MetricResult]) -> str:
        """Format metrics as a printable table."""
        lines = [f"{'Metric':<20} {'Value':>10} {'Std':>10}"]
        lines.append("-" * 42)
        for key, m in sorted(metrics.items()):
            std = float(np.std(m.per_query)) if m.per_query else 0.0
            lines.append(f"{key:<20} {m.value:>10.4f} {std:>10.4f}")
        return "\n".join(lines)
