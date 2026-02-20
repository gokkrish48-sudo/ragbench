import numpy as np
from typing import List, Set

class RetrievalMetrics:
    """
    Evaluation suite for Information Retrieval (IR) performance.
    """
    
    @staticmethod
    def precision_at_k(retrieved: List[str], ground_truth: Set[str], k: int) -> float:
        """
        Precision@K: (Relevant Docs in top K) / K
        """
        if k <= 0: return 0.0
        top_k = retrieved[:k]
        relevant_hits = len([doc for doc in top_k if doc in ground_truth])
        return relevant_hits / k

    @staticmethod
    def recall_at_k(retrieved: List[str], ground_truth: Set[Set], k: int) -> float:
        """
        Recall@K: (Relevant Docs in top K) / (Total Relevant Docs)
        """
        if not ground_truth: return 0.0
        top_k = retrieved[:k]
        relevant_hits = len([doc for doc in top_k if doc in ground_truth])
        return relevant_hits / len(ground_truth)

    @staticmethod
    def dcg_at_k(r: List[int], k: int) -> float:
        """
        Discounted Cumulative Gain at K.
        r: list of relevancy scores (usually binary 0 or 1)
        """
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    @staticmethod
    def ndcg_at_k(retrieved: List[str], ground_truth: Set[str], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at K.
        """
        relevance = [1 if doc in ground_truth else 0 for doc in retrieved[:k]]
        idcg = RetrievalMetrics.dcg_at_k(sorted(relevance, reverse=True), k)
        if not idcg:
            return 0.
        return RetrievalMetrics.dcg_at_k(relevance, k) / idcg

    @staticmethod
    def mrr(retrieved: List[str], ground_truth: Set[str]) -> float:
        """
        Mean Reciprocal Rank.
        """
        for i, doc in enumerate(retrieved, start=1):
            if doc in ground_truth:
                return 1.0 / i
        return 0.0
