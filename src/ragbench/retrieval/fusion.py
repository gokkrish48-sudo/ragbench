import numpy as np
from typing import List, Dict, Any

class FusionEngine:
    """
    Implements advanced retrieval fusion strategies for hybrid search.
    """
    
    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[Dict[str, float]], 
        k: int = 60
    ) -> Dict[str, float]:
        """
        Computes the RRF score for a list of retrieval results.
        RRF(d) = sum_{r in results} 1 / (k + rank(r, d))
        """
        combined_scores = {}
        for results in results_list:
            # Sort results by score to get ranks
            sorted_docs = sorted(results.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_docs, start=1):
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = 0.0
                combined_scores[doc_id] += 1 / (k + rank)
        
        # Sort by final RRF score
        return dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def weighted_fusion(
        results_list: List[Dict[str, float]], 
        weights: List[float]
    ) -> Dict[str, float]:
        """
        Linearly combines scores using provided weights.
        Assumes scores are already calibrated/normalized.
        """
        if len(results_list) != len(weights):
            raise ValueError("Number of results must match number of weights.")
            
        combined_scores = {}
        for results, weight in zip(results_list, weights):
            for doc_id, score in results.items():
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = 0.0
                combined_scores[doc_id] += score * weight
                
        return dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def z_score_normalize(scores: Dict[str, float]) -> Dict[str, float]:
        """
        Applies Z-score normalization to retrieval scores.
        """
        vals = np.array(list(scores.values()))
        mean = np.mean(vals)
        std = np.std(vals) if np.std(vals) > 0 else 1.0
        
        normalized = {k: (v - mean) / std for k, v in scores.items()}
        return normalized
