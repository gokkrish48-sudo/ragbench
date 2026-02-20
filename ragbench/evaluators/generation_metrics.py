"""Generation quality metrics: faithfulness, relevance, hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GenerationEvalResult:
    query: str
    answer: str
    context: str
    faithfulness: float  # 0-1: is answer grounded in context?
    relevance: float     # 0-1: does answer address the query?
    hallucination_rate: float  # 0-1: fraction of unsupported claims


class GenerationEvaluator:
    """Evaluate generation quality using overlap heuristics and optional LLM judge."""

    def __init__(self, judge_model: str | None = None):
        self.judge_model = judge_model

    def evaluate(
        self,
        queries: list[str],
        answers: list[str],
        contexts: list[str],
    ) -> list[GenerationEvalResult]:
        results = []
        for q, a, c in zip(queries, answers, contexts):
            if self.judge_model:
                faith, rel, hall = self._llm_judge(q, a, c)
            else:
                faith = self._token_overlap_faithfulness(a, c)
                rel = self._token_overlap_relevance(q, a)
                hall = 1.0 - faith

            results.append(GenerationEvalResult(
                query=q, answer=a, context=c,
                faithfulness=faith, relevance=rel, hallucination_rate=hall,
            ))
        return results

    def aggregate(self, results: list[GenerationEvalResult]) -> dict[str, float]:
        return {
            "faithfulness_mean": float(np.mean([r.faithfulness for r in results])),
            "relevance_mean": float(np.mean([r.relevance for r in results])),
            "hallucination_rate_mean": float(np.mean([r.hallucination_rate for r in results])),
            "faithfulness_std": float(np.std([r.faithfulness for r in results])),
        }

    @staticmethod
    def _token_overlap_faithfulness(answer: str, context: str) -> float:
        """Simple token-level overlap as faithfulness proxy."""
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        if not answer_tokens:
            return 0.0
        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    @staticmethod
    def _token_overlap_relevance(query: str, answer: str) -> float:
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())
        if not query_tokens:
            return 0.0
        overlap = query_tokens & answer_tokens
        return len(overlap) / len(query_tokens)

    def _llm_judge(self, query: str, answer: str, context: str) -> tuple[float, float, float]:
        """Use LLM to judge generation quality. Returns (faithfulness, relevance, hallucination)."""
        # Placeholder — integrate with actual LLM API
        prompt = f"""Rate the following answer on three dimensions (0.0 to 1.0):

Query: {query}
Context: {context[:2000]}
Answer: {answer}

Return JSON: {{"faithfulness": float, "relevance": float, "hallucination_rate": float}}"""

        # In production, call the judge_model API here
        # For now, fall back to heuristic
        faith = self._token_overlap_faithfulness(answer, context)
        rel = self._token_overlap_relevance(query, answer)
        return faith, rel, 1.0 - faith
