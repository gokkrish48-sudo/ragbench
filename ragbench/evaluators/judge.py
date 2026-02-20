"""LLM-as-Judge for retrieval and generation quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeVerdict:
    query: str
    score: float  # 0-1
    reasoning: str
    dimension: str  # faithfulness | relevance | coherence


class LLMJudge:
    """Use an LLM to judge retrieval/generation quality."""

    JUDGE_PROMPT = """You are an expert evaluator for RAG systems.

Given a query, retrieved context, and generated answer, rate the {dimension} on a scale of 0.0 to 1.0.

Query: {query}
Context: {context}
Answer: {answer}

Rate {dimension} (0.0 = completely fails, 1.0 = perfect).
Respond with ONLY a JSON object: {{"score": float, "reasoning": "brief explanation"}}"""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def judge(
        self,
        query: str,
        context: str,
        answer: str,
        dimension: str = "faithfulness",
    ) -> JudgeVerdict:
        """Judge a single query-context-answer triple."""
        prompt = self.JUDGE_PROMPT.format(
            query=query,
            context=context[:3000],
            answer=answer,
            dimension=dimension,
        )

        # In production, call LLM API here
        # Placeholder: heuristic scoring
        score = self._heuristic_score(query, context, answer, dimension)

        return JudgeVerdict(
            query=query,
            score=score,
            reasoning="Heuristic score (LLM judge not configured)",
            dimension=dimension,
        )

    def batch_judge(
        self,
        queries: list[str],
        contexts: list[str],
        answers: list[str],
        dimensions: list[str] | None = None,
    ) -> list[JudgeVerdict]:
        dims = dimensions or ["faithfulness"] * len(queries)
        return [
            self.judge(q, c, a, d)
            for q, c, a, d in zip(queries, contexts, answers, dims)
        ]

    @staticmethod
    def _heuristic_score(query: str, context: str, answer: str, dimension: str) -> float:
        a_tokens = set(answer.lower().split())
        c_tokens = set(context.lower().split())
        q_tokens = set(query.lower().split())

        if dimension == "faithfulness":
            return len(a_tokens & c_tokens) / max(len(a_tokens), 1)
        elif dimension == "relevance":
            return len(a_tokens & q_tokens) / max(len(q_tokens), 1)
        elif dimension == "coherence":
            return min(1.0, len(answer.split()) / 50)
        return 0.5
