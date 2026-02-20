"""LLM generation with multi-provider routing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ragbench.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class GenerationResult:
    query: str
    answer: str
    context_used: str
    model: str
    latency_ms: float
    token_count: int = 0


class LLMGenerator:
    """Generate answers from retrieved context using configurable LLM providers."""

    RAG_PROMPT = """Answer the question based ONLY on the provided context. 
If the context doesn't contain enough information, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        api_key: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, query: str, context: str) -> GenerationResult:
        """Generate an answer for a query given retrieved context."""
        prompt = self.RAG_PROMPT.format(context=context[:4000], query=query)

        start = time.perf_counter()

        if self.provider == "anthropic":
            answer = self._call_anthropic(prompt)
        elif self.provider == "openai":
            answer = self._call_openai(prompt)
        else:
            answer = self._mock_generate(query, context)

        latency = (time.perf_counter() - start) * 1000

        return GenerationResult(
            query=query,
            answer=answer,
            context_used=context[:1000],
            model=self.model,
            latency_ms=latency,
            token_count=len(answer.split()),
        )

    def batch_generate(
        self, queries: list[str], contexts: list[str]
    ) -> list[GenerationResult]:
        return [self.generate(q, c) for q, c in zip(queries, contexts)]

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            log.warning(f"Anthropic API error: {e}")
            return self._mock_generate(prompt, "")

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            log.warning(f"OpenAI API error: {e}")
            return self._mock_generate(prompt, "")

    @staticmethod
    def _mock_generate(query: str, context: str) -> str:
        """Extractive fallback: return first sentence of context."""
        if context:
            sentences = context.split(".")
            return sentences[0].strip() + "." if sentences else "No context available."
        return "No context available."
