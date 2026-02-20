"""Integration test for the full pipeline."""

import pytest
from ragbench.utils.config import RAGBenchConfig, IngestConfig, RetrieverConfig, EvalConfig


class TestConfig:
    def test_defaults(self):
        config = RAGBenchConfig()
        assert config.ingest.chunk_size == 512
        assert len(config.retrievers) == 1
        assert config.evaluate.metrics == ["precision", "recall", "ndcg", "mrr"]

    def test_custom_config(self):
        config = RAGBenchConfig(
            ingest=IngestConfig(chunk_size=256),
            retrievers=[
                RetrieverConfig(type="bm25", top_k=5),
                RetrieverConfig(type="dense", top_k=10),
            ],
        )
        assert config.ingest.chunk_size == 256
        assert len(config.retrievers) == 2
        assert config.retrievers[0].type == "bm25"
        assert config.retrievers[1].top_k == 10
