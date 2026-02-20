"""YAML config loader with validation and defaults."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IngestConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256


@dataclass
class RetrieverConfig:
    type: str = "bm25"
    top_k: int = 10
    index: str = "faiss_flat"
    alpha: float = 0.6  # hybrid weight
    params: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["precision", "recall", "ndcg", "mrr"])
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    judge_model: str | None = None


@dataclass
class OptimizeConfig:
    method: str = "bayesian"
    n_trials: int = 50
    metric: str = "ndcg@5"
    search_space: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str = "ragbench-run"
    tracking_uri: str = "mlruns/"


@dataclass
class RAGBenchConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrievers: list[RetrieverConfig] = field(default_factory=lambda: [RetrieverConfig()])
    evaluate: EvalConfig = field(default_factory=EvalConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGBenchConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()

        if "experiment" in raw:
            config.experiment = ExperimentConfig(**raw["experiment"])

        if "ingest" in raw:
            config.ingest = IngestConfig(**raw["ingest"])

        if "retrievers" in raw:
            config.retrievers = [RetrieverConfig(**r) for r in raw["retrievers"]]

        if "evaluate" in raw:
            config.evaluate = EvalConfig(**raw["evaluate"])

        if "optimize" in raw:
            config.optimize = OptimizeConfig(**raw["optimize"])

        return config

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)
