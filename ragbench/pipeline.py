"""Core RAGBench pipeline — orchestrates ingest, retrieve, evaluate, optimize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.table import Table

from ragbench.ingest import DocumentIngestor, Chunk
from ragbench.retrievers import create_retriever, BaseRetriever
from ragbench.evaluators import RetrievalEvaluator, EvalQuery
from ragbench.evaluators.generation_metrics import GenerationEvaluator
from ragbench.evaluators.latency_profiler import ScaleProfiler, LatencyProfiler
from ragbench.generators import LLMGenerator
from ragbench.optimizers import BayesianOptimizer, GridSearchOptimizer, AblationRunner
from ragbench.utils.config import RAGBenchConfig
from ragbench.utils.logging import get_logger, timer, console
from ragbench.utils.tracking import ExperimentTracker

log = get_logger(__name__)


class BenchmarkResults:
    """Container for benchmark results across multiple retrievers."""

    def __init__(self):
        self.retriever_metrics: dict[str, dict] = {}
        self.latency_reports: dict[str, Any] = {}
        self.generation_metrics: dict[str, Any] = {}

    def add(self, retriever_name: str, metrics: dict, latency: Any = None):
        self.retriever_metrics[retriever_name] = metrics
        if latency:
            self.latency_reports[retriever_name] = latency

    def summary(self):
        table = Table(title="RAGBench Results")
        table.add_column("Retriever", style="bold cyan")
        table.add_column("P@5", justify="right")
        table.add_column("R@5", justify="right")
        table.add_column("nDCG@5", justify="right")
        table.add_column("MRR@5", justify="right")
        table.add_column("p99 (ms)", justify="right")

        for name, metrics in self.retriever_metrics.items():
            p5 = metrics.get("precision@5", metrics.get("precision@3", None))
            r5 = metrics.get("recall@5", metrics.get("recall@3", None))
            n5 = metrics.get("ndcg@5", metrics.get("ndcg@3", None))
            m5 = metrics.get("mrr@5", metrics.get("mrr@3", None))
            lat = self.latency_reports.get(name)
            p99 = f"{lat.p99_ms:.1f}" if lat else "-"

            table.add_row(
                name,
                f"{p5.value:.4f}" if p5 else "-",
                f"{r5.value:.4f}" if r5 else "-",
                f"{n5.value:.4f}" if n5 else "-",
                f"{m5.value:.4f}" if m5 else "-",
                p99,
            )

        console.print(table)


class RAGBenchPipeline:
    """Main pipeline: ingest → retrieve → evaluate → optimize."""

    def __init__(self, config: RAGBenchConfig):
        self.config = config
        self.ingestor = DocumentIngestor(
            chunk_size=config.ingest.chunk_size,
            chunk_overlap=config.ingest.chunk_overlap,
            embedding_model=config.ingest.embedding_model,
            batch_size=config.ingest.batch_size,
        )
        self.retrievers: list[BaseRetriever] = []
        self.chunks: list[Chunk] = []
        self.tracker = ExperimentTracker(
            config.experiment.name, config.experiment.tracking_uri
        )

        # Initialize retrievers
        for rc in config.retrievers:
            retriever = create_retriever(
                type=rc.type,
                top_k=rc.top_k,
                index_type=rc.index,
                alpha=rc.alpha,
                embedding_model=config.ingest.embedding_model,
                **rc.params,
            )
            self.retrievers.append(retriever)

        log.info(
            f"Pipeline initialized: [bold]{len(self.retrievers)}[/] retrievers, "
            f"chunk_size={config.ingest.chunk_size}"
        )

    @classmethod
    def from_config(cls, path: str) -> RAGBenchPipeline:
        config = RAGBenchConfig.from_yaml(path)
        return cls(config)

    def ingest(self, path: str, text_field: str = "text") -> list[Chunk]:
        """Ingest documents and build all retriever indices."""
        with timer("Ingestion", log):
            self.chunks = self.ingestor.ingest(path, text_field=text_field)

        with timer("Indexing", log):
            for retriever in self.retrievers:
                retriever.index(self.chunks)

        return self.chunks

    def evaluate(
        self,
        queries: str | list[EvalQuery],
        profile_latency: bool = True,
    ) -> BenchmarkResults:
        """Run evaluation across all retrievers."""
        # Load queries
        if isinstance(queries, str):
            queries = self._load_eval_queries(queries)

        self.tracker.start_run(run_name=self.config.experiment.name)
        self.tracker.log_params(self.config.to_dict())

        evaluator = RetrievalEvaluator(
            metrics=self.config.evaluate.metrics,
            k_values=self.config.evaluate.k_values,
        )

        results = BenchmarkResults()

        for retriever in self.retrievers:
            log.info(f"Evaluating [bold]{retriever.name}[/]...")

            # Retrieve
            query_texts = [q.query for q in queries]
            retrieval_results = retriever.batch_retrieve(query_texts)

            # Evaluate metrics
            metrics = evaluator.evaluate(retrieval_results, queries)

            # Latency profiling
            latency_report = None
            if profile_latency:
                profiler = LatencyProfiler(retriever)
                latency_report = profiler.profile(query_texts)

            results.add(retriever.name, metrics, latency_report)

            # Log to MLflow
            flat_metrics = {k: m.value for k, m in metrics.items()}
            self.tracker.log_metrics(flat_metrics)

            log.info(f"  {retriever.name}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in list(flat_metrics.items())[:4]
            ))

        self.tracker.end_run()
        return results

    def optimize(
        self,
        queries: str | list[EvalQuery],
        metric: str = "ndcg@5",
        n_trials: int = 50,
        method: str = "bayesian",
    ) -> dict[str, Any]:
        """Optimize hyperparameters for the first retriever."""
        if isinstance(queries, str):
            queries = self._load_eval_queries(queries)

        evaluator = RetrievalEvaluator(
            metrics=self.config.evaluate.metrics,
            k_values=self.config.evaluate.k_values,
        )

        def objective(params: dict) -> float:
            # Rebuild ingestor with new params
            ingestor = DocumentIngestor(
                chunk_size=params.get("chunk_size", self.config.ingest.chunk_size),
                chunk_overlap=params.get("chunk_overlap", self.config.ingest.chunk_overlap),
                embedding_model=self.config.ingest.embedding_model,
            )
            # Re-chunk existing docs (fast, no re-embedding if not needed)
            # For simplicity, use existing chunks and just vary top_k/alpha
            retriever = self.retrievers[0]
            top_k = params.get("top_k", retriever.top_k)

            query_texts = [q.query for q in queries]
            results = retriever.batch_retrieve(query_texts, top_k=top_k)
            metrics = evaluator.evaluate(results, queries)
            return metrics[metric].value

        search_space = self.config.optimize.search_space or {
            "top_k": [3, 5, 10, 20],
        }

        if method == "bayesian":
            optimizer = BayesianOptimizer(
                search_space=search_space,
                objective_fn=objective,
                metric=metric,
                n_trials=n_trials,
            )
        else:
            optimizer = GridSearchOptimizer(
                search_space=search_space,
                objective_fn=objective,
                metric=metric,
            )

        return optimizer.optimize()

    @staticmethod
    def _load_eval_queries(path: str) -> list[EvalQuery]:
        """Load evaluation queries from JSON file."""
        with open(path) as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append(EvalQuery(
                query=item["query"],
                relevant_ids=item.get("relevant_ids", item.get("relevant_doc_ids", [])),
            ))
        return queries
