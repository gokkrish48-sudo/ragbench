"""Latency and throughput profiler for scale testing."""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ragbench.retrievers.base import BaseRetriever
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LatencyReport:
    retriever_name: str
    num_queries: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float
    latencies_ms: list[float] = field(default_factory=list)


@dataclass
class ScaleReport:
    reports: list[LatencyReport] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'Retriever':<25} {'Queries':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'QPS':>8}"
        ]
        lines.append("-" * 75)
        for r in self.reports:
            lines.append(
                f"{r.retriever_name:<25} {r.num_queries:>8} "
                f"{r.p50_ms:>7.1f}ms {r.p95_ms:>7.1f}ms {r.p99_ms:>7.1f}ms "
                f"{r.throughput_qps:>7.1f}"
            )
        return "\n".join(lines)


class LatencyProfiler:
    """Profile retrieval latency and throughput."""

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    def profile(
        self,
        queries: list[str],
        warmup: int = 5,
        concurrency: int = 1,
    ) -> LatencyReport:
        # Warmup
        for q in queries[:warmup]:
            self.retriever.retrieve(q)

        latencies = []

        if concurrency <= 1:
            start_all = time.perf_counter()
            for q in queries:
                result = self.retriever.retrieve(q)
                latencies.append(result.latency_ms)
            total_time = time.perf_counter() - start_all
        else:
            start_all = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(self.retriever.retrieve, q): q for q in queries}
                for f in as_completed(futures):
                    result = f.result()
                    latencies.append(result.latency_ms)
            total_time = time.perf_counter() - start_all

        latencies.sort()
        n = len(latencies)
        qps = n / total_time if total_time > 0 else 0

        return LatencyReport(
            retriever_name=self.retriever.name,
            num_queries=n,
            p50_ms=latencies[int(n * 0.50)] if n else 0,
            p95_ms=latencies[int(n * 0.95)] if n else 0,
            p99_ms=latencies[int(n * 0.99)] if n else 0,
            mean_ms=statistics.mean(latencies) if n else 0,
            throughput_qps=qps,
            latencies_ms=latencies,
        )


class ScaleProfiler:
    """Run scale tests across different document counts and QPS targets."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(
        self,
        queries: list[str],
        doc_counts: list[int] | None = None,
        concurrency_levels: list[int] | None = None,
    ) -> ScaleReport:
        concurrency_levels = concurrency_levels or [1, 4, 8]
        report = ScaleReport()

        for retriever in self.pipeline.retrievers:
            for conc in concurrency_levels:
                profiler = LatencyProfiler(retriever)
                lr = profiler.profile(queries, concurrency=conc)
                lr.retriever_name = f"{retriever.name} (conc={conc})"
                report.reports.append(lr)
                log.info(
                    f"{lr.retriever_name}: p99={lr.p99_ms:.1f}ms, QPS={lr.throughput_qps:.1f}"
                )

        return report
