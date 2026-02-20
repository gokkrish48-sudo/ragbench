"""RAGBench: Large-Scale Retrieval Evaluation & Optimization System."""

__version__ = "0.1.0"

from ragbench.pipeline import RAGBenchPipeline
from ragbench.ingest import DocumentIngestor
from ragbench.evaluators.retrieval_metrics import RetrievalEvaluator
from ragbench.evaluators.latency_profiler import ScaleProfiler

__all__ = [
    "RAGBenchPipeline",
    "DocumentIngestor",
    "RetrievalEvaluator",
    "ScaleProfiler",
]
