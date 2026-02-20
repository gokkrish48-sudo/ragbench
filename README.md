# RAGBench: A Large-Scale Retrieval Evaluation & Optimization System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RAGBench** is a production-grade benchmarking and optimization framework for Retrieval-Augmented Generation (RAG) systems. It provides end-to-end evaluation of retrieval quality, generation faithfulness, and system-level performance across multiple retriever backends, embedding models, and LLM generators — at scale.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RAGBench                             │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│ Ingest   │ Retrieve │ Evaluate │ Optimize │   Dashboard    │
│          │          │          │          │                │
│ • PDF    │ • BM25   │ • P@K    │ • Grid   │ • Metrics      │
│ • JSON   │ • Dense  │ • R@K    │ • Bayesian│ • Comparisons │
│ • CSV    │ • Hybrid │ • nDCG   │ • Ablation│ • Export       │
│ • Web    │ • Graph  │ • MRR    │          │                │
│          │          │ • Faith. │          │                │
│          │          │ • Latency│          │                │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

## Key Features

- **Multi-Retriever Benchmarking** — BM25 (sparse), Dense (FAISS/Pinecone), Hybrid (RRF fusion), Graph (Neo4j entity-linked)
- **Comprehensive Metrics** — Precision@K, Recall@K, nDCG@K, MRR, Hit Rate, faithfulness scoring, latency profiling
- **Hyperparameter Optimization** — Grid search, Bayesian optimization (Optuna), ablation studies over chunk size, overlap, top-k, reranking
- **LLM-as-Judge Evaluation** — Faithfulness, relevance, and hallucination detection via configurable judge models
- **Scale Testing** — Benchmark from 1K to 10M+ documents with latency/throughput profiling
- **Experiment Tracking** — MLflow integration for reproducible runs with full config logging

## Quick Start

```bash
# Clone
git clone https://github.com/gokkrish48-sudo/ragbench.git
cd ragbench

# Install
pip install -e ".[dev]"

# Run a benchmark
ragbench run --config configs/default.yaml

# View results
ragbench report --run-id latest
```

## Project Structure

```
ragbench/
├── ragbench/                # Core Python package
├── dashboard/               # Interactive Visualization Dashboard (Open index.html)
│   ├── index.html           # Main Entry point
│   ├── style.css            # Premium Interstellar Styling
│   ├── app.js               # Logic & Chart.js integration
│   └── architecture.png     # System Topology Visualization
├── tests/
├── configs/
├── scripts/
├── data/sample/
├── setup.py
├── pyproject.toml
└── README.md
```

## 📊 Evaluation Dashboard

RAGBench includes a high-end **interstellar-themed dashboard** for real-time visualization of your benchmarking runs.

**To view the dashboard:**
1. Navigate to the `dashboard/` directory.
2. Open `index.html` in any modern web browser.

**Features include:**
- **System Topology:** High-fidelity 2D vector architecture diagram.
- **Latency Over Time:** Line graphs tracking p50/p95/p99 latency sweeps.
- **Faithfulness Heatmaps:** Qualitative analysis of LLM-as-a-judge scores.
- **Deployment Roadmap:** Quick-start guide for K8s and Docker orchestration.

## Usage

### 1. Define a Benchmark Config

```yaml
# configs/default.yaml
experiment:
  name: "ragbench-baseline"
  tracking_uri: "mlruns/"

ingest:
  chunk_size: 512
  chunk_overlap: 64
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

retrievers:
  - type: bm25
    top_k: 10
  - type: dense
    top_k: 10
    index: faiss_flat
  - type: hybrid
    top_k: 10
    alpha: 0.6  # dense weight in RRF

evaluate:
  metrics: [precision, recall, ndcg, mrr, hit_rate]
  k_values: [1, 3, 5, 10]
  judge_model: "claude-sonnet-4-20250514"

optimize:
  method: bayesian
  n_trials: 50
  search_space:
    chunk_size: [256, 512, 1024]
    top_k: [3, 5, 10, 20]
    alpha: [0.3, 0.5, 0.7, 0.9]
```

### 2. Run Programmatically

```python
from ragbench import RAGBenchPipeline

pipeline = RAGBenchPipeline.from_config("configs/default.yaml")

# Ingest documents
pipeline.ingest("data/documents/")

# Run evaluation across all retrievers
results = pipeline.evaluate(queries="data/sample/sample_qa.json")

# Print comparison table
results.summary()

# Optimize hyperparameters
best_config = pipeline.optimize(metric="ndcg@5", n_trials=50)
print(f"Best config: {best_config}")
```

### 3. Scale Testing

```python
from ragbench import RAGBenchPipeline, ScaleProfiler

profiler = ScaleProfiler(pipeline)
report = profiler.run(
    doc_counts=[1_000, 10_000, 100_000, 1_000_000],
    qps_targets=[10, 50, 100, 500],
    measure=["p99_latency", "throughput", "memory"]
)
report.plot()
```

## Metrics Reference

| Metric | Type | Description |
|--------|------|-------------|
| Precision@K | Retrieval | Fraction of retrieved docs that are relevant |
| Recall@K | Retrieval | Fraction of relevant docs that are retrieved |
| nDCG@K | Retrieval | Normalized discounted cumulative gain |
| MRR | Retrieval | Mean reciprocal rank of first relevant doc |
| Hit Rate@K | Retrieval | Binary: is any relevant doc in top-K? |
| Faithfulness | Generation | Does the answer stay grounded in retrieved context? |
| Relevance | Generation | Does the answer address the query? |
| Hallucination Rate | Generation | Fraction of claims not supported by context |
| p50/p95/p99 Latency | System | End-to-end response time percentiles |
| Throughput (QPS) | System | Queries processed per second |

## License

MIT
