# RAGBench: Production-Grade RAG Evaluation & Optimization

[![CI](https://github.com/gokkrish48-sudo/ragbench/actions/workflows/ci.yml/badge.svg)](https://github.com/gokkrish48-sudo/ragbench/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RAGBench** is a high-fidelity evaluation framework for Retrieval-Augmented Generation systems. It is designed to model the complexities of production-scale retrieval, including hybrid fusion, score calibration, and end-to-end latency-cost optimization.

---

## 🚀 Key Technical Pillars

- **Hybrid Fusion & Calibration:** Implements Reciprocal Rank Fusion (RRF), Weighted Fusion, and Z-Score normalization across sparse (BM25) and dense (Vector) retrievers.
- **Hierarchical Evaluation:** Offline metrics (nDCG, MRR, Hit Rate) coupled with online-simulation click-models.
- **Production Profiling:** Stage-by-stage latency analysis (p50/p95/p99) and unit-cost modeling ($/query).
- **Extensible Architecture:** Registry-based system for indices, retrievers, and rerankers.

---

## 🏗 System Architecture

The project follows a modular, registry-driven design:

```text
ragbench/
├── src/ragbench/
│   ├── ingestion/    # Semantic chunking & deduplication
│   ├── indices/      # HNSW, DiskANN, and Graph-based indexing
│   ├── retrieval/    # Hybrid search, Fusion, and Calibration
│   ├── reranking/    # Cross-encoders and ColBERT late-interaction
│   ├── eval/         # Unified metric engine (Offline & Simulated Online)
│   └── cost/         # Economic modeling of AI pipelines
```

Refer to [docs/architecture.md](docs/architecture.md) for a deep dive into the topology.

---

## 📊 Feature Highlights

### 1. Advanced Fusion Strategies
We don't just concatenate results. RAGBench implements:
- **RRF (Reciprocal Rank Fusion):** Robust merging without needing score normalization.
- **Score Calibration:** Solving the "range mismatch" problem between BM25 and Cosine Similarity through isotonic regression and Platt scaling.

### 2. Failure Analysis & Error Taxonomy
Automatically clusters failures using semantic embeddings to identify systematic gaps in retrieval (e.g., "Ambiguous Queries", "Missing Key Entities").

### 3. Economic Profiling
Calculate the exact cost of your RAG pipeline.
- `Cost = (Embedding_Tokens * Price) + (Index_Lookup_Ops * Price) + (Rerank_Steps * Price)`

---

## 🛠 Quick Start

### Installation
```bash
git clone https://github.com/gokkrish48-sudo/ragbench.git
cd ragbench
pip install -e "."
```

### Run a Full Benchmark
```bash
bash scripts/run_full_benchmark.sh --config configs/default.yaml
```

### Dashboard Visualization
```bash
# Launch the Streamlit dashboard for visual analysis
python -m ragbench.reporting.dashboard
```

---

## 📈 Roadmap
- [ ] Integration with Unstructured.io for complex PDF parsing.
- [ ] Native support for ScaNN (Space Partitioning) indices.
- [ ] Real-time feedback loops via Reinforcement Learning from Human Feedback (RLHF).

---

## 📜 License
Internal FAANG Preparation Track. Distributed under the MIT License.
