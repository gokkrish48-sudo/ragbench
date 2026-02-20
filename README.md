# RAGBench: Large-Scale Retrieval Evaluation & Optimization

![RAG Architecture](architecture.png)

RAGBench is a high-performance evaluation framework designed to benchmark and optimize Retrieval-Augmented Generation (RAG) pipelines at scale. It focuses on the critical trade-offs between retrieval latency, chunking strategies, and generational faithfulness.

## 🚀 Key Features

- **Large-Scale Metrics:** Evaluates millions of queries using metrics like Faithfulness, Relevancy, Hit Rate, and Recall@K.
- **Optimization Engine:** Automated selection of chunking strategies (Semantic vs. Recursive) based on document topology.
- **Multi-Stage Retrieval:** Benchmarks Bi-encoders (HNSW) and Cross-encoder reranking layers.
- **Interactive Dashboard:** Premium interstellar UI for real-time monitoring of evaluation sweeps.
- **LLM-as-a-Judge:** Integration for automated qualitative scoring using SOTA models (GPT-4o, Claude 3.5).

## 🛠 Tech Stack

- **Core Logic:** Python, NumPy, PyTorch (stubs).
- **Visualization:** Vanilla JS, CSS3 (Glassmorphism), Chart.js.
- **Data Layers:** Vector DB integration (stubs for Milvus/Pinecone).

## 🏗 System Architecture

The system is built on a modular topology:
1. **Retrieval Tier:** Handles high-concurrency vector searches.
2. **Reranking Layer:** Refines top-K results using compute-intensive cross-encoders.
3. **Evaluation Engine:** Asynchronously grades responses for factual accuracy and grounding.

## 📦 Deployment

```bash
# Start the full stack (VectorDB + Evaluator + UI)
docker-compose up --build -d
```

## 📈 Performance Targets

- **Retrieval Latency:** < 50ms (p99) for 1M+ vectors.
- **Eval Throughput:** 1.2k queries/second.
- **Avg Faithfulness:** > 0.94 on standardized benchmarks.

---
Created for the **Senior GenAI Engineer** interview track.
