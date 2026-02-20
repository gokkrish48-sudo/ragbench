#!/usr/bin/env python3
"""Run RAGBench evaluation from command line."""

import argparse
import sys
from pathlib import Path

from ragbench import RAGBenchPipeline
from ragbench.utils.logging import get_logger

log = get_logger("ragbench.cli")


def main():
    parser = argparse.ArgumentParser(description="RAGBench — RAG Evaluation & Optimization")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--data", "-d", required=True, help="Path to documents directory")
    parser.add_argument("--queries", "-q", required=True, help="Path to evaluation queries JSON")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--metric", default="ndcg@5", help="Optimization target metric")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    args = parser.parse_args()

    # Load pipeline
    log.info(f"Loading config: [bold]{args.config}[/]")
    pipeline = RAGBenchPipeline.from_config(args.config)

    # Ingest
    log.info(f"Ingesting documents from: [bold]{args.data}[/]")
    pipeline.ingest(args.data)

    # Evaluate
    log.info(f"Evaluating with queries: [bold]{args.queries}[/]")
    results = pipeline.evaluate(args.queries)
    results.summary()

    # Optimize
    if args.optimize:
        log.info(f"Optimizing for [bold]{args.metric}[/]...")
        best = pipeline.optimize(
            queries=args.queries,
            metric=args.metric,
            n_trials=args.trials,
        )
        log.info(f"Best config: {best['best_params']}")
        log.info(f"Best {args.metric}: {best['best_value']:.4f}")


if __name__ == "__main__":
    main()
