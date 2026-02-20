"""Exhaustive grid search over RAG hyperparameters."""

from __future__ import annotations

import itertools
from typing import Any, Callable

from tqdm import tqdm
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class GridSearchOptimizer:
    """Exhaustive grid search over hyperparameter space."""

    def __init__(
        self,
        search_space: dict[str, list],
        objective_fn: Callable[[dict], float],
        metric: str = "ndcg@5",
        direction: str = "maximize",
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.metric = metric
        self.direction = direction

    def optimize(self) -> dict[str, Any]:
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        combos = list(itertools.product(*values))

        log.info(f"Grid search: [bold]{len(combos)}[/] combinations")

        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        best_params = {}
        all_results = []

        for combo in tqdm(combos, desc="Grid search"):
            params = dict(zip(keys, combo))
            score = self.objective_fn(params)
            all_results.append({"params": params, "value": score})

            if self.direction == "maximize" and score > best_score:
                best_score = score
                best_params = params
            elif self.direction == "minimize" and score < best_score:
                best_score = score
                best_params = params

        log.info(f"Best {self.metric}: [bold]{best_score:.4f}[/] → {best_params}")

        return {
            "best_params": best_params,
            "best_value": best_score,
            "n_trials": len(combos),
            "all_trials": all_results,
        }
