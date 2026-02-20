"""Bayesian hyperparameter optimization using Optuna."""

from __future__ import annotations

from typing import Any, Callable

from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class BayesianOptimizer:
    """Optuna-based Bayesian optimization for RAG hyperparameters."""

    def __init__(
        self,
        search_space: dict[str, list],
        objective_fn: Callable[[dict], float],
        metric: str = "ndcg@5",
        n_trials: int = 50,
        direction: str = "maximize",
    ):
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.metric = metric
        self.n_trials = n_trials
        self.direction = direction

    def optimize(self) -> dict[str, Any]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            log.warning("Optuna not installed — falling back to grid search")
            return self._fallback_grid()

        def objective(trial):
            params = {}
            for name, values in self.search_space.items():
                if all(isinstance(v, int) for v in values):
                    params[name] = trial.suggest_categorical(name, values)
                elif all(isinstance(v, float) for v in values):
                    params[name] = trial.suggest_categorical(name, values)
                else:
                    params[name] = trial.suggest_categorical(name, values)

            score = self.objective_fn(params)
            return score

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        log.info(
            f"Best {self.metric}: [bold]{study.best_value:.4f}[/] "
            f"with params: {study.best_params}"
        )

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "all_trials": [
                {"params": t.params, "value": t.value}
                for t in study.trials
                if t.value is not None
            ],
        }

    def _fallback_grid(self) -> dict[str, Any]:
        """Simple grid search fallback."""
        import itertools
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        best_params = {}

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            score = self.objective_fn(params)

            if self.direction == "maximize" and score > best_score:
                best_score = score
                best_params = params
            elif self.direction == "minimize" and score < best_score:
                best_score = score
                best_params = params

        return {"best_params": best_params, "best_value": best_score}
