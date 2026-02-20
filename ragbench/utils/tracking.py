"""MLflow experiment tracking wrapper."""

from __future__ import annotations

from typing import Any
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """Thin wrapper around MLflow for experiment tracking."""

    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns/"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run = None

        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            log.info(f"MLflow tracking: [bold]{tracking_uri}[/]")
        except ImportError:
            self._mlflow = None
            log.warning("MLflow not installed — tracking disabled")

    def start_run(self, run_name: str | None = None) -> None:
        if self._mlflow:
            self._run = self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        if self._mlflow and self._run:
            self._mlflow.end_run()
            self._run = None

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow:
            flat = self._flatten(params)
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            for k, v in metrics.items():
                self._mlflow.log_metric(k, v, step=step)

    def log_artifact(self, path: str) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(path)

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(ExperimentTracker._flatten(v, key))
            else:
                items[key] = str(v)
        return items
