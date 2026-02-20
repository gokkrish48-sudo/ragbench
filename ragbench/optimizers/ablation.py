"""Ablation study runner — measure impact of each component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ragbench.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AblationResult:
    parameter: str
    baseline_value: Any
    ablated_value: Any
    baseline_score: float
    ablated_score: float

    @property
    def delta(self) -> float:
        return self.ablated_score - self.baseline_score

    @property
    def relative_change(self) -> float:
        if self.baseline_score == 0:
            return 0.0
        return self.delta / abs(self.baseline_score)


class AblationRunner:
    """Run ablation studies — disable/modify one parameter at a time."""

    def __init__(
        self,
        baseline_params: dict[str, Any],
        ablation_space: dict[str, list],
        objective_fn: Callable[[dict], float],
        metric: str = "ndcg@5",
    ):
        self.baseline_params = baseline_params
        self.ablation_space = ablation_space
        self.objective_fn = objective_fn
        self.metric = metric

    def run(self) -> list[AblationResult]:
        # Baseline score
        baseline_score = self.objective_fn(self.baseline_params)
        log.info(f"Baseline {self.metric}: [bold]{baseline_score:.4f}[/]")

        results = []
        for param, values in self.ablation_space.items():
            baseline_val = self.baseline_params.get(param)
            for ablated_val in values:
                if ablated_val == baseline_val:
                    continue

                ablated_params = {**self.baseline_params, param: ablated_val}
                ablated_score = self.objective_fn(ablated_params)

                result = AblationResult(
                    parameter=param,
                    baseline_value=baseline_val,
                    ablated_value=ablated_val,
                    baseline_score=baseline_score,
                    ablated_score=ablated_score,
                )
                results.append(result)
                log.info(
                    f"  {param}: {baseline_val} → {ablated_val} | "
                    f"Δ = {result.delta:+.4f} ({result.relative_change:+.1%})"
                )

        return results

    def summary(self, results: list[AblationResult]) -> str:
        lines = [
            f"{'Parameter':<20} {'Baseline':>10} {'Ablated':>10} "
            f"{'Base Score':>12} {'New Score':>12} {'Delta':>10}"
        ]
        lines.append("-" * 80)
        for r in results:
            lines.append(
                f"{r.parameter:<20} {str(r.baseline_value):>10} {str(r.ablated_value):>10} "
                f"{r.baseline_score:>12.4f} {r.ablated_score:>12.4f} {r.delta:>+10.4f}"
            )
        return "\n".join(lines)
