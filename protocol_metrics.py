"""Online continual learning metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageMetrics:
    stage_id: int
    internal_accuracy_by_generator: dict[str, float]
    external_accuracy_by_subset: dict[str, float]
    new_generators: list[str]


def compute_online_metrics(stage_metrics: list[StageMetrics]) -> dict[str, Any]:
    best_by_generator: dict[str, float] = {}
    avg_accuracy_by_stage: dict[int, float] = {}
    forgetting_by_stage: dict[int, float] = {}
    plasticity_by_stage: dict[int, float] = {}
    external_accuracy_by_stage: dict[int, float | None] = {}
    per_stage_forgetting: dict[int, dict[str, float]] = {}

    for stage_metric in sorted(stage_metrics, key=lambda item: item.stage_id):
        internal = stage_metric.internal_accuracy_by_generator
        if not internal:
            avg_accuracy_by_stage[stage_metric.stage_id] = 0.0
            forgetting_by_stage[stage_metric.stage_id] = 0.0
            plasticity_by_stage[stage_metric.stage_id] = 0.0
            external_accuracy_by_stage[stage_metric.stage_id] = None
            per_stage_forgetting[stage_metric.stage_id] = {}
            continue

        avg_accuracy_by_stage[stage_metric.stage_id] = sum(internal.values()) / len(internal)

        stage_forgetting: dict[str, float] = {}
        for generator_name, accuracy in internal.items():
            best_so_far = best_by_generator.get(generator_name, accuracy)
            stage_forgetting[generator_name] = max(best_so_far - accuracy, 0.0)
            best_by_generator[generator_name] = max(best_so_far, accuracy)
        per_stage_forgetting[stage_metric.stage_id] = stage_forgetting
        forgetting_by_stage[stage_metric.stage_id] = sum(stage_forgetting.values()) / len(stage_forgetting)

        new_scores = [
            internal[generator_name]
            for generator_name in stage_metric.new_generators
            if generator_name in internal
        ]
        plasticity_by_stage[stage_metric.stage_id] = (
            sum(new_scores) / len(new_scores) if new_scores else 0.0
        )

        external = stage_metric.external_accuracy_by_subset
        external_accuracy_by_stage[stage_metric.stage_id] = (
            sum(external.values()) / len(external) if external else None
        )

    return {
        "avg_accuracy_by_stage": avg_accuracy_by_stage,
        "forgetting_by_stage": forgetting_by_stage,
        "plasticity_by_stage": plasticity_by_stage,
        "external_accuracy_by_stage": external_accuracy_by_stage,
        "per_generator_forgetting_by_stage": per_stage_forgetting,
    }
