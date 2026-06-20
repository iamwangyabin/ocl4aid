"""Online continual learning metric helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


DETECTION_METRICS = ("accuracy", "f1", "ap", "auc")


MetricValue = float | None
MetricDict = dict[str, MetricValue]


def compute_binary_detection_metrics(
    binary_targets: list[int],
    binary_predictions: list[int],
    fake_scores: list[float],
) -> MetricDict:
    """Compute binary deepfake detection metrics.

    ``fake_scores`` must be a continuous confidence score for the fake class.
    It is used for AP and ROC-AUC; hard predictions are used for accuracy/F1.
    """
    if not binary_targets:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "ap": None,
            "auc": None,
        }

    targets = [int(item) for item in binary_targets]
    predictions = [int(item) for item in binary_predictions]
    scores = [float(item) for item in fake_scores]

    metrics: MetricDict = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "ap": None,
        "auc": None,
    }
    if any(target == 1 for target in targets):
        metrics["ap"] = float(average_precision_score(targets, scores))
    if len(set(targets)) > 1:
        metrics["auc"] = float(roc_auc_score(targets, scores))
    return metrics


def _valid_average(values: list[MetricValue]) -> MetricValue:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _zero_if_none(value: MetricValue) -> float:
    return 0.0 if value is None else float(value)


def _metric_values(
    scores_by_name: dict[str, MetricDict],
    metric_name: str,
) -> list[MetricValue]:
    return [scores.get(metric_name) for scores in scores_by_name.values()]


@dataclass(frozen=True)
class StageMetrics:
    stage_id: int
    internal_metrics_by_generator: dict[str, MetricDict]
    external_metrics_by_subset: dict[str, MetricDict]
    new_generators: list[str]
    matrix_metrics_by_generator: dict[str, MetricDict] = field(default_factory=dict)

    @property
    def internal_accuracy_by_generator(self) -> dict[str, float]:
        return {
            name: _zero_if_none(metrics.get("accuracy"))
            for name, metrics in self.internal_metrics_by_generator.items()
        }

    @property
    def external_accuracy_by_subset(self) -> dict[str, float]:
        return {
            name: _zero_if_none(metrics.get("accuracy"))
            for name, metrics in self.external_metrics_by_subset.items()
        }


def compute_online_metrics(stage_metrics: list[StageMetrics]) -> dict[str, Any]:
    avg_by_metric: dict[str, dict[int, MetricValue]] = {
        metric_name: {} for metric_name in DETECTION_METRICS
    }
    forgetting_by_metric: dict[str, dict[int, MetricValue]] = {
        metric_name: {} for metric_name in DETECTION_METRICS
    }
    plasticity_by_metric: dict[str, dict[int, MetricValue]] = {
        metric_name: {} for metric_name in DETECTION_METRICS
    }
    external_by_metric: dict[str, dict[int, MetricValue]] = {
        metric_name: {} for metric_name in DETECTION_METRICS
    }
    per_stage_forgetting_by_metric: dict[str, dict[int, dict[str, float]]] = {
        metric_name: {} for metric_name in DETECTION_METRICS
    }
    best_by_metric_generator: dict[str, dict[str, float]] = {
        metric_name: {} for metric_name in DETECTION_METRICS
    }

    for stage_metric in sorted(stage_metrics, key=lambda item: item.stage_id):
        internal = stage_metric.internal_metrics_by_generator
        external = stage_metric.external_metrics_by_subset
        stage_id = stage_metric.stage_id

        for metric_name in DETECTION_METRICS:
            avg_value = _valid_average(_metric_values(internal, metric_name))
            avg_by_metric[metric_name][stage_id] = avg_value

            stage_forgetting: dict[str, float] = {}
            best_by_generator = best_by_metric_generator[metric_name]
            for generator_name, generator_metrics in internal.items():
                value = generator_metrics.get(metric_name)
                if value is None:
                    continue
                value = float(value)
                best_so_far = best_by_generator.get(generator_name, value)
                stage_forgetting[generator_name] = max(best_so_far - value, 0.0)
                best_by_generator[generator_name] = max(best_so_far, value)
            per_stage_forgetting_by_metric[metric_name][stage_id] = stage_forgetting
            forgetting_by_metric[metric_name][stage_id] = _valid_average(
                list(stage_forgetting.values())
            )

            new_scores = [
                internal[generator_name].get(metric_name)
                for generator_name in stage_metric.new_generators
                if generator_name in internal
            ]
            plasticity_by_metric[metric_name][stage_id] = _valid_average(new_scores)
            external_by_metric[metric_name][stage_id] = _valid_average(
                _metric_values(external, metric_name)
            )

    metrics: dict[str, Any] = {}
    for metric_name in DETECTION_METRICS:
        metrics[f"avg_{metric_name}_by_stage"] = {
            stage_id: _zero_if_none(value) if metric_name in {"accuracy", "f1"} else value
            for stage_id, value in avg_by_metric[metric_name].items()
        }
        metrics[f"{metric_name}_forgetting_by_stage"] = {
            stage_id: _zero_if_none(value) if metric_name in {"accuracy", "f1"} else value
            for stage_id, value in forgetting_by_metric[metric_name].items()
        }
        metrics[f"{metric_name}_plasticity_by_stage"] = {
            stage_id: _zero_if_none(value) if metric_name in {"accuracy", "f1"} else value
            for stage_id, value in plasticity_by_metric[metric_name].items()
        }
        metrics[f"external_{metric_name}_by_stage"] = external_by_metric[metric_name]
        metrics[f"per_generator_{metric_name}_forgetting_by_stage"] = (
            per_stage_forgetting_by_metric[metric_name]
        )

    metrics.update(
        {
            "avg_accuracy_by_stage": metrics["avg_accuracy_by_stage"],
            "forgetting_by_stage": metrics["accuracy_forgetting_by_stage"],
            "plasticity_by_stage": metrics["accuracy_plasticity_by_stage"],
            "external_accuracy_by_stage": metrics["external_accuracy_by_stage"],
            "per_generator_forgetting_by_stage": metrics[
                "per_generator_accuracy_forgetting_by_stage"
            ],
        }
    )
    return metrics


def build_protocol_metric_matrix(
    stage_metrics: list[StageMetrics],
    generator_order: list[str],
) -> dict[str, Any]:
    """Build dense stage x generator matrices for paper analysis.

    ``internal_metrics_by_generator`` intentionally contains only seen
    generators so online averages remain causal. ``matrix_metrics_by_generator``
    can contain all protocol generators after a stage-end full evaluation.
    """
    ordered_stages = sorted(stage_metrics, key=lambda item: item.stage_id)
    matrices: dict[str, list[list[MetricValue]]] = {
        metric_name: [] for metric_name in DETECTION_METRICS
    }
    records: list[dict[str, Any]] = []

    for stage_metric in ordered_stages:
        source = (
            stage_metric.matrix_metrics_by_generator
            or stage_metric.internal_metrics_by_generator
        )
        stage_id = int(stage_metric.stage_id)
        for metric_name in DETECTION_METRICS:
            matrices[metric_name].append(
                [
                    source.get(generator_name, {}).get(metric_name)
                    for generator_name in generator_order
                ]
            )

        for generator_id, generator_name in enumerate(generator_order):
            generator_metrics = source.get(generator_name, {})
            records.append(
                {
                    "stage_id": stage_id,
                    "stage_name": (
                        generator_order[stage_id]
                        if 0 <= stage_id < len(generator_order)
                        else str(stage_id)
                    ),
                    "generator_id": generator_id,
                    "generator_name": generator_name,
                    "seen": generator_id <= stage_id,
                    **{
                        metric_name: generator_metrics.get(metric_name)
                        for metric_name in DETECTION_METRICS
                    },
                }
            )

    return {
        "stage_order": [
            (
                generator_order[stage_metric.stage_id]
                if 0 <= stage_metric.stage_id < len(generator_order)
                else str(stage_metric.stage_id)
            )
            for stage_metric in ordered_stages
        ],
        "stage_ids": [int(stage_metric.stage_id) for stage_metric in ordered_stages],
        "generator_order": list(generator_order),
        "metrics": matrices,
        "records": records,
    }


def compute_protocol_matrix_summary(
    stage_metrics: list[StageMetrics],
    generator_order: list[str],
) -> dict[str, Any]:
    """Compute final paper metrics from the full protocol matrix."""
    matrix = build_protocol_metric_matrix(stage_metrics, generator_order)
    stage_ids = matrix["stage_ids"]
    if not stage_ids:
        return {}

    final_row = len(stage_ids) - 1
    final_stage_id = int(stage_ids[final_row])
    stage_row_by_id = {int(stage_id): row for row, stage_id in enumerate(stage_ids)}
    summary: dict[str, Any] = {"final_stage_id": final_stage_id}

    for metric_name in DETECTION_METRICS:
        metric_matrix = matrix["metrics"][metric_name]
        final_values = metric_matrix[final_row]
        summary[f"final_avg_{metric_name}"] = _valid_average(final_values)

        forgetting_values = []
        bwt_values = []
        plasticity_values = []
        pre_task_values = []
        fwt_from_base_values = []
        for generator_id in range(len(generator_order)):
            final_value = final_values[generator_id]
            if final_value is None:
                continue
            learned_row = stage_row_by_id.get(generator_id)
            if learned_row is not None:
                learned_value = metric_matrix[learned_row][generator_id]
                if learned_value is not None:
                    learned_value = float(learned_value)
                    plasticity_values.append(learned_value)
                    bwt_values.append(float(final_value) - learned_value)

            after_learning_values = [
                metric_matrix[row][generator_id]
                for row, stage_id in enumerate(stage_ids)
                if int(stage_id) >= generator_id
                and metric_matrix[row][generator_id] is not None
            ]
            if after_learning_values:
                forgetting_values.append(
                    max(float(value) for value in after_learning_values)
                    - float(final_value)
                )

            previous_row = stage_row_by_id.get(generator_id - 1)
            base_row = stage_row_by_id.get(0)
            if generator_id > 0 and previous_row is not None:
                pre_task_value = metric_matrix[previous_row][generator_id]
                if pre_task_value is not None:
                    pre_task_value = float(pre_task_value)
                    pre_task_values.append(pre_task_value)
                    if base_row is not None:
                        base_value = metric_matrix[base_row][generator_id]
                        if base_value is not None:
                            fwt_from_base_values.append(pre_task_value - float(base_value))

        summary[f"final_{metric_name}_forgetting"] = _valid_average(forgetting_values)
        summary[f"final_{metric_name}_bwt"] = _valid_average(bwt_values)
        summary[f"mean_plasticity_{metric_name}"] = _valid_average(plasticity_values)
        summary[f"mean_pre_task_{metric_name}"] = _valid_average(pre_task_values)
        summary[f"fwt_from_base_{metric_name}"] = _valid_average(fwt_from_base_values)

    return summary
