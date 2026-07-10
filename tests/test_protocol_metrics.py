from __future__ import annotations

import unittest

from protocol_metrics import (
    StageMetrics,
    build_protocol_metric_matrix,
    compute_binary_detection_metrics,
    compute_online_metrics,
    compute_protocol_matrix_summary,
)


class ProtocolMetricsTests(unittest.TestCase):
    def test_binary_detection_metrics_use_fake_scores(self):
        metrics = compute_binary_detection_metrics(
            binary_targets=[0, 1, 0, 1],
            binary_predictions=[0, 1, 0, 0],
            fake_scores=[0.1, 0.8, 0.4, 0.3],
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["ap"], 5 / 6)
        self.assertAlmostEqual(metrics["auc"], 0.75)

    def test_online_metrics_summarize_all_detection_metrics(self):
        stage_metrics = [
            StageMetrics(
                stage_id=0,
                internal_metrics_by_generator={
                    "G1": {"accuracy": 0.8, "f1": 0.75, "ap": 0.9, "auc": 0.85}
                },
                external_metrics_by_subset={},
                new_generators=["G1"],
            ),
            StageMetrics(
                stage_id=1,
                internal_metrics_by_generator={
                    "G1": {"accuracy": 0.7, "f1": 0.65, "ap": 0.88, "auc": 0.8},
                    "G2": {"accuracy": 0.6, "f1": 0.55, "ap": 0.7, "auc": 0.65},
                },
                external_metrics_by_subset={},
                new_generators=["G2"],
            ),
        ]

        summary = compute_online_metrics(stage_metrics)

        self.assertAlmostEqual(summary["avg_accuracy_by_stage"][1], 0.65)
        self.assertAlmostEqual(summary["avg_f1_by_stage"][1], 0.6)
        self.assertAlmostEqual(summary["avg_ap_by_stage"][1], 0.79)
        self.assertAlmostEqual(summary["avg_auc_by_stage"][1], 0.725)
        self.assertAlmostEqual(summary["accuracy_forgetting_by_stage"][1], 0.1)
        self.assertAlmostEqual(summary["f1_forgetting_by_stage"][1], 0.1)
        self.assertAlmostEqual(summary["accuracy_plasticity_by_stage"][1], 0.6)
        self.assertEqual(
            summary["forgetting_by_stage"],
            summary["accuracy_forgetting_by_stage"],
        )

    def test_protocol_matrix_supports_forward_backward_metrics(self):
        stage_metrics = [
            StageMetrics(
                stage_id=0,
                internal_metrics_by_generator={
                    "G1": {"accuracy": 0.8, "f1": 0.75, "ap": 0.9, "auc": 0.8}
                },
                external_metrics_by_subset={},
                new_generators=["G1"],
                matrix_metrics_by_generator={
                    "G1": {"accuracy": 0.8, "f1": 0.75, "ap": 0.9, "auc": 0.8},
                    "G2": {"accuracy": 0.5, "f1": 0.4, "ap": 0.55, "auc": 0.5},
                },
            ),
            StageMetrics(
                stage_id=1,
                internal_metrics_by_generator={
                    "G1": {"accuracy": 0.7, "f1": 0.65, "ap": 0.85, "auc": 0.7},
                    "G2": {"accuracy": 0.9, "f1": 0.88, "ap": 0.95, "auc": 0.9},
                },
                external_metrics_by_subset={},
                new_generators=["G2"],
                matrix_metrics_by_generator={
                    "G1": {"accuracy": 0.7, "f1": 0.65, "ap": 0.85, "auc": 0.7},
                    "G2": {"accuracy": 0.9, "f1": 0.88, "ap": 0.95, "auc": 0.9},
                },
            ),
        ]

        matrix = build_protocol_metric_matrix(stage_metrics, ["G1", "G2"])
        self.assertEqual(matrix["metrics"]["auc"], [[0.8, 0.5], [0.7, 0.9]])
        self.assertFalse(matrix["records"][1]["seen"])
        self.assertTrue(matrix["records"][3]["seen"])

        summary = compute_protocol_matrix_summary(stage_metrics, ["G1", "G2"])
        self.assertAlmostEqual(summary["final_avg_auc"], 0.8)
        self.assertAlmostEqual(summary["final_auc_forgetting"], 0.1)
        self.assertAlmostEqual(summary["final_auc_bwt"], -0.1)
        self.assertAlmostEqual(summary["mean_plasticity_auc"], 0.85)
        self.assertAlmostEqual(summary["fwt_from_base_auc"], 0.0)
        self.assertEqual(summary["num_auc_forgetting_generators"], 1)
        self.assertEqual(summary["num_auc_bwt_generators"], 1)
        self.assertEqual(summary["num_auc_fwt_generators"], 1)

    def test_blurry_exposure_uses_clean_pre_exposure_row(self):
        def scores(g0, g1, g2):
            return {
                "G0": {"accuracy": g0, "f1": g0, "ap": g0, "auc": g0},
                "G1": {"accuracy": g1, "f1": g1, "ap": g1, "auc": g1},
                "G2": {"accuracy": g2, "f1": g2, "ap": g2, "auc": g2},
            }

        matrices = [
            scores(0.8, 0.4, 0.3),
            scores(0.75, 0.7, 0.6),
            scores(0.7, 0.8, 0.9),
        ]
        stage_metrics = [
            StageMetrics(
                stage_id=stage_id,
                internal_metrics_by_generator={
                    generator_name: values
                    for generator_name, values in matrix.items()
                    if generator_name == "G0" or stage_id >= 1
                },
                external_metrics_by_subset={},
                new_generators=[f"G{stage_id}"],
                matrix_metrics_by_generator=matrix,
            )
            for stage_id, matrix in enumerate(matrices)
        ]
        exposure = {
            "G0": {"first_stage_id": 0, "last_stage_id": 0},
            "G1": {"first_stage_id": 1, "last_stage_id": 2},
            "G2": {"first_stage_id": 1, "last_stage_id": 2},
        }

        matrix = build_protocol_metric_matrix(
            stage_metrics,
            ["G0", "G1", "G2"],
            exposure,
        )
        stage_one_g2 = next(
            record
            for record in matrix["records"]
            if record["stage_id"] == 1 and record["generator_name"] == "G2"
        )
        self.assertTrue(stage_one_g2["seen"])

        online = compute_online_metrics(stage_metrics, exposure)
        self.assertAlmostEqual(online["accuracy_forgetting_by_stage"][2], 0.1)
        self.assertAlmostEqual(online["accuracy_plasticity_by_stage"][2], 0.85)

        summary = compute_protocol_matrix_summary(
            stage_metrics,
            ["G0", "G1", "G2"],
            exposure,
        )
        self.assertAlmostEqual(summary["final_auc_forgetting"], 0.1)
        self.assertAlmostEqual(summary["final_auc_bwt"], -0.1)
        self.assertAlmostEqual(summary["mean_pre_task_auc"], 0.35)
        self.assertAlmostEqual(summary["fwt_from_base_auc"], 0.0)
        self.assertEqual(summary["num_auc_forgetting_generators"], 1)
        self.assertEqual(summary["num_auc_bwt_generators"], 1)
        self.assertEqual(summary["num_auc_fwt_generators"], 2)

    def test_partial_exposure_does_not_inflate_forgetting_reference(self):
        def score(value):
            return {
                "G1": {
                    "accuracy": value,
                    "f1": value,
                    "ap": value,
                    "auc": value,
                }
            }

        stage_metrics = [
            StageMetrics(0, score(0.95), {}, ["G1"]),
            StageMetrics(1, score(0.8), {}, []),
            StageMetrics(2, score(0.7), {}, []),
        ]
        exposure = {"G1": {"first_stage_id": 0, "last_stage_id": 1}}

        summary = compute_online_metrics(stage_metrics, exposure)

        self.assertAlmostEqual(
            summary["per_generator_accuracy_forgetting_by_stage"][2]["G1"],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
