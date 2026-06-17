from __future__ import annotations

import unittest

from protocol_metrics import (
    StageMetrics,
    compute_binary_detection_metrics,
    compute_online_metrics,
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
        self.assertAlmostEqual(summary["accuracy_forgetting_by_stage"][1], 0.05)
        self.assertAlmostEqual(summary["f1_forgetting_by_stage"][1], 0.05)
        self.assertAlmostEqual(summary["accuracy_plasticity_by_stage"][1], 0.6)
        self.assertEqual(
            summary["forgetting_by_stage"],
            summary["accuracy_forgetting_by_stage"],
        )


if __name__ == "__main__":
    unittest.main()
