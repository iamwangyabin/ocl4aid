from __future__ import annotations

import unittest

import torch

from methods.rigev2 import RIGEv2
from models.rigev1 import LowRankResidualHead
from models.rigev2 import RIGEv2 as RIGEv2Model


class _FakeRIGEv2Model:
    def __init__(self, indices=None):
        self.online_feature_dim = 2
        self.raw_online_feature_dim = 4
        self.online_feature_indices = (
            torch.empty(0, dtype=torch.long)
            if indices is None
            else torch.as_tensor(indices, dtype=torch.long)
        )

    def set_online_feature_indices(self, indices):
        self.online_feature_indices = indices.detach().cpu().long().clone()


class _RIGEv2Harness(RIGEv2):
    def __init__(self, indices=None):
        self.model_without_ddp = _FakeRIGEv2Model(indices)
        self.task_id = 0
        self._route_stats_by_expert = {}
        self._threshold_scores = []
        self._threshold_labels = []
        self._threshold_scores_by_expert = {}
        self._threshold_labels_by_expert = {}
        self._decision_thresholds = {}
        self._decision_threshold = 0.0
        self._calibration_score_drift_ema_by_expert = {}
        self._rigev2_base_stats_selector = None
        self.ensure_calls = 0
        self.rebuild_calls = 0

    def _base_stage_id(self):
        return 0

    def _ensure_online_feature_indices(self):
        self.ensure_calls += 1
        if self.model_without_ddp.online_feature_indices.numel() == 0:
            self.model_without_ddp.set_online_feature_indices(torch.tensor([1, 3]))

    def _rebuild_stage_statistics_from_train_data(self, stage_id):
        stage_id = int(stage_id)
        self.rebuild_calls += 1
        if stage_id in self._route_stats_by_expert:
            raise AssertionError("route stats must be cleared before rebuilding")
        if stage_id in self._decision_thresholds:
            raise AssertionError("decision threshold must be cleared before rebuilding")
        if stage_id in self._threshold_scores_by_expert:
            raise AssertionError("threshold scores must be cleared before rebuilding")

        selected = self.model_without_ddp.online_feature_indices.float()
        self._route_stats_by_expert[stage_id] = {
            "count": 2,
            "mean": selected.clone(),
            "m2": torch.zeros_like(selected),
        }
        self._threshold_scores_by_expert[stage_id] = [-1.0, 1.0]
        self._threshold_labels_by_expert[stage_id] = [0, 1]
        self._threshold_scores = [-1.0, 1.0]
        self._threshold_labels = [0, 1]
        self._decision_thresholds[stage_id] = 0.5
        self._decision_threshold = 0.5

    def seed_stale_base_statistics(self):
        self._route_stats_by_expert[0] = {
            "count": 99,
            "mean": torch.tensor([100.0, 200.0]),
            "m2": torch.tensor([3.0, 4.0]),
        }
        self._threshold_scores_by_expert[0] = [99.0]
        self._threshold_labels_by_expert[0] = [1]
        self._threshold_scores = [99.0]
        self._threshold_labels = [1]
        self._decision_thresholds[0] = 99.0
        self._decision_threshold = 99.0
        self._calibration_score_drift_ema_by_expert[0] = 9.0


def _checkpoint_state(*, selector_marker=None):
    state = {
        "decision_thresholds": {0: 99.0},
        "route_stats_by_expert": {
            0: {
                "count": 99,
                "mean": torch.tensor([100.0, 200.0]),
                "m2": torch.tensor([3.0, 4.0]),
            }
        },
        "rigev2_online_feature_indices": torch.tensor([1, 3]),
    }
    if selector_marker is not None:
        state[RIGEv2._BASE_STATS_SELECTOR_STATE_KEY] = {
            "version": RIGEv2._BASE_STATS_SELECTOR_VERSION,
            "base_stage_id": 0,
            "online_feature_indices": torch.as_tensor(selector_marker, dtype=torch.long),
        }
    return state


class RIGEv2StatisticsTests(unittest.TestCase):
    def test_race_uses_continuous_stream_by_default(self):
        trainer = _RIGEv2Harness(indices=[1, 3])

        self.assertTrue(trainer._uses_continuous_online_stream())

    def test_random_feature_selector_is_seeded_and_budget_matched(self):
        trainer = _RIGEv2Harness(indices=[1, 3])
        trainer.rigev2_feature_selector = "random"
        trainer.rigev2_feature_selector_seed = 7
        scores = torch.arange(20, dtype=torch.float32)

        first = trainer._select_headweight_indices(scores, 6)
        second = trainer._select_headweight_indices(scores, 6)

        self.assertEqual(first.numel(), 6)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.unique().numel(), 6)

    def test_identity_selector_requires_full_dimension(self):
        trainer = _RIGEv2Harness(indices=[1, 3])
        trainer.rigev2_feature_selector = "identity"
        scores = torch.ones(4)

        self.assertTrue(
            torch.equal(
                trainer._select_headweight_indices(scores, 4),
                torch.arange(4),
            )
        )
        with self.assertRaisesRegex(ValueError, "requires replay_dim"):
            trainer._select_headweight_indices(scores, 2)

    def test_residual_output_layer_is_zero_initialized(self):
        head = LowRankResidualHead(feature_dim=8, rank=3, num_classes=2)
        RIGEv2Model._zero_output_layer(head)

        output = head(torch.randn(5, 8))

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_fixed_and_random_allocations_are_batch_aligned(self):
        trainer = _RIGEv2Harness(indices=[1, 3])
        trainer.batchsize = 16

        fixed = trainer._batch_aligned_allocation_positions("fixed", 5, 1000, 7)
        random = trainer._batch_aligned_allocation_positions("random", 5, 1000, 7)

        self.assertEqual(len(fixed), 4)
        self.assertEqual(len(random), 4)
        self.assertEqual(fixed[0], 0)
        self.assertEqual(random[0], 0)
        self.assertTrue(all(offset % 16 == 0 for offset in fixed + random))

    def test_base_calibration_interleaves_grouped_binary_labels(self):
        trainer = _RIGEv2Harness(indices=[1, 3])
        features = torch.arange(24, dtype=torch.float32).view(12, 2)
        labels = torch.tensor([0] * 6 + [1] * 6)

        mixed_features, mixed_labels = trainer._interleave_base_calibration(
            features,
            labels,
        )

        self.assertEqual(mixed_features.shape, features.shape)
        self.assertEqual(mixed_labels.tolist()[::2], [0] * 6)
        self.assertEqual(mixed_labels.tolist()[1::2], [1] * 6)
        again, again_labels = trainer._interleave_base_calibration(features, labels)
        self.assertTrue(torch.equal(mixed_features, again))
        self.assertTrue(torch.equal(mixed_labels, again_labels))

    def test_rejects_raw_route_space_instead_of_only_changing_checkpoint_validation(self):
        trainer = _RIGEv2Harness(indices=[1, 3])
        trainer.rigev2_route_space = "raw"

        with self.assertRaisesRegex(ValueError, "must be 'online'"):
            trainer.online_before_task(0)

    def test_fresh_base_rebuilds_stats_after_finalizing_selector(self):
        trainer = _RIGEv2Harness()
        trainer.seed_stale_base_statistics()
        trainer._route_stats_by_expert[1] = {
            "count": 3,
            "mean": torch.tensor([7.0, 8.0]),
            "m2": torch.tensor([1.0, 1.0]),
        }

        trainer.after_base_stage_train(0)

        self.assertEqual(trainer.ensure_calls, 1)
        self.assertEqual(trainer.rebuild_calls, 1)
        self.assertEqual(trainer._route_stats_by_expert[0]["count"], 2)
        self.assertTrue(
            torch.equal(
                trainer._route_stats_by_expert[0]["mean"],
                torch.tensor([1.0, 3.0]),
            )
        )
        self.assertEqual(trainer._route_stats_by_expert[1]["count"], 3)
        self.assertEqual(trainer._threshold_scores_by_expert[0], [-1.0, 1.0])
        self.assertNotIn(0, trainer._calibration_score_drift_ema_by_expert)

        method_state = trainer._checkpoint_method_state()
        marker = method_state[RIGEv2._BASE_STATS_SELECTOR_STATE_KEY]
        self.assertEqual(marker["base_stage_id"], 0)
        self.assertTrue(
            torch.equal(
                marker["online_feature_indices"],
                method_state["rigev2_online_feature_indices"],
            )
        )

    def test_legacy_checkpoint_without_selector_marker_is_rebuilt(self):
        trainer = _RIGEv2Harness()
        state = _checkpoint_state()

        trainer._load_checkpoint_method_state(state)
        trainer._after_base_checkpoint_loaded({"method_state": state})

        self.assertEqual(trainer.rebuild_calls, 1)
        self.assertEqual(trainer._route_stats_by_expert[0]["count"], 2)
        self.assertTrue(trainer._base_stats_selector_matches_current(0))

    def test_same_dim_stats_with_mismatched_selector_are_rebuilt(self):
        trainer = _RIGEv2Harness()
        state = _checkpoint_state(selector_marker=[0, 2])

        trainer._load_checkpoint_method_state(state)
        trainer._after_base_checkpoint_loaded({"method_state": state})

        self.assertEqual(trainer.rebuild_calls, 1)
        self.assertTrue(
            torch.equal(
                trainer._route_stats_by_expert[0]["mean"],
                torch.tensor([1.0, 3.0]),
            )
        )
        self.assertTrue(trainer._base_stats_selector_matches_current(0))

    def test_matching_selector_marker_keeps_checkpoint_stats(self):
        trainer = _RIGEv2Harness()
        state = _checkpoint_state(selector_marker=[1, 3])

        trainer._load_checkpoint_method_state(state)
        trainer._after_base_checkpoint_loaded({"method_state": state})

        self.assertEqual(trainer.rebuild_calls, 0)
        self.assertEqual(trainer._route_stats_by_expert[0]["count"], 99)
        self.assertEqual(trainer._decision_thresholds[0], 99.0)


if __name__ == "__main__":
    unittest.main()
