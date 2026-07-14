from __future__ import annotations

import unittest

import torch

from methods.race_components import (
    AllocationController,
    ClassConditionalChangeDetector,
    DiagonalMoments,
    PersistentFeatureReplay,
    symmetric_diagonal_gaussian_kl,
)


class DiagonalMomentsTests(unittest.TestCase):
    def test_streaming_merge_matches_direct_statistics(self):
        values = torch.tensor(
            [
                [1.0, 2.0, 4.0],
                [2.0, 4.0, 8.0],
                [3.0, 6.0, 12.0],
                [5.0, 10.0, 20.0],
            ]
        )
        moments = DiagonalMoments.from_features(values[:2])
        moments.merge(DiagonalMoments.from_features(values[2:]))

        self.assertEqual(moments.count, 4)
        self.assertTrue(torch.allclose(moments.mean, values.mean(dim=0)))
        self.assertTrue(
            torch.allclose(moments.variance(), values.var(dim=0, unbiased=True))
        )

    def test_symmetric_kl_is_zero_for_identical_and_symmetric_for_shifted(self):
        left = DiagonalMoments.from_features(
            torch.tensor([[0.0, 1.0], [1.0, 2.0], [-1.0, 0.0]])
        )
        same = left.clone()
        right = DiagonalMoments.from_features(
            torch.tensor([[2.0, -1.0], [3.0, 0.0], [1.0, -2.0]])
        )

        self.assertAlmostEqual(symmetric_diagonal_gaussian_kl(left, same), 0.0, places=7)
        forward = symmetric_diagonal_gaussian_kl(left, right)
        reverse = symmetric_diagonal_gaussian_kl(right, left)
        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(forward, reverse, places=7)


class ChangeDetectorTests(unittest.TestCase):
    @staticmethod
    def _balanced_stationary_features(blocks=8, block_size=20):
        generator = torch.Generator().manual_seed(7)
        labels = torch.tensor([0, 1] * (blocks * block_size // 2))
        means = torch.where(labels[:, None] == 0, 1.0, -1.0)
        features = torch.cat(
            [
                means + 0.08 * torch.randn(labels.numel(), 1, generator=generator),
                0.08 * torch.randn(labels.numel(), 1, generator=generator),
            ],
            dim=1,
        )
        return features, labels

    def test_base_calibration_is_deterministic_and_serializable(self):
        features, labels = self._balanced_stationary_features()
        kwargs = dict(
            feature_dim=2,
            window_size=20,
            min_class_count=4,
            min_calibration_scores=2,
            warmup_samples=20,
        )
        first = ClassConditionalChangeDetector(**kwargs)
        second = ClassConditionalChangeDetector(**kwargs)

        threshold = first.calibrate(features, labels)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertEqual(threshold, second.calibrate(features, labels))

        restored = ClassConditionalChangeDetector(**kwargs)
        restored.load_state_dict(first.state_dict())
        self.assertEqual(restored.diagnostics(), first.diagnostics())

    def test_calibration_fails_when_base_has_too_few_valid_windows(self):
        detector = ClassConditionalChangeDetector(
            2,
            window_size=20,
            min_class_count=4,
            min_calibration_scores=4,
        )
        features, labels = self._balanced_stationary_features(blocks=3)
        with self.assertRaisesRegex(ValueError, "insufficient valid"):
            detector.calibrate(features, labels)

    def test_class_conditional_score_ignores_binary_prior_change(self):
        class_zero = torch.tensor([[1.0, 0.1], [1.0, -0.1]])
        class_one = torch.tensor([[-1.0, 0.1], [-1.0, -0.1]])
        reference = torch.cat([class_zero.repeat(10, 1), class_one.repeat(10, 1)])
        reference_labels = torch.tensor([0] * 20 + [1] * 20)
        recent = torch.cat([class_zero.repeat(15, 1), class_one.repeat(5, 1)])
        recent_labels = torch.tensor([0] * 30 + [1] * 10)
        detector = ClassConditionalChangeDetector(
            2,
            window_size=40,
            ewma_beta=0.0,
            persistence=1,
            warmup_samples=40,
            cooldown_samples=0,
            min_class_count=4,
            threshold=0.1,
        )
        detector.reset_reference(reference, reference_labels, start_cooldown=False)

        event = detector.observe(recent, recent_labels, sample_offset=0)

        self.assertIsNone(event)
        # Repeating the same support with different counts changes the unbiased
        # finite-sample variance by a tiny n/(n-1) factor, but not materially.
        self.assertLess(detector.diagnostics()["last_raw_score"], 1e-3)

    def test_shift_requires_persistence_and_cooldown_blocks_retrigger(self):
        reference = torch.tensor(
            [[1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1]]
        ).repeat(4, 1)
        reference_labels = torch.tensor([0, 0, 1, 1] * 4)
        shifted = torch.tensor(
            [[0.1, 1.0], [-0.1, 1.0], [0.1, -1.0], [-0.1, -1.0]]
        )
        shifted_labels = torch.tensor([0, 0, 1, 1])
        detector = ClassConditionalChangeDetector(
            2,
            window_size=8,
            ewma_beta=0.0,
            persistence=2,
            warmup_samples=8,
            cooldown_samples=8,
            min_class_count=2,
            threshold=0.1,
        )
        detector.reset_reference(reference, reference_labels, start_cooldown=False)

        self.assertIsNone(detector.observe(shifted, shifted_labels, 0))
        self.assertIsNone(detector.observe(shifted, shifted_labels, 4))
        event = detector.observe(shifted, shifted_labels, 8)
        self.assertIsNotNone(event)
        self.assertEqual(event.persistence, 2)
        self.assertEqual(event.sample_offset, 8)
        self.assertEqual(event.candidate_features.size(0), 8)

        detector.reset_reference(
            event.candidate_features,
            event.candidate_labels,
            start_cooldown=True,
        )
        self.assertIsNone(detector.observe(shifted, shifted_labels, 12))
        self.assertGreater(detector.diagnostics()["cooldown_remaining"], 0)


class PersistentReplayTests(unittest.TestCase):
    @staticmethod
    def _batch(start, count):
        ids = torch.arange(start, start + count, dtype=torch.float32)
        features = torch.stack([ids, ids + 0.25], dim=1)
        logits = torch.stack([-ids, ids], dim=1)
        labels = torch.arange(start, start + count) % 2
        return features, logits, labels

    def test_fifo_is_bounded_recent_is_chronological_and_clear_is_explicit(self):
        replay = PersistentFeatureReplay(6, 2, seed=3)
        replay.add(*self._batch(0, 4), regime_id=0)
        replay.add(*self._batch(4, 4), regime_id=1)

        self.assertEqual(len(replay), 6)
        recent = replay.recent(3)
        self.assertTrue(torch.equal(recent.features[:, 0], torch.tensor([5.0, 6.0, 7.0])))
        self.assertEqual(replay.diagnostics()["seen"], 8)

        replay.clear()
        self.assertEqual(len(replay), 0)
        self.assertEqual(replay.diagnostics()["seen"], 8)

    def test_balanced_sampling_uses_equal_quotas_with_replacement(self):
        replay = PersistentFeatureReplay(8, 2, sampling="class_balanced", seed=11)
        features, logits, _labels = self._batch(0, 5)
        labels = torch.tensor([0, 0, 0, 0, 1])
        replay.add(features, logits, labels, regime_id=0)

        sampled = replay.sample(8, "cpu")

        self.assertEqual(len(sampled), 8)
        self.assertEqual(int((sampled.labels == 0).sum()), 4)
        self.assertEqual(int((sampled.labels == 1).sum()), 4)

    def test_regime_filter_requires_opt_in_and_state_restores_rng(self):
        replay = PersistentFeatureReplay(
            10,
            2,
            seed=5,
            store_regime_ids=True,
        )
        replay.add(*self._batch(0, 4), regime_id=0)
        replay.add(*self._batch(4, 4), regime_id=1)
        state = replay.state_dict()
        restored = PersistentFeatureReplay(
            10,
            2,
            seed=999,
            store_regime_ids=True,
        )
        restored.load_state_dict(state)

        original_sample = replay.sample(6, regime_id=1)
        restored_sample = restored.sample(6, regime_id=1)
        self.assertTrue(torch.equal(original_sample.features, restored_sample.features))
        self.assertTrue(torch.all(restored_sample.regime_ids == 1))

    def test_reservoir_replacement_is_seed_reproducible(self):
        first = PersistentFeatureReplay(5, 2, replacement="reservoir", seed=17)
        second = PersistentFeatureReplay(5, 2, replacement="reservoir", seed=17)
        batch = self._batch(0, 30)
        first.add(*batch, regime_id=0)
        second.add(*batch, regime_id=0)

        self.assertTrue(
            torch.equal(
                first.state_dict()["features"],
                second.state_dict()["features"],
            )
        )
        with self.assertRaisesRegex(ValueError, "FIFO"):
            first.recent(2)

    def test_zero_capacity_records_no_samples_and_returns_no_replay(self):
        replay = PersistentFeatureReplay(0, 2)
        replay.add(*self._batch(0, 4), regime_id=0)
        self.assertEqual(len(replay), 0)
        self.assertIsNone(replay.sample(4))
        self.assertEqual(replay.diagnostics()["seen"], 4)


class AllocationControllerTests(unittest.TestCase):
    def test_detected_single_none_and_oracle_modes_are_explicit(self):
        detected = AllocationController("detected")
        self.assertIsNone(detected.should_allocate(0))
        event = detected.should_allocate(64, detected=True)
        self.assertEqual(event.residual_expert_id, 1)
        self.assertFalse(detected.diagnostics()["uses_boundary_signal"])

        single = AllocationController("single")
        self.assertIsNotNone(single.should_allocate(0))
        self.assertIsNone(single.should_allocate(16))

        none = AllocationController("none")
        self.assertIsNone(none.should_allocate(0, detected=True, oracle=True))

        oracle = AllocationController("oracle")
        self.assertIsNone(oracle.should_allocate(0))
        self.assertIsNotNone(oracle.should_allocate(0, oracle=True))
        self.assertTrue(oracle.diagnostics()["uses_boundary_signal"])

    def test_fixed_schedule_is_even_and_includes_first_batch(self):
        controller = AllocationController(
            "fixed",
            num_regimes=5,
            total_online_samples=100,
        )
        self.assertEqual(controller.positions, (0, 25, 50, 75))
        observed = []
        for offset in range(0, 100, 5):
            event = controller.should_allocate(offset)
            if event is not None:
                observed.append(event.sample_offset)
        self.assertEqual(observed, [0, 25, 50, 75])

    def test_random_schedule_is_seeded_capacity_matched_and_includes_zero(self):
        first = AllocationController(
            "random",
            num_regimes=6,
            total_online_samples=100,
            seed=23,
        )
        second = AllocationController(
            "random",
            num_regimes=6,
            total_online_samples=100,
            seed=23,
        )
        other = AllocationController(
            "random",
            num_regimes=6,
            total_online_samples=100,
            seed=24,
        )

        self.assertEqual(first.positions, second.positions)
        self.assertEqual(first.positions[0], 0)
        self.assertEqual(len(first.positions), 5)
        self.assertEqual(len(set(first.positions)), 5)
        self.assertNotEqual(first.positions, other.positions)

    def test_controller_state_roundtrip_preserves_next_schedule_position(self):
        original = AllocationController(
            "fixed",
            num_regimes=4,
            total_online_samples=90,
        )
        self.assertIsNotNone(original.should_allocate(0))
        restored = AllocationController(
            "fixed",
            num_regimes=4,
            total_online_samples=90,
        )
        restored.load_state_dict(original.state_dict())

        self.assertIsNone(restored.should_allocate(20))
        event = restored.should_allocate(30)
        self.assertEqual(event.residual_expert_id, 2)
        self.assertEqual(event.scheduled_offset, 30)


if __name__ == "__main__":
    unittest.main()
