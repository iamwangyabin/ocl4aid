from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from methods._trainer import _Trainer, _ensure_single_process_launch


class TrainerProtocolTests(unittest.TestCase):
    def test_distributed_launch_is_rejected(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "single-process, single-GPU"):
                _ensure_single_process_launch()

    def test_single_process_launch_is_allowed(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}, clear=True):
            _ensure_single_process_launch()

    def test_blurry_exposure_is_derived_from_actual_sampler_buckets(self):
        trainer = object.__new__(_Trainer)
        trainer.protocol_generator_order = [
            {"generator_name": "G0"},
            {"generator_name": "G1"},
            {"generator_name": "G2"},
        ]
        trainer.train_dataset = SimpleNamespace(
            online_stage_targets=[0, 0, 1, 1, 2, 2]
        )
        trainer.train_sampler = SimpleNamespace(
            indices={0: [0, 1, 2], 1: [3, 4], 2: [5]}
        )
        trainer.protocol_stage_ids = [0, 1, 2]
        trainer.base_stage_epochs = 0

        exposure = trainer._build_protocol_exposure_by_generator()
        trainer.protocol_exposure_by_generator = exposure

        self.assertEqual(
            exposure,
            {
                "G0": {"first_stage_id": 0, "last_stage_id": 0},
                "G1": {"first_stage_id": 0, "last_stage_id": 1},
                "G2": {"first_stage_id": 1, "last_stage_id": 2},
            },
        )
        self.assertEqual(trainer._seen_protocol_generators(0), ["G0", "G1"])
        self.assertEqual(
            trainer._exposed_train_indices_by_generator(0),
            {"G0": [0, 1], "G1": [2]},
        )
        self.assertEqual(
            trainer._seen_protocol_generators_at_online_sample(1),
            ["G0"],
        )
        self.assertEqual(
            trainer._seen_protocol_generators_at_online_sample(4),
            ["G0", "G1"],
        )
        self.assertEqual(
            trainer._seen_protocol_generators_at_online_sample(5),
            ["G0", "G1", "G2"],
        )


if __name__ == "__main__":
    unittest.main()
