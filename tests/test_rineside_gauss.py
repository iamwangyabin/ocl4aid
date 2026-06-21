from __future__ import annotations

import unittest

import torch

from models.rineside_gauss import RineSideGauss


class RineSideGaussTests(unittest.TestCase):
    def test_extracts_quartile_block_cls_tokens_and_scores_gaussian_head(self):
        torch.manual_seed(0)
        model = RineSideGauss(
            backbone_name="vit_tiny_patch16_224",
            pretrained=False,
            num_classes=2,
            task_num=2,
            rine_gauss_min_count=2,
        )

        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])

        z = model.extract_z(images)
        self.assertEqual(model.feature_layers, [2, 5, 8, 11])
        self.assertEqual(z.shape, (4, len(model.feature_layers) * model.embed_dim))
        self.assertEqual(model.feature_dim, len(model.feature_layers) * model.embed_dim)

        model.update_statistics(0, z, labels)
        self.assertTrue(torch.all(model.counts[0] == torch.tensor([2.0, 2.0])))

        logits = model.gaussian_logits_from_z(z)
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_can_still_extract_all_block_cls_tokens(self):
        model = RineSideGauss(
            backbone_name="vit_tiny_patch16_224",
            pretrained=False,
            num_classes=2,
            task_num=2,
            rine_gauss_feature_layers="all",
        )

        images = torch.randn(2, 3, 224, 224)
        z = model.extract_z(images)

        self.assertEqual(model.feature_layers, list(range(model.depth)))
        self.assertEqual(z.shape, (2, model.depth * model.embed_dim))

    def test_backbone_is_frozen(self):
        model = RineSideGauss(
            backbone_name="vit_tiny_patch16_224",
            pretrained=False,
            num_classes=2,
            task_num=2,
        )

        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertEqual(trainable, ["ddp_anchor"])

    def test_projected_stage_head_tracks_stats_and_replay(self):
        torch.manual_seed(0)
        model = RineSideGauss(
            backbone_name="vit_tiny_patch16_224",
            pretrained=False,
            num_classes=2,
            task_num=2,
            rine_gauss_projector_dim=8,
            rine_gauss_hidden_dim=16,
            rine_gauss_min_count=2,
        )

        model.begin_stage(0)
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 0, 1])
        z = model.extract_z(images)

        logits = model.projected_logits_from_z(z)
        self.assertEqual(logits.shape, (4, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(model.active_head_ids(), [0])

        model.update_projected_statistics_from_z(0, z, labels)
        self.assertTrue(torch.all(model.proj_counts[0] == torch.tensor([2.0, 2.0])))
        replay_x, replay_y = model.sample_projected_replay(0, 3)
        self.assertEqual(replay_x.shape, (6, 8))
        self.assertEqual(replay_y.tolist(), [0, 0, 0, 1, 1, 1])

        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(any(name.startswith("projectors.0") for name in trainable))
        self.assertTrue(any(name.startswith("detectors.0") for name in trainable))
        self.assertFalse(any(name.startswith("projectors.1") for name in trainable))


if __name__ == "__main__":
    unittest.main()
