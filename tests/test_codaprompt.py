from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.codaprompt import CodaPrompt, ortho_penalty


class _TinyBackbone(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 16):
        super().__init__()
        self.num_features = feature_dim
        self.fc = nn.Linear(feature_dim, num_classes)


class CodaPromptTests(unittest.TestCase):
    def _model(self, *, e_pool: int, task_num: int) -> CodaPrompt:
        backbone = _TinyBackbone(num_classes=2)
        with patch("models.codaprompt.timm.create_model", return_value=backbone):
            return CodaPrompt(
                pos_e_prompt=[0],
                len_e_prompt=2,
                e_pool=e_pool,
                task_num=task_num,
                num_classes=2,
                backbone_name="_codaprompt_test_backbone",
                key_dim=16,
                pretrained=False,
            )

    def test_pool_expands_when_task_count_exceeds_requested_pool(self):
        model = self._model(e_pool=2, task_num=5)

        self.assertEqual(model.num_pt_per_task, 1)
        self.assertEqual(model.e_pool, 5)
        self.assertEqual(model.e_pool % model.task_num, 0)
        self.assertEqual(model.e_k_0.shape[0], 5)
        self.assertEqual(model.e_a_0.shape[0], 5)
        self.assertEqual(model.e_p_0.shape[0], 5)
        for task_id in range(model.task_num):
            start = task_id * model.num_pt_per_task
            stop = start + model.num_pt_per_task
            self.assertGreater(stop - start, 0)
            self.assertLessEqual(stop, model.e_pool)
            if task_id + 1 < model.task_num:
                model.process_task_count()

    def test_pool_rounds_up_without_dropping_remainder(self):
        model = self._model(e_pool=7, task_num=3)

        self.assertEqual(model.num_pt_per_task, 3)
        self.assertEqual(model.e_pool, 9)
        self.assertEqual(model.e_pool % model.task_num, 0)

    def test_impossible_orthogonal_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds feature capacity"):
            self._model(e_pool=17, task_num=1)

    def test_legacy_non_divisible_checkpoint_requires_regeneration(self):
        model = self._model(e_pool=7, task_num=3)
        legacy_state = model.state_dict()
        legacy_state["e_p_0"] = torch.zeros(7, 2, 16)
        legacy_state["e_k_0"] = torch.zeros(7, 16)
        legacy_state["e_a_0"] = torch.zeros(7, 16)

        with self.assertRaisesRegex(RuntimeError, "Regenerate the base checkpoint"):
            model.load_state_dict(legacy_state)

    def test_task_transition_preserves_parameter_and_optimizer_identity(self):
        model = self._model(e_pool=5, task_num=3)
        prompt_parameter = model.e_p_0
        parameter = model.e_k_0
        attention_parameter = model.e_a_0
        optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)

        optimizer.zero_grad()
        parameter[: model.num_pt_per_task].sum().backward()
        optimizer.step()
        self.assertIn(parameter, optimizer.state)

        model.process_task_count()

        self.assertIs(model.e_k_0, parameter)
        self.assertIs(model.e_p_0, prompt_parameter)
        self.assertIs(model.e_a_0, attention_parameter)
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
        self.assertIn(parameter, optimizer.state)

        start = model.task_count * model.num_pt_per_task
        stop = start + model.num_pt_per_task
        before = parameter[start:stop].detach().clone()
        optimizer.zero_grad()
        parameter[start:stop].sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(parameter[start:stop].detach(), before))

    def test_ortho_penalty_uses_input_device(self):
        value = torch.eye(3)
        penalty = ortho_penalty(value)

        self.assertEqual(penalty.device, value.device)
        self.assertEqual(penalty.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
