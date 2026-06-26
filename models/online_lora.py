import logging
import math
from typing import Dict, Iterable, Tuple

import timm
import torch
import torch.nn as nn

import models.vit as vit


logger = logging.getLogger()


class OnlineLoRAQKVAdapter(nn.Module):
    """Official-style Online-LoRA qkv adapter.

    The frozen old branch stores consolidated LoRA weights. The trainable new
    branch learns the current online residual and is periodically merged into
    the old branch, then reset, following the official Online-LoRA code.
    """

    def __init__(self, qkv: nn.Linear, dim: int, rank: int):
        super().__init__()
        self.base_qkv = qkv
        self.dim = int(dim)
        self.rank = int(rank)

        self.old_A_q = nn.Linear(self.dim, self.rank, bias=False)
        self.old_B_q = nn.Linear(self.rank, self.dim, bias=False)
        self.old_A_v = nn.Linear(self.dim, self.rank, bias=False)
        self.old_B_v = nn.Linear(self.rank, self.dim, bias=False)
        self.new_A_q = nn.Linear(self.dim, self.rank, bias=False)
        self.new_B_q = nn.Linear(self.rank, self.dim, bias=False)
        self.new_A_v = nn.Linear(self.dim, self.rank, bias=False)
        self.new_B_v = nn.Linear(self.rank, self.dim, bias=False)

        for param in self.base_qkv.parameters():
            param.requires_grad = False
        for module in (self.old_A_q, self.old_B_q, self.old_A_v, self.old_B_v):
            for param in module.parameters():
                param.requires_grad = False
        for module in (self.new_A_q, self.new_B_q, self.new_A_v, self.new_B_v):
            for param in module.parameters():
                param.requires_grad = True

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.old_A_q, self.old_B_q, self.old_A_v, self.old_B_v):
            nn.init.zeros_(module.weight)
        nn.init.kaiming_uniform_(self.new_A_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.new_B_q.weight)
        nn.init.kaiming_uniform_(self.new_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.new_B_v.weight)

    def reset_new_parameters(self) -> None:
        nn.init.zeros_(self.new_A_q.weight)
        nn.init.zeros_(self.new_B_q.weight)
        nn.init.zeros_(self.new_A_v.weight)
        nn.init.zeros_(self.new_B_v.weight)

    def merge_and_reset_new(self) -> None:
        with torch.no_grad():
            self.old_A_q.weight.add_(self.new_A_q.weight)
            self.old_B_q.weight.add_(self.new_B_q.weight)
            self.old_A_v.weight.add_(self.new_A_v.weight)
            self.old_B_v.weight.add_(self.new_B_v.weight)
        self.reset_new_parameters()

    def forward(self, x: torch.Tensor, use_new: bool = True) -> torch.Tensor:
        base_out = self.base_qkv(x)
        bsz, tokens, _ = base_out.shape
        base_q, base_k, base_v = base_out.chunk(3, dim=-1)

        x_flat = x.reshape(-1, self.dim)
        q_delta = self.old_B_q(self.old_A_q(x_flat))
        v_delta = self.old_B_v(self.old_A_v(x_flat))
        if use_new:
            q_delta = q_delta + self.new_B_q(self.new_A_q(x_flat))
            v_delta = v_delta + self.new_B_v(self.new_A_v(x_flat))

        q = base_q + q_delta.view(bsz, tokens, self.dim)
        v = base_v + v_delta.view(bsz, tokens, self.dim)
        return torch.cat([q, base_k, v], dim=-1)


class OnlineLoRAModel(nn.Module):
    """Online-LoRA model wrapper for CAID binary online continual detection."""

    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 100,
        backbone_name: str = None,
        online_lora_rank: int = 4,
        online_lora_layers: str = "all",
        **kwargs,
    ):
        super().__init__()
        del task_num

        self.num_classes = int(num_classes)
        self.backbone_name = backbone_name
        self.online_lora_rank = int(online_lora_rank)
        self.online_lora_layers = str(online_lora_layers)
        pretrained = bool(kwargs.get("pretrained", True))

        assert backbone_name is not None, "backbone_name must be specified"
        if hasattr(vit, backbone_name):
            logger.info("Using custom ViT model: %s", backbone_name)
            self.add_module(
                "backbone",
                getattr(vit, backbone_name)(pretrained=pretrained, num_classes=num_classes),
            )
        else:
            logger.info("Using timm model: %s", backbone_name)
            self.add_module(
                "backbone",
                timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes),
            )

        for _, param in self.backbone.named_parameters():
            param.requires_grad = False
        self._enable_classifier_head()

        self.lora_layers = []
        self._inject_lora()
        self._omega: Dict[str, torch.Tensor] = {}
        self._omega_update_count = 0
        self.reset_omega_state()

    def _enable_classifier_head(self) -> None:
        for attr in ("fc", "head"):
            head = getattr(self.backbone, attr, None)
            if isinstance(head, nn.Module):
                for param in head.parameters():
                    param.requires_grad = True

    def _select_block_indices(self, depth: int):
        layer_spec = self.online_lora_layers.strip().lower()
        if layer_spec == "all":
            return list(range(depth))
        if layer_spec.startswith("last"):
            try:
                count = int(layer_spec[4:])
            except ValueError:
                return list(range(depth))
            return list(range(max(0, depth - count), depth))
        try:
            indices = [int(item.strip()) for item in layer_spec.split(",")]
            return [idx for idx in indices if 0 <= idx < depth]
        except ValueError:
            return list(range(depth))

    def _inject_lora(self) -> None:
        dim = self.backbone.embed_dim
        depth = len(self.backbone.blocks)
        target_indices = self._select_block_indices(depth)
        logger.info("Injecting official-style Online-LoRA into blocks: %s", target_indices)
        for idx, block in enumerate(self.backbone.blocks):
            if idx not in target_indices:
                continue
            adapter = OnlineLoRAQKVAdapter(block.attn.qkv, dim, self.online_lora_rank)
            block.attn.qkv = adapter
            self.lora_layers.append(adapter)

    def wnew_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            if (
                ".new_A_q." in name
                or ".new_B_q." in name
                or ".new_A_v." in name
                or ".new_B_v." in name
            ):
                yield name, param

    def reset_omega_state(self) -> None:
        self._omega = {
            name: torch.zeros_like(param.detach())
            for name, param in self.wnew_named_parameters()
        }
        self._omega_update_count = 0

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for name, param in self.wnew_named_parameters():
            omega = self._omega.get(name)
            if omega is None or omega.shape != param.shape:
                continue
            loss = loss + (omega.to(param.device) * param.pow(2)).sum()
        return loss

    def update_omega_from_gradients(self, gradients: Dict[str, torch.Tensor]) -> int:
        if not gradients:
            return 0
        if not self._omega:
            self.reset_omega_state()

        self._omega_update_count += 1
        weight = 1.0 / float(self._omega_update_count)
        updated = 0
        param_by_name = dict(self.wnew_named_parameters())
        for name, param in param_by_name.items():
            grad_score = gradients.get(name)
            if grad_score is None or tuple(grad_score.shape) != tuple(param.shape):
                continue
            old = self._omega.get(name)
            if old is None or tuple(old.shape) != tuple(param.shape):
                old = torch.zeros_like(param.detach())
            self._omega[name] = weight * grad_score.to(param.device) + (1.0 - weight) * old.to(param.device)
            updated += 1
        return updated

    def merge_and_reset_lora(self) -> None:
        for adapter in self.lora_layers:
            adapter.merge_and_reset_new()

    def export_importance_state(self):
        return {
            "omega": {name: value.detach().cpu() for name, value in self._omega.items()},
            "omega_update_count": int(self._omega_update_count),
        }

    def load_importance_state(self, state) -> None:
        omega = state.get("omega", {}) if isinstance(state, dict) else {}
        self.reset_omega_state()
        param_by_name = dict(self.wnew_named_parameters())
        loaded = 0
        for name, param in param_by_name.items():
            saved = omega.get(name)
            if torch.is_tensor(saved) and tuple(saved.shape) == tuple(param.shape):
                self._omega[name] = saved.to(param.device)
                loaded += 1
        self._omega_update_count = int(state.get("omega_update_count", 0)) if isinstance(state, dict) else 0
        logger.info(
            "Loaded Online-LoRA omega state for %s tensors | updates=%s",
            loaded,
            self._omega_update_count,
        )

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.backbone(inputs)
