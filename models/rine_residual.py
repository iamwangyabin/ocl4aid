import logging
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit


logger = logging.getLogger()


class LowRankResidualHead(nn.Module):
    def __init__(self, feature_dim: int, rank: int = 64, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.out = nn.Linear(rank, num_classes)

    def forward(self, z):
        return self.out(self.down(self.norm(z.float())))


class LinearResidualHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z.float())


class MLPResidualHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z.float())


class RINEResidual(nn.Module):
    """Frozen high-dimensional ViT features with progressive residual heads."""

    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 2,
        backbone_name: str = None,
        rine_residual_feature_layers="quartile",
        rine_residual_online_feature_layers=None,
        rine_residual_head_type: str = "lowrank",
        rine_residual_online_head_type: str = None,
        rine_residual_rank: int = 64,
        rine_residual_hidden_dim: int = 512,
        rine_residual_eval_mode: str = "task_oracle",
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if backbone_name is None:
            raise ValueError("backbone_name must be specified")
        if rine_residual_head_type not in {"lowrank", "linear", "mlp"}:
            raise ValueError(f"Unsupported rine_residual_head_type: {rine_residual_head_type}")
        if rine_residual_online_head_type in {None, "", "same"}:
            rine_residual_online_head_type = rine_residual_head_type
        if rine_residual_online_head_type not in {"lowrank", "linear", "mlp"}:
            raise ValueError(f"Unsupported rine_residual_online_head_type: {rine_residual_online_head_type}")

        self.task_num = int(task_num)
        self.num_classes = int(num_classes)
        self.head_type = str(rine_residual_head_type)
        self.online_head_type = str(rine_residual_online_head_type)
        self.rank = int(rine_residual_rank)
        self.hidden_dim = int(rine_residual_hidden_dim)
        self.eval_mode = str(rine_residual_eval_mode)
        self.active_stage = 0
        if self.eval_mode not in {"max_fake", "max_confidence", "task_oracle"}:
            raise ValueError(f"Unsupported rine_residual_eval_mode: {self.eval_mode}")

        if hasattr(vit, backbone_name):
            logger.info("Using custom ViT model: %s", backbone_name)
            self.backbone = getattr(vit, backbone_name)(
                pretrained=pretrained,
                num_classes=num_classes,
            )
        else:
            logger.info("Using timm model: %s", backbone_name)
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.depth = len(self.backbone.blocks)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", self.backbone.num_features))
        self.feature_layers = self._resolve_feature_layers(rine_residual_feature_layers)
        if rine_residual_online_feature_layers in {None, "", "same"}:
            rine_residual_online_feature_layers = rine_residual_feature_layers
        self.online_feature_layers = self._resolve_feature_layers(rine_residual_online_feature_layers)
        self.feature_dim = len(self.feature_layers) * self.embed_dim
        self.online_feature_dim = len(self.online_feature_layers) * self.embed_dim
        logger.info(
            "RINE-Residual base feature layers: %s of %s blocks | feature_dim=%s",
            [layer + 1 for layer in self.feature_layers],
            self.depth,
            self.feature_dim,
        )
        logger.info(
            "RINE-Residual online feature layers: %s of %s blocks | feature_dim=%s",
            [layer + 1 for layer in self.online_feature_layers],
            self.depth,
            self.online_feature_dim,
        )

        self.base_head = self._make_head(self.head_type)
        self.residual_heads = nn.ModuleList()

    def set_backbone_trainable(self, trainable: bool):
        for param in self.backbone.parameters():
            param.requires_grad = bool(trainable)

    @property
    def online_features_match_base(self):
        return self.online_feature_layers == self.feature_layers

    def _make_head(self, head_type=None, feature_dim=None):
        head_type = self.head_type if head_type is None else str(head_type)
        feature_dim = self.feature_dim if feature_dim is None else int(feature_dim)
        if head_type == "linear":
            return LinearResidualHead(feature_dim, self.num_classes)
        if head_type == "mlp":
            return MLPResidualHead(feature_dim, self.hidden_dim, self.num_classes)
        return LowRankResidualHead(feature_dim, self.rank, self.num_classes)

    def add_residual_head(self, stage_id: int):
        stage_id = int(stage_id)
        while len(self.residual_heads) < stage_id:
            head = self._make_head(self.online_head_type, self.online_feature_dim)
            if (
                self.online_head_type == self.head_type
                and self.online_feature_dim == self.feature_dim
            ):
                try:
                    head.load_state_dict(self.base_head.state_dict())
                except RuntimeError:
                    logger.warning("Failed to initialize residual head from base head; using random init.")
            self.residual_heads.append(head)
        self.active_stage = stage_id
        self.set_train_stage(stage_id)
        return self.residual_heads[stage_id - 1]

    def set_train_stage(self, stage_id: int):
        stage_id = int(stage_id)
        for param in self.base_head.parameters():
            param.requires_grad = stage_id == 0
        for idx, head in enumerate(self.residual_heads, start=1):
            for param in head.parameters():
                param.requires_grad = idx == stage_id

    def current_head(self):
        if self.active_stage == 0:
            return self.base_head
        return self.residual_heads[self.active_stage - 1]

    def extract_z(self, inputs: torch.Tensor, feature_layers=None) -> torch.Tensor:
        layers = self.feature_layers if feature_layers is None else list(feature_layers)
        return self._extract_feature_layers(inputs, layers)

    def extract_base_and_online_z(self, inputs: torch.Tensor):
        if self.online_features_match_base:
            z = self.extract_z(inputs, self.feature_layers)
            return z, z
        base_layers = list(self.feature_layers)
        online_layers = list(self.online_feature_layers)
        union_layers = []
        for layer in base_layers + online_layers:
            if layer not in union_layers:
                union_layers.append(layer)
        layer_outputs = self._extract_feature_layer_map(inputs, union_layers)
        base_z = torch.cat([layer_outputs[layer] for layer in base_layers], dim=1).float()
        online_z = torch.cat([layer_outputs[layer] for layer in online_layers], dim=1).float()
        return base_z, online_z

    def _extract_feature_layers(self, inputs: torch.Tensor, layers) -> torch.Tensor:
        layer_outputs = self._extract_feature_layer_map(inputs, layers)
        return torch.cat([layer_outputs[layer] for layer in layers], dim=1).float()

    def _extract_feature_layer_map(self, inputs: torch.Tensor, layers):
        x = self.backbone.patch_embed(inputs)
        if getattr(self.backbone, "cls_token", None) is not None:
            x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.backbone.pos_embed[:, : x.size(1), :]
        x = self.backbone.pos_drop(x + pos_embed)

        selected_layers = set(layers)
        cls_tokens = {}
        for layer_idx, block in enumerate(self.backbone.blocks):
            x = block(x)
            if layer_idx not in selected_layers:
                continue
            if getattr(self.backbone, "cls_token", None) is not None:
                cls_tokens[layer_idx] = x[:, 0]
            else:
                cls_tokens[layer_idx] = x.mean(dim=1)
        return cls_tokens

    def expert_logits_from_z(self, z: torch.Tensor, online_z: torch.Tensor = None) -> torch.Tensor:
        online_z = z if online_z is None else online_z
        experts = [self.base_head(z)]
        for head in self.residual_heads:
            experts.append(head(online_z))
        return torch.stack(experts, dim=1)

    def eval_logits_from_z(self, z: torch.Tensor, online_z: torch.Tensor = None) -> torch.Tensor:
        online_z = z if online_z is None else online_z
        if not self.residual_heads:
            return self.base_head(z)
        expert_logits = self.expert_logits_from_z(z, online_z=online_z)
        if self.eval_mode == "max_confidence":
            expert_scores = torch.softmax(expert_logits, dim=-1).max(dim=-1).values
        else:
            expert_scores = torch.softmax(expert_logits, dim=-1)[:, :, 1]
        expert_ids = torch.argmax(expert_scores, dim=1)
        batch_ids = torch.arange(z.size(0), device=z.device)
        return expert_logits[batch_ids, expert_ids]

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        z, online_z = self.extract_base_and_online_z(inputs)
        return self.eval_logits_from_z(z, online_z=online_z)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def _resolve_feature_layers(self, spec):
        depth = self.depth
        if spec is None:
            return self._quartile_feature_layers(depth)
        if isinstance(spec, (list, tuple)):
            raw_layers = spec
        else:
            spec = str(spec).strip().lower()
            if spec in {"quartile", "quartiles"}:
                return self._quartile_feature_layers(depth)
            if spec == "all":
                return list(range(depth))
            if spec == "last4":
                return list(range(max(0, depth - 4), depth))
            raw_layers = [item.strip() for item in spec.split(",") if item.strip()]
        if not raw_layers:
            raise ValueError("rine_residual_feature_layers must not be empty")

        layers = []
        for item in raw_layers:
            layer = int(item) - 1
            if layer < 0 or layer >= depth:
                raise ValueError(
                    f"rine_residual_feature_layers values must be 1..{depth}, got {item}"
                )
            if layer not in layers:
                layers.append(layer)
        return layers

    @staticmethod
    def _quartile_feature_layers(depth: int):
        layers = []
        for pos in (1, 2, 3, 4):
            layer = math.ceil(depth * pos / 4.0) - 1
            layer = min(max(layer, 0), depth - 1)
            if layer not in layers:
                layers.append(layer)
        return layers
