import logging

import torch

from .rigev1 import RIGEv1 as RIGEv1Model


logger = logging.getLogger()


class RIGEv2(RIGEv1Model):
    """RIGEv2 with head-weight-selected compressed online features.

    The base detector keeps the raw 3072-dim feature representation. Online
    residual experts and replay use a fixed subset of those features selected
    from the trained base head weights.
    """

    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 2,
        backbone_name: str = None,
        rigev2_feature_layers="quartile",
        rigev2_online_feature_layers=None,
        rigev2_head_type: str = "lowrank",
        rigev2_online_head_type: str = "lowrank",
        rigev2_rank: int = 16,
        rigev2_online_rank: int = 4,
        rigev2_hidden_dim: int = 512,
        rigev2_eval_mode: str = "feature_gaussian",
        rigev2_alpha_init: float = 0.2,
        rigev2_replay_dim: int = 1536,
        pretrained: bool = True,
        **kwargs,
    ):
        if rigev2_online_feature_layers in {None, "", "same"}:
            rigev2_online_feature_layers = rigev2_feature_layers
        if rigev2_online_feature_layers != rigev2_feature_layers:
            raise ValueError(
                "RIGEv2 requires base and online feature layers to match; "
                "online compression is applied after raw feature extraction."
            )

        # RIGEv1 has a single rank parameter. Build the base head first with
        # rigev2_rank, then switch rank before online residual heads are added.
        super().__init__(
            task_num=task_num,
            num_classes=num_classes,
            backbone_name=backbone_name,
            rigev1_feature_layers=rigev2_feature_layers,
            rigev1_online_feature_layers=rigev2_online_feature_layers,
            rigev1_head_type=rigev2_head_type,
            rigev1_online_head_type=rigev2_online_head_type,
            rigev1_rank=rigev2_rank,
            rigev1_hidden_dim=rigev2_hidden_dim,
            rigev1_eval_mode=rigev2_eval_mode,
            rigev1_alpha_init=rigev2_alpha_init,
            pretrained=pretrained,
            **kwargs,
        )

        self.raw_feature_dim = int(self.feature_dim)
        self.raw_online_feature_dim = int(self.online_feature_dim)
        self.online_feature_dim = int(rigev2_replay_dim)
        if self.online_feature_dim <= 0 or self.online_feature_dim > self.raw_online_feature_dim:
            raise ValueError(
                f"rigev2_replay_dim must be in 1..{self.raw_online_feature_dim}, "
                f"got {self.online_feature_dim}"
            )

        self.rank = int(rigev2_online_rank)
        if self.rank <= 0:
            raise ValueError(f"rigev2_online_rank must be positive, got {self.rank}")
        self.online_feature_space = "headweight"
        self.register_buffer(
            "online_feature_indices",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.set_train_stage(0)
        logger.info(
            "RIGEv2 initialized | raw_dim=%s | online_dim=%s | "
            "selector=headweight | base_rank=%s | online_rank=%s",
            self.raw_feature_dim,
            self.online_feature_dim,
            rigev2_rank,
            self.rank,
        )

    def set_online_feature_indices(self, indices: torch.Tensor):
        indices = indices.detach().to(
            device=self.online_feature_indices.device,
            dtype=torch.long,
        )
        if indices.dim() != 1:
            raise ValueError("RIGEv2 online feature indices must be a 1D tensor")
        if int(indices.numel()) != int(self.online_feature_dim):
            raise ValueError(
                f"RIGEv2 expected {self.online_feature_dim} selected features, "
                f"got {indices.numel()}"
            )
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= self.raw_online_feature_dim:
            raise ValueError(
                f"RIGEv2 selected feature indices must be in [0, {self.raw_online_feature_dim})"
            )
        self.online_feature_indices = indices

    def online_features_from_raw(self, raw_features: torch.Tensor) -> torch.Tensor:
        indices = self.online_feature_indices
        if int(indices.numel()) != int(self.online_feature_dim):
            # Before the base head is trained or when a legacy base checkpoint
            # is loaded, use a deterministic temporary subset until the trainer
            # finalizes head-weight selection.
            indices = torch.linspace(
                0,
                raw_features.size(1) - 1,
                steps=self.online_feature_dim,
                device=raw_features.device,
                dtype=torch.long,
            )
        else:
            indices = indices.to(raw_features.device)
        return raw_features.index_select(1, indices)

    def extract_base_and_online_z(self, inputs: torch.Tensor):
        raw_z = self.extract_z(inputs, self.feature_layers)
        return raw_z, self.online_features_from_raw(raw_z)
