import logging
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit


logger = logging.getLogger()


class RineSideGauss(nn.Module):
    """Frozen intermediate-CLS features with optional per-stage projected heads."""

    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 2,
        backbone_name: str = None,
        rine_gauss_var_floor: float = 1e-4,
        rine_gauss_min_count: int = 2,
        rine_gauss_aggregation: str = "logmeanexp",
        rine_gauss_feature_layers="quartile",
        rine_gauss_projector_dim: int = 0,
        rine_gauss_hidden_dim: int = 256,
        rine_gauss_head_aggregation: str = "mean_logit",
        rine_gauss_cov_shrink: float = 0.01,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if backbone_name is None:
            raise ValueError("backbone_name must be specified")
        if rine_gauss_aggregation not in {"logmeanexp", "logsumexp", "mean", "max"}:
            raise ValueError(f"Unsupported rine_gauss_aggregation: {rine_gauss_aggregation}")
        if rine_gauss_head_aggregation not in {"mean_logit", "max_logit", "noisy_or"}:
            raise ValueError(
                f"Unsupported rine_gauss_head_aggregation: {rine_gauss_head_aggregation}"
            )

        self.task_num = int(task_num)
        self.num_classes = int(num_classes)
        self.var_floor = float(rine_gauss_var_floor)
        self.min_count = int(rine_gauss_min_count)
        self.aggregation = rine_gauss_aggregation
        self.projector_dim = int(rine_gauss_projector_dim or 0)
        self.hidden_dim = int(rine_gauss_hidden_dim or 256)
        self.head_aggregation = rine_gauss_head_aggregation
        self.cov_shrink = float(rine_gauss_cov_shrink)

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
        self.feature_layers = self._resolve_feature_layers(rine_gauss_feature_layers)
        self.feature_dim = len(self.feature_layers) * self.embed_dim
        logger.info(
            "RINE-side Gaussian feature layers: %s of %s blocks",
            [layer + 1 for layer in self.feature_layers],
            self.depth,
        )

        self.register_buffer("counts", torch.zeros(self.task_num, self.num_classes))
        self.register_buffer("means", torch.zeros(self.task_num, self.num_classes, self.feature_dim))
        self.register_buffer("m2", torch.zeros(self.task_num, self.num_classes, self.feature_dim))

        if self.projector_dim > 0:
            self.projectors = nn.ModuleList(
                [
                    nn.Linear(self.feature_dim, self.projector_dim, bias=False)
                    for _ in range(self.task_num)
                ]
            )
            self.detectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.projector_dim),
                        nn.GELU(),
                        nn.Linear(self.projector_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, self.num_classes),
                    )
                    for _ in range(self.task_num)
                ]
            )
            self.register_buffer("head_active", torch.zeros(self.task_num, dtype=torch.bool))
            self.register_buffer(
                "proj_counts",
                torch.zeros(self.task_num, self.num_classes),
            )
            self.register_buffer(
                "proj_means",
                torch.zeros(self.task_num, self.num_classes, self.projector_dim),
            )
            self.register_buffer(
                "proj_m2",
                torch.zeros(
                    self.task_num,
                    self.num_classes,
                    self.projector_dim,
                    self.projector_dim,
                ),
            )
            self.set_train_stage(None)
        else:
            self.projectors = nn.ModuleList()
            self.detectors = nn.ModuleList()

        # Keeps DDP happy for a statistics-only method. The trainer never steps it.
        self.ddp_anchor = nn.Parameter(torch.zeros(()))

    def _resolve_feature_layers(self, spec):
        depth = len(self.backbone.blocks)
        if depth <= 0:
            raise ValueError("RINE-side Gaussian requires a block-based ViT backbone")

        if spec is None:
            spec = "quartile"

        if isinstance(spec, str):
            text = spec.strip().lower()
            if text in {"all", "every"}:
                return list(range(depth))
            if text in {"quartile", "quarters", "four", "4"}:
                return self._quartile_feature_layers(depth)
            if text == "last4":
                return list(range(max(depth - 4, 0), depth))
            tokens = text.replace(",", " ").split()
            if not tokens:
                raise ValueError("rine_gauss_feature_layers must not be empty")
            try:
                values = [int(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(
                    "rine_gauss_feature_layers must be 'quartile', 'all', 'last4', "
                    "or a comma-separated 1-based block list"
                ) from exc
        else:
            values = [int(item) for item in spec]

        if all(1 <= value <= depth for value in values):
            layers = [value - 1 for value in values]
        elif all(0 <= value < depth for value in values):
            layers = values
        else:
            raise ValueError(
                f"rine_gauss_feature_layers values must be 1..{depth} "
                f"(or 0..{depth - 1} for zero-based lists), got {values}"
            )

        layers = sorted(set(layers))
        if not layers:
            raise ValueError("rine_gauss_feature_layers selected no layers")
        return layers

    @staticmethod
    def _quartile_feature_layers(depth: int):
        layers = []
        for pos in range(1, 5):
            layer = math.ceil(depth * pos / 4.0) - 1
            layer = min(max(layer, 0), depth - 1)
            if layer not in layers:
                layers.append(layer)
        return layers

    @torch.no_grad()
    def extract_z(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return z = concat(cls_1, ..., cls_L) from frozen ViT block outputs."""
        self.backbone.eval()
        x = self.backbone.patch_embed(inputs)
        if getattr(self.backbone, "cls_token", None) is not None:
            x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.backbone.pos_embed[:, : x.size(1), :]
        x = self.backbone.pos_drop(x + pos_embed)

        cls_tokens = []
        selected_layers = set(self.feature_layers)
        for layer_idx, block in enumerate(self.backbone.blocks):
            x = block(x)
            if layer_idx not in selected_layers:
                continue
            if getattr(self.backbone, "cls_token", None) is not None:
                cls_tokens.append(x[:, 0])
            else:
                cls_tokens.append(x.mean(dim=1))
        return torch.cat(cls_tokens, dim=1).float()

    @torch.no_grad()
    def update_statistics(self, stage_id: int, z: torch.Tensor, labels: torch.Tensor) -> None:
        if stage_id < 0 or stage_id >= self.task_num:
            raise ValueError(f"stage_id must be in [0, {self.task_num}), got {stage_id}")
        if z.numel() == 0:
            return

        z = z.detach().float()
        labels = labels.detach().long()
        for class_idx in labels.unique(sorted=True).tolist():
            if class_idx < 0 or class_idx >= self.num_classes:
                continue
            class_z = z[labels == class_idx]
            if class_z.numel() == 0:
                continue
            self._merge_batch(stage_id, int(class_idx), class_z)

    @torch.no_grad()
    def _merge_batch(self, stage_id: int, class_idx: int, z: torch.Tensor) -> None:
        n_a = self.counts[stage_id, class_idx]
        mean_a = self.means[stage_id, class_idx]
        m2_a = self.m2[stage_id, class_idx]

        n_b = torch.tensor(float(z.size(0)), device=z.device, dtype=mean_a.dtype)
        mean_b = z.mean(dim=0)
        diff_b = z - mean_b
        m2_b = (diff_b * diff_b).sum(dim=0)

        n = n_a + n_b
        delta = mean_b - mean_a
        mean = mean_a + delta * (n_b / n)
        m2 = m2_a + m2_b + delta * delta * (n_a * n_b / n)

        self.counts[stage_id, class_idx] = n
        self.means[stage_id, class_idx].copy_(mean)
        self.m2[stage_id, class_idx].copy_(m2)

    def gaussian_logits_from_z(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        valid_stages = torch.all(self.counts >= float(self.min_count), dim=1)
        if not torch.any(valid_stages):
            return z.new_zeros(z.size(0), self.num_classes) + self.ddp_anchor * 0.0

        counts = self.counts[valid_stages].clamp_min(2.0)
        means = self.means[valid_stages]
        variances = self.m2[valid_stages] / (counts - 1.0).unsqueeze(-1)
        variances = variances.clamp_min(self.var_floor)

        centered = z[:, None, None, :] - means[None, :, :, :]
        log_probs = -0.5 * (
            (centered * centered / variances[None, :, :, :])
            + torch.log(variances[None, :, :, :])
        ).sum(dim=-1)

        if self.aggregation == "max":
            logits = log_probs.max(dim=1).values
        elif self.aggregation == "mean":
            logits = log_probs.mean(dim=1)
        elif self.aggregation == "logsumexp":
            logits = torch.logsumexp(log_probs, dim=1)
        else:
            logits = torch.logsumexp(log_probs, dim=1) - math.log(log_probs.size(1))

        return logits + self.ddp_anchor * 0.0

    def set_train_stage(self, stage_id) -> None:
        if self.projector_dim <= 0:
            return

        active_stage = None if stage_id is None else int(stage_id)
        for idx, projector in enumerate(self.projectors):
            train_projector = active_stage is not None and idx == active_stage
            for param in projector.parameters():
                param.requires_grad = train_projector

        for idx, detector in enumerate(self.detectors):
            train_detector = active_stage is not None and (
                idx == active_stage or bool(self.head_active[idx].item())
            )
            for param in detector.parameters():
                param.requires_grad = train_detector

    @torch.no_grad()
    def begin_stage(self, stage_id: int) -> None:
        if self.projector_dim <= 0:
            return
        if stage_id < 0 or stage_id >= self.task_num:
            raise ValueError(f"stage_id must be in [0, {self.task_num}), got {stage_id}")
        self.head_active[stage_id] = True
        self.set_train_stage(stage_id)

    def active_head_ids(self):
        if self.projector_dim <= 0:
            return []
        return [
            idx
            for idx, is_active in enumerate(self.head_active.detach().cpu().tolist())
            if bool(is_active)
        ]

    def project_z(self, stage_id: int, z: torch.Tensor) -> torch.Tensor:
        return self.projectors[int(stage_id)](z.float())

    def detector_logits_from_h(self, stage_id: int, h: torch.Tensor) -> torch.Tensor:
        return self.detectors[int(stage_id)](h.float())

    def head_logits_from_z(self, stage_id: int, z: torch.Tensor) -> torch.Tensor:
        return self.detector_logits_from_h(stage_id, self.project_z(stage_id, z))

    def projected_logits_from_z(self, z: torch.Tensor, head_ids=None) -> torch.Tensor:
        if self.projector_dim <= 0:
            return self.gaussian_logits_from_z(z)

        if head_ids is None:
            head_ids = self.active_head_ids()
        head_ids = [int(item) for item in head_ids if 0 <= int(item) < self.task_num]
        if not head_ids:
            return z.new_zeros(z.size(0), self.num_classes) + self.ddp_anchor * 0.0

        stacked_logits = torch.stack(
            [self.head_logits_from_z(stage_id, z) for stage_id in head_ids],
            dim=1,
        )
        return self._aggregate_projected_logits(stacked_logits) + self.ddp_anchor * 0.0

    def _aggregate_projected_logits(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        if self.head_aggregation == "mean_logit":
            return stacked_logits.mean(dim=1)

        if self.num_classes != 2:
            if self.head_aggregation == "max_logit":
                return stacked_logits.max(dim=1).values
            return stacked_logits.mean(dim=1)

        margins = stacked_logits[:, :, 1] - stacked_logits[:, :, 0]
        if self.head_aggregation == "max_logit":
            fake_logit = margins.max(dim=1).values
        else:
            fake_prob = torch.sigmoid(margins).clamp(1e-6, 1.0 - 1e-6)
            noisy_or = 1.0 - torch.prod(1.0 - fake_prob, dim=1)
            noisy_or = noisy_or.clamp(1e-6, 1.0 - 1e-6)
            fake_logit = torch.logit(noisy_or)
        real_logit = torch.zeros_like(fake_logit)
        return torch.stack([real_logit, fake_logit], dim=-1)

    @torch.no_grad()
    def reset_projected_statistics(self, stage_id: int) -> None:
        if self.projector_dim <= 0:
            return
        self.proj_counts[stage_id].zero_()
        self.proj_means[stage_id].zero_()
        self.proj_m2[stage_id].zero_()

    @torch.no_grad()
    def update_projected_statistics(self, stage_id: int, h: torch.Tensor, labels: torch.Tensor) -> None:
        if self.projector_dim <= 0:
            return
        if stage_id < 0 or stage_id >= self.task_num:
            raise ValueError(f"stage_id must be in [0, {self.task_num}), got {stage_id}")
        h = h.detach().float()
        labels = labels.detach().long()
        for class_idx in labels.unique(sorted=True).tolist():
            if class_idx < 0 or class_idx >= self.num_classes:
                continue
            class_h = h[labels == class_idx]
            if class_h.numel() == 0:
                continue
            self._merge_projected_batch(stage_id, int(class_idx), class_h)

    @torch.no_grad()
    def update_projected_statistics_from_z(self, stage_id: int, z: torch.Tensor, labels: torch.Tensor) -> None:
        if self.projector_dim <= 0:
            return
        h = self.project_z(stage_id, z.detach().float())
        self.update_projected_statistics(stage_id, h, labels)

    @torch.no_grad()
    def _merge_projected_batch(self, stage_id: int, class_idx: int, h: torch.Tensor) -> None:
        n_a = self.proj_counts[stage_id, class_idx]
        mean_a = self.proj_means[stage_id, class_idx]
        m2_a = self.proj_m2[stage_id, class_idx]

        n_b = torch.tensor(float(h.size(0)), device=h.device, dtype=mean_a.dtype)
        mean_b = h.mean(dim=0)
        centered = h - mean_b.unsqueeze(0)
        m2_b = centered.T @ centered

        n = n_a + n_b
        delta = mean_b - mean_a
        mean = mean_a + delta * (n_b / n)
        m2 = m2_a + m2_b + torch.outer(delta, delta) * (n_a * n_b / n)

        self.proj_counts[stage_id, class_idx] = n
        self.proj_means[stage_id, class_idx].copy_(mean)
        self.proj_m2[stage_id, class_idx].copy_(m2)

    def has_projected_replay(self, stage_id: int) -> bool:
        if self.projector_dim <= 0:
            return False
        if stage_id < 0 or stage_id >= self.task_num:
            return False
        return bool(torch.all(self.proj_counts[stage_id] >= float(self.min_count)).item())

    @torch.no_grad()
    def sample_projected_replay(self, stage_id: int, n_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.projector_dim <= 0 or n_per_class <= 0:
            empty_x = self.proj_means.new_zeros((0, self.projector_dim))
            empty_y = torch.zeros(0, dtype=torch.long, device=empty_x.device)
            return empty_x, empty_y
        if not self.has_projected_replay(stage_id):
            empty_x = self.proj_means.new_zeros((0, self.projector_dim))
            empty_y = torch.zeros(0, dtype=torch.long, device=empty_x.device)
            return empty_x, empty_y

        samples = []
        labels = []
        for class_idx in range(self.num_classes):
            count = self.proj_counts[stage_id, class_idx].clamp_min(2.0)
            mean = self.proj_means[stage_id, class_idx]
            cov = self.proj_m2[stage_id, class_idx] / (count - 1.0)
            diag_mean = torch.diagonal(cov).mean().clamp_min(self.var_floor)
            eye = torch.eye(self.projector_dim, device=cov.device, dtype=cov.dtype)
            cov = cov + max(self.cov_shrink, 1e-6) * diag_mean * eye
            chol = torch.linalg.cholesky(cov)
            eps = torch.randn(n_per_class, self.projector_dim, device=mean.device, dtype=mean.dtype)
            samples.append(eps @ chol.T + mean.unsqueeze(0))
            labels.append(
                torch.full(
                    (n_per_class,),
                    class_idx,
                    dtype=torch.long,
                    device=mean.device,
                )
            )
        return torch.cat(samples, dim=0), torch.cat(labels, dim=0)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        z = self.extract_z(inputs)
        if self.projector_dim > 0:
            return self.projected_logits_from_z(z)
        return self.gaussian_logits_from_z(z)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)
