import logging

import torch

from .rigev1 import RIGEv1


logger = logging.getLogger()


class RIGEv2(RIGEv1):
    """Residual Incremental Gaussian Experts v2.

    V2 keeps RIGEv1's residual expert training and Gaussian routing, but stores
    only a head-weight-selected subset of the raw online features in replay.
    """

    def _cfg(self, name, default):
        return getattr(
            self,
            f"rigev2_{name}",
            getattr(self, f"rigev1_{name}", default),
        )

    def _expected_route_feature_dim(self):
        route_space = str(self._cfg("route_space", "online") or "online").lower()
        model = self.model_without_ddp
        if route_space == "raw":
            return int(getattr(model, "raw_online_feature_dim", 0) or 0)
        return int(getattr(model, "online_feature_dim", 0) or 0)

    def _load_checkpoint_method_state(self, state):
        super()._load_checkpoint_method_state(state)
        selected_indices = state.get("rigev2_online_feature_indices")
        if torch.is_tensor(selected_indices):
            try:
                self.model_without_ddp.set_online_feature_indices(selected_indices.cpu())
                logger.info(
                    "Loaded RIGEv2 online feature indices from checkpoint | dim=%s",
                    int(selected_indices.numel()),
                )
            except (AttributeError, ValueError) as exc:
                logger.warning("Ignored incompatible RIGEv2 feature indices: %s", exc)

        expected_dim = self._expected_route_feature_dim()
        if expected_dim <= 0:
            return
        dropped = []
        for expert_id, stats in list(self._route_stats_by_expert.items()):
            mean = stats.get("mean")
            if not torch.is_tensor(mean) or int(mean.numel()) != expected_dim:
                dropped.append(int(expert_id))
                self._route_stats_by_expert.pop(expert_id, None)
        if dropped:
            logger.info(
                "Dropped incompatible RIGEv2 route stats from checkpoint: experts=%s expected_dim=%s",
                dropped,
                expected_dim,
            )

    def _checkpoint_method_state(self):
        state = super()._checkpoint_method_state()
        indices = getattr(self.model_without_ddp, "online_feature_indices", None)
        if torch.is_tensor(indices) and indices.numel() > 0:
            state["rigev2_online_feature_indices"] = indices.detach().cpu()
        return state

    def _after_base_checkpoint_loaded(self, checkpoint):
        self._ensure_online_feature_indices()
        # RIGEv1 may rebuild base route stats here. Do it after the V2 indices
        # are fixed so Gaussian routing is built in the same feature space used
        # by online experts and replay.
        super()._after_base_checkpoint_loaded(checkpoint)

    def after_base_stage_train(self, base_stage_id):
        self._ensure_online_feature_indices()
        return super().after_base_stage_train(base_stage_id)

    def _ensure_online_feature_indices(self):
        model = self.model_without_ddp
        indices = getattr(model, "online_feature_indices", None)
        expected_dim = int(getattr(model, "online_feature_dim", 0) or 0)
        if torch.is_tensor(indices) and int(indices.numel()) == expected_dim:
            return

        scores = self._head_weight_feature_scores()
        if scores is None:
            raise RuntimeError("RIGEv2 cannot select online features before the base head exists")
        selected = self._select_headweight_indices(scores, expected_dim)
        model.set_online_feature_indices(selected)
        logger.info(
            "RIGEv2 online feature selection finalized | raw_dim=%s | selected_dim=%s | "
            "score_mean=%.6f | selected_score_mean=%.6f",
            int(scores.numel()),
            int(selected.numel()),
            float(scores.mean().item()),
            float(scores[selected].mean().item()),
        )

    def _head_weight_feature_scores(self):
        head = getattr(self.model_without_ddp, "base_head", None)
        weight = None
        if hasattr(head, "down"):
            weight = getattr(head.down, "weight", None)
        elif hasattr(head, "net"):
            for module in head.net:
                if hasattr(module, "weight") and int(module.weight.dim()) == 2:
                    weight = module.weight
                    break
        if weight is None or not torch.is_tensor(weight):
            return None
        scores = weight.detach().float().pow(2).sum(dim=0).cpu()
        return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _select_headweight_indices(self, scores, selected_dim: int):
        selected_dim = int(selected_dim)
        feature_dim = int(scores.numel())
        if selected_dim <= 0 or selected_dim > feature_dim:
            raise ValueError(
                f"RIGEv2 selected_dim must be in 1..{feature_dim}, got {selected_dim}"
            )

        block_dim = int(self._cfg("feature_block_dim", 768) or 0)
        if (
            block_dim > 0
            and feature_dim % block_dim == 0
            and selected_dim % (feature_dim // block_dim) == 0
        ):
            block_count = feature_dim // block_dim
            per_block = selected_dim // block_count
            selected = []
            for block_id in range(block_count):
                start = block_id * block_dim
                block_scores = scores[start : start + block_dim]
                selected.append(torch.topk(block_scores, k=per_block).indices + start)
            indices = torch.cat(selected, dim=0)
        else:
            indices = torch.topk(scores, k=selected_dim).indices
        return torch.sort(indices.long()).values
