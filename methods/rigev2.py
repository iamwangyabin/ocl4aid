import logging

import torch

from .rigev1 import RIGEv1


logger = logging.getLogger()


class RIGEv2(RIGEv1):
    """Residual Incremental Gaussian Experts v2.

    V2 keeps RIGEv1's residual expert training and Gaussian routing, but stores
    only a head-weight-selected subset of the raw online features in replay.
    """

    _BASE_STATS_SELECTOR_VERSION = 1
    _BASE_STATS_SELECTOR_STATE_KEY = "rigev2_base_stats_selector"

    def _cfg(self, name, default):
        return getattr(
            self,
            f"rigev2_{name}",
            getattr(self, f"rigev1_{name}", default),
        )

    def _validate_route_space(self):
        route_space = str(self._cfg("route_space", "online") or "online").lower()
        if route_space != "online":
            raise ValueError(
                "RIGEv2 Gaussian routing uses the compressed online features; "
                f"rigev2_route_space must be 'online', got {route_space!r}"
            )

    def online_before_task(self, task_id):
        self._validate_route_space()
        return super().online_before_task(task_id)

    def _expected_route_feature_dim(self):
        self._validate_route_space()
        model = self.model_without_ddp
        return int(getattr(model, "online_feature_dim", 0) or 0)

    def _load_checkpoint_method_state(self, state):
        super()._load_checkpoint_method_state(state)
        self._rigev2_base_stats_selector = self._normalize_base_stats_selector(
            state.get(self._BASE_STATS_SELECTOR_STATE_KEY)
        )
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
        marker = getattr(self, "_rigev2_base_stats_selector", None)
        if marker is not None:
            state[self._BASE_STATS_SELECTOR_STATE_KEY] = {
                "version": int(marker["version"]),
                "base_stage_id": int(marker["base_stage_id"]),
                "online_feature_indices": marker["online_feature_indices"].detach().cpu(),
            }
        return state

    def _after_base_checkpoint_loaded(self, checkpoint):
        self._ensure_online_feature_indices()
        base_stage_id = self._base_stage_id()
        if base_stage_id is None:
            return super()._after_base_checkpoint_loaded(checkpoint)

        stats_complete = (
            int(base_stage_id) in self._route_stats_by_expert
            and int(base_stage_id) in self._decision_thresholds
        )
        selector_matches = self._base_stats_selector_matches_current(base_stage_id)
        if not selector_matches or not stats_complete:
            reason = "selector marker missing or mismatched"
            if selector_matches:
                reason = "checkpoint statistics incomplete"
            logger.info(
                "Rebuilding RIGEv2 base statistics in finalized selector space | "
                "stage=%s | reason=%s",
                base_stage_id,
                reason,
            )
            self._clear_stage_route_and_threshold_statistics(base_stage_id)
            result = super()._after_base_checkpoint_loaded(checkpoint)
            self._record_base_stats_selector(base_stage_id)
            return result

        return super()._after_base_checkpoint_loaded(checkpoint)

    def after_base_stage_train(self, base_stage_id):
        self._ensure_online_feature_indices()
        # Base training uses a deterministic temporary subset before the head is
        # ready. Rebuild routing and calibration only after finalizing the
        # head-weight selector so no temporary-coordinate statistics survive.
        self._clear_stage_route_and_threshold_statistics(base_stage_id)
        self._rebuild_stage_statistics_from_train_data(base_stage_id)
        self._record_base_stats_selector(base_stage_id)
        return super().after_base_stage_train(base_stage_id)

    def _normalize_base_stats_selector(self, marker):
        if not isinstance(marker, dict):
            return None
        indices = marker.get("online_feature_indices")
        if not torch.is_tensor(indices):
            return None
        try:
            version = int(marker.get("version"))
            base_stage_id = int(marker.get("base_stage_id"))
        except (TypeError, ValueError):
            return None
        if version != self._BASE_STATS_SELECTOR_VERSION:
            return None
        return {
            "version": version,
            "base_stage_id": base_stage_id,
            "online_feature_indices": indices.detach().cpu().long().clone(),
        }

    def _base_stats_selector_matches_current(self, base_stage_id):
        marker = getattr(self, "_rigev2_base_stats_selector", None)
        if marker is None or int(marker["base_stage_id"]) != int(base_stage_id):
            return False
        current = getattr(self.model_without_ddp, "online_feature_indices", None)
        if not torch.is_tensor(current):
            return False
        return torch.equal(
            marker["online_feature_indices"],
            current.detach().cpu().long(),
        )

    def _record_base_stats_selector(self, base_stage_id):
        base_stage_id = int(base_stage_id)
        indices = getattr(self.model_without_ddp, "online_feature_indices", None)
        stats_complete = (
            base_stage_id in self._route_stats_by_expert
            and base_stage_id in self._decision_thresholds
        )
        if not stats_complete or not torch.is_tensor(indices) or indices.numel() == 0:
            self._rigev2_base_stats_selector = None
            return
        self._rigev2_base_stats_selector = {
            "version": self._BASE_STATS_SELECTOR_VERSION,
            "base_stage_id": base_stage_id,
            "online_feature_indices": indices.detach().cpu().long().clone(),
        }

    def _clear_stage_route_and_threshold_statistics(self, stage_id):
        stage_id = int(stage_id)
        self._route_stats_by_expert.pop(stage_id, None)
        self._threshold_scores_by_expert.pop(stage_id, None)
        self._threshold_labels_by_expert.pop(stage_id, None)
        self._decision_thresholds.pop(stage_id, None)
        self._calibration_score_drift_ema_by_expert.pop(stage_id, None)

        self._threshold_scores = [
            score
            for expert_scores in self._threshold_scores_by_expert.values()
            for score in expert_scores
        ]
        self._threshold_labels = [
            label
            for expert_labels in self._threshold_labels_by_expert.values()
            for label in expert_labels
        ]
        if self._threshold_scores and self._threshold_labels:
            self._decision_threshold = self._calibrated_threshold(
                self._threshold_scores,
                self._threshold_labels,
            )
        elif self._decision_thresholds:
            active_stage = int(getattr(self, "task_id", stage_id) or 0)
            self._decision_threshold = self._decision_thresholds.get(
                active_stage,
                next(iter(self._decision_thresholds.values())),
            )
        else:
            self._decision_threshold = 0.0

        marker = getattr(self, "_rigev2_base_stats_selector", None)
        if marker is not None and int(marker["base_stage_id"]) == stage_id:
            self._rigev2_base_stats_selector = None

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
