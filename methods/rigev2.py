from collections import deque
import logging
import math

import torch
from torch.utils.data import DataLoader

from datasets import safe_collate_drop_bad
from .race_components import (
    AllocationController,
    ClassConditionalChangeDetector,
    PersistentFeatureReplay,
    ReplayBatch,
)
from .rigev1 import RIGEv1


logger = logging.getLogger()


class RIGEv2(RIGEv1):
    """Residual Incremental Gaussian Experts v2.

    V2 keeps RIGEv1's residual expert training and Gaussian routing, but stores
    only a head-weight-selected subset of the raw online features in replay.
    """

    _BASE_STATS_SELECTOR_VERSION = 1
    _BASE_STATS_SELECTOR_STATE_KEY = "rigev2_base_stats_selector"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._race_detector = None
        self._race_replay = None
        self._race_controller = None
        self._race_detection_events = []
        self._race_allocation_events = []
        self._race_candidate = deque(
            maxlen=max(0, int(self._cfg("change_backfill_size", 128) or 0))
        )
        self._race_stream_started = False
        self._race_stream_finished = False
        self._race_replay_sampled_batches = 0
        self._race_replay_balanced_batches = 0
        self._race_pending_backfill = None

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

    def _uses_continuous_online_stream(self):
        return bool(self._cfg("continuous_stream", True))

    def _uses_oracle_boundaries(self):
        return str(self._cfg("allocation_mode", "detected") or "detected").lower() == "oracle"

    def online_before_stream(self):
        self._validate_route_space()
        self._ensure_online_feature_indices()
        model = self.model_without_ddp
        model.active_stage = 0
        model.set_train_stage(0)
        model.set_backbone_trainable(False)
        for parameter in model.base_head.parameters():
            parameter.requires_grad = False
        self.task_id = 0
        self._current_head_optimizer = None
        self._reset_feature_replay()

        residual_enabled = bool(self._cfg("residual_enabled", True))
        allocation_mode = str(
            self._cfg("allocation_mode", "detected") or "detected"
        ).lower()
        if not residual_enabled:
            allocation_mode = "none"
        allocation_num_regimes = int(self._cfg("allocation_num_regimes", 0) or 0)
        allocation_seed = int(
            self._cfg("allocation_seed", getattr(self, "rnd_seed", 1)) or 0
        )
        allocation_positions = self._batch_aligned_allocation_positions(
            allocation_mode,
            allocation_num_regimes,
            int(getattr(self, "online_stream_length", 0) or 0),
            allocation_seed,
        )
        self._race_controller = AllocationController(
            allocation_mode,
            num_regimes=allocation_num_regimes,
            total_online_samples=int(getattr(self, "online_stream_length", 0) or 0),
            positions=allocation_positions,
            seed=allocation_seed,
            max_residual_experts=int(self._cfg("max_residual_experts", 32) or 32),
        )
        replay_capacity = self._feature_replay_window() if residual_enabled else 0
        self._race_replay = PersistentFeatureReplay(
            replay_capacity,
            self._expected_route_feature_dim(),
            logit_dim=int(getattr(model, "num_classes", 2) or 2),
            sampling=str(self._cfg("replay_sampling", "class_balanced")),
            replacement=str(self._cfg("replay_replacement", "fifo")),
            seed=int(getattr(self, "rnd_seed", 1) or 0),
            store_regime_ids=bool(self._cfg("replay_active_only", False)),
        )
        self._race_candidate.clear()
        self._race_detection_events = []
        self._race_allocation_events = []
        self._race_replay_sampled_batches = 0
        self._race_replay_balanced_batches = 0
        self._race_pending_backfill = None

        self._race_detector = None
        if allocation_mode == "detected":
            class_conditional = bool(self._cfg("change_class_conditional", True))
            threshold_mode = str(
                self._cfg("change_threshold_mode", "base_quantile") or "base_quantile"
            ).lower()
            fixed_threshold = None
            if threshold_mode == "fixed":
                fixed_threshold = float(self._cfg("change_fixed_threshold", 1.0))
            elif threshold_mode != "base_quantile":
                raise ValueError(
                    f"Unsupported RIGEv2 change threshold mode: {threshold_mode!r}"
                )
            self._race_detector = ClassConditionalChangeDetector(
                self._expected_route_feature_dim(),
                window_size=int(self._cfg("change_window", 512) or 512),
                ewma_beta=float(self._cfg("change_ewma_beta", 0.9) or 0.0),
                false_alarm_rate=float(
                    self._cfg("change_false_alarm_rate", 0.05) or 0.05
                ),
                persistence=int(self._cfg("change_persistence", 3) or 1),
                warmup_samples=int(self._cfg("change_warmup_samples", 512) or 0),
                cooldown_samples=int(self._cfg("change_cooldown_samples", 1024) or 0),
                min_class_count=int(self._cfg("change_min_class_count", 16) or 1),
                min_valid_classes=(
                    int(self._cfg("change_min_valid_classes", 2) or 1)
                    if class_conditional
                    else 1
                ),
                class_conditional=class_conditional,
                variance_floor=float(self._cfg("route_variance_floor", 1e-4) or 1e-4),
                threshold=fixed_threshold,
                min_calibration_scores=int(
                    self._cfg("change_calibration_min_scores", 10) or 1
                ),
                normalize_features=bool(self._cfg("route_normalize_features", True)),
            )
            base_features, base_labels = self._base_change_calibration_data()
            if threshold_mode == "base_quantile":
                self._race_detector.calibrate(base_features, base_labels)
            else:
                self._race_detector.reset_reference(
                    base_features,
                    base_labels,
                    start_cooldown=False,
                )
            logger.info(
                "RACE detector ready | mode=%s | base_samples=%s | threshold=%s",
                threshold_mode,
                int(base_labels.numel()),
                self._race_detector.threshold,
            )

        self._race_stream_started = True
        self._race_stream_finished = False
        logger.info(
            "RACE continuous stream ready | allocation=%s | total_online_samples=%s | "
            "replay_capacity=%s | replay_sampling=%s",
            allocation_mode,
            int(getattr(self, "online_stream_length", 0) or 0),
            self._feature_replay_window(),
            self._cfg("replay_sampling", "class_balanced"),
        )

    def _batch_aligned_allocation_positions(self, mode, num_regimes, total, seed):
        if mode not in {"fixed", "random"}:
            return None
        target = int(num_regimes) - 1
        if target <= 0:
            return []
        batch_size = max(1, int(getattr(self, "batchsize", 1)))
        offsets = list(range(0, int(total), batch_size))
        if target > len(offsets):
            raise ValueError(
                "Requested more fixed/random residual experts than online batches"
            )
        if mode == "fixed":
            return [offsets[index * len(offsets) // target] for index in range(target)]
        if target == 1:
            return [0]
        generator = torch.Generator().manual_seed(int(seed))
        selected = torch.randperm(len(offsets) - 1, generator=generator)[: target - 1]
        return [0] + sorted(offsets[int(index) + 1] for index in selected.tolist())

    @torch.no_grad()
    def _base_change_calibration_data(self):
        base_stage_id = self._base_stage_id()
        if base_stage_id is None:
            raise ValueError("RACE change detector requires a supervised base stage")
        indices = list(self.train_dataset.stage_indices.get(int(base_stage_id), []))
        maximum = int(self._cfg("change_calibration_max_samples", 20000) or 0)
        if maximum > 0 and len(indices) > maximum:
            stride = max(1, len(indices) // maximum)
            indices = indices[::stride][:maximum]
        if not indices:
            raise ValueError("RACE change detector found no base-stage samples")

        subset = self.train_dataset.make_eval_subset(indices)
        loader = DataLoader(
            subset,
            # Two campaign jobs share each GPU; keep calibration inference
            # comfortably below the training/evaluation memory peak.
            batch_size=max(1, min(32, int(self.batchsize) * 2)),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=safe_collate_drop_bad,
        )
        features = []
        labels = []
        model = self.model_without_ddp
        was_training = self.model.training
        model.eval()
        try:
            for batch in loader:
                if batch is None:
                    continue
                images, _targets, binary_targets = batch
                images = self.test_transform_tensor(images.to(self.device, non_blocking=True))
                _z, online_z = model.extract_base_and_online_z(images)
                features.append(online_z.detach().float().cpu())
                labels.append(torch.as_tensor(binary_targets, dtype=torch.long).cpu())
        finally:
            if was_training:
                self.model.train()
        if not features:
            raise ValueError("RACE could not read any base-stage calibration samples")
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        return self._interleave_base_calibration(features, labels)

    def _interleave_base_calibration(self, features, labels):
        """Build deterministic mixed-class windows for null calibration."""
        generator = torch.Generator().manual_seed(
            int(getattr(self, "rnd_seed", 1) or 0) + 7_919
        )
        grouped = []
        for label in (0, 1):
            indices = torch.nonzero(labels == label, as_tuple=False).reshape(-1)
            if indices.numel():
                order = torch.randperm(indices.numel(), generator=generator)
                indices = indices[order]
            grouped.append(indices)
        if any(indices.numel() == 0 for indices in grouped):
            raise ValueError(
                "RACE base calibration requires both binary classes"
            )

        mixed = []
        limit = max(int(indices.numel()) for indices in grouped)
        for offset in range(limit):
            for indices in grouped:
                if offset < indices.numel():
                    mixed.append(indices[offset])
        mixed_indices = torch.stack(mixed).long()
        return features[mixed_indices], labels[mixed_indices]

    def online_oracle_boundary(self):
        if self._race_controller is None:
            raise RuntimeError("RACE stream is not initialized")
        event = self._race_controller.should_allocate(
            int(getattr(self, "online_samples_seen", 0)),
            oracle=True,
        )
        if event is not None:
            self._activate_race_expert(event, change_event=None)

    def online_after_stream(self):
        self._race_stream_finished = True
        if self._race_detector is not None:
            diagnostics = self._race_detector.diagnostics()
            if diagnostics.get("awaiting_reference_reset"):
                raise RuntimeError("RACE detector ended while awaiting a reference reset")

    def _race_binary_labels(self, mapped_labels):
        values = []
        for label in mapped_labels.detach().cpu().tolist():
            original = self.exposed_classes[int(label)]
            values.append(0 if int(original) == 0 else 1)
        return torch.tensor(values, dtype=torch.long)

    def _make_race_head_optimizer(self, head):
        model = self.model_without_ddp
        direct = bool(getattr(model, "direct_classifier", False))
        fixed_alpha = bool(self._cfg("fixed_alpha", False))
        unfreeze_base = bool(self._cfg("unfreeze_base_head", False))
        parameters = list(head.parameters())
        alpha = model.current_alpha()
        if alpha is not None:
            alpha.requires_grad = not fixed_alpha and not direct
            if alpha.requires_grad:
                parameters.append(alpha)
        for parameter in model.base_head.parameters():
            parameter.requires_grad = unfreeze_base
        if unfreeze_base:
            parameters.extend(model.base_head.parameters())
        return torch.optim.AdamW(
            parameters,
            lr=float(getattr(self, "lr", 1e-3)),
            weight_decay=float(self._cfg("weight_decay", 1e-4) or 0.0),
        )

    def _route_statistics_enabled(self):
        return str(getattr(self.model_without_ddp, "eval_mode", "")) == "feature_gaussian"

    def _activate_race_expert(self, allocation_event, change_event):
        expert_id = int(allocation_event.residual_expert_id)
        model = self.model_without_ddp
        if expert_id != len(model.residual_heads) + 1:
            raise RuntimeError("RACE allocation/controller expert counts diverged")
        head = model.add_residual_head(expert_id).to(self.device)
        self.task_id = expert_id
        model.active_stage = expert_id
        self._current_head_optimizer = self._make_race_head_optimizer(head)

        if bool(self._cfg("replay_reset_on_change", False)):
            self._race_replay.clear()
        if (
            bool(self._cfg("change_candidate_backfill", True))
            and self._race_candidate
        ):
            candidate_features = torch.stack(
                [record[0] for record in self._race_candidate], dim=0
            )
            candidate_base_logits = torch.stack(
                [record[1] for record in self._race_candidate], dim=0
            )
            candidate_labels = torch.tensor(
                [record[2] for record in self._race_candidate], dtype=torch.long
            )
            self._race_pending_backfill = ReplayBatch(
                candidate_features,
                candidate_base_logits,
                candidate_labels,
            )
            if self._route_statistics_enabled():
                self._update_route_stats(expert_id, candidate_features)

        if change_event is not None:
            self._race_detection_events.append(
                {
                    "sample_offset": int(change_event.sample_offset),
                    "raw_score": float(change_event.raw_score),
                    "ewma_score": float(change_event.ewma_score),
                    "threshold": float(change_event.threshold),
                    "persistence": int(change_event.persistence),
                    "valid_classes": list(change_event.valid_classes),
                }
            )
            self._race_detector.reset_reference(
                change_event.candidate_features,
                change_event.candidate_labels,
                start_cooldown=True,
            )
        self._race_allocation_events.append(
            {
                "sample_offset": int(allocation_event.sample_offset),
                "residual_expert_id": expert_id,
                "total_regimes": int(allocation_event.total_regimes),
                "mode": str(allocation_event.mode),
                "reason": str(allocation_event.reason),
                "scheduled_offset": allocation_event.scheduled_offset,
            }
        )
        logger.info(
            "RACE allocated residual expert %s at online sample %s | reason=%s",
            expert_id,
            allocation_event.sample_offset,
            allocation_event.reason,
        )

    def online_step(self, images, labels, idx):
        if not self._race_stream_started:
            return super().online_step(images, labels, idx)
        del idx
        self.add_new_class(labels)
        mapped = labels.clone()
        for index in range(len(mapped)):
            mapped[index] = self.exposed_classes.index(mapped[index].item())
        y = mapped.to(self.device)
        binary_y = self._race_binary_labels(y)
        raw_x = images.to(self.device, non_blocking=True)

        model = self.model_without_ddp
        model.backbone.eval()
        with torch.no_grad():
            deterministic_x = self.test_transform_tensor(raw_x)
            route_z, route_online_z = model.extract_base_and_online_z(deterministic_x)
            route_z = route_z.detach()
            route_online_z = route_online_z.detach()
            route_base_logits = model.base_head(route_z).detach()
            if self._race_controller.mode == "none":
                z, online_z = route_z, route_online_z
            else:
                z, online_z = model.extract_base_and_online_z(self.train_transform(raw_x))
                z = z.detach()
                online_z = online_z.detach()

        start_offset = int(getattr(self, "online_samples_seen", 0))
        end_offset = start_offset + int(images.size(0))
        change_event = None
        if self._race_detector is not None:
            change_event = self._race_detector.observe(
                route_online_z,
                binary_y,
                sample_offset=end_offset,
            )

        if change_event is not None:
            allocation = self._race_controller.should_allocate(
                end_offset,
                detected=True,
            )
            if allocation is None:
                raise RuntimeError("RACE detector triggered without allocating an expert")
            self._activate_race_expert(allocation, change_event)
        elif self._race_controller.mode in {"single", "fixed", "random"}:
            while True:
                allocation = self._race_controller.should_allocate(start_offset)
                if allocation is None:
                    break
                self._activate_race_expert(allocation, change_event=None)

        with torch.no_grad():
            pre_update_logits = model.combined_logits_from_z(
                z,
                online_z=online_z,
                expert_id=int(self.task_id),
            )

        if self.task_id <= 0 or self._current_head_optimizer is None:
            current_logits = pre_update_logits
            loss = self.criterion(current_logits, y)
            _, predictions = current_logits.topk(self.topk, 1, True, True)
            accuracy = torch.sum(predictions == y.unsqueeze(1)).item() / y.size(0)
            loss_value = float(loss.detach().cpu())
            accuracy_value = float(accuracy)
        else:
            optimizer = self._current_head_optimizer
            model.current_head().train()
            if bool(self._cfg("unfreeze_base_head", False)):
                model.base_head.train()
            else:
                model.base_head.eval()
            inner_steps = max(1, int(self._cfg("inner_steps", 1) or 1))
            replay_weight = float(self._cfg("replay_loss_weight", 1.0) or 0.0)
            loss_value = 0.0
            accuracy_value = 0.0
            for _ in range(inner_steps):
                optimizer.zero_grad(set_to_none=True)
                current_logits = model.combined_logits_from_z(
                    z,
                    online_z=online_z,
                    expert_id=int(self.task_id),
                )
                current_loss = self.criterion(current_logits, y)
                loss = current_loss
                replay = self._sample_race_replay()
                if replay is not None and replay_weight > 0.0:
                    replay_logits = self._race_replay_logits(
                        replay.features,
                        replay.base_logits,
                    )
                    replay_loss = self.criterion(replay_logits, replay.labels)
                    loss = current_loss + replay_weight * replay_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "RACE produced a non-finite online loss at "
                        f"internal_regime={self.task_id}"
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                with torch.no_grad():
                    current_logits = model.combined_logits_from_z(
                        z,
                        online_z=online_z,
                        expert_id=int(self.task_id),
                    )
                    if not torch.isfinite(current_logits).all():
                        raise FloatingPointError("RACE produced non-finite online logits")
                    _, predictions = current_logits.topk(self.topk, 1, True, True)
                    accuracy = (
                        torch.sum(predictions == y.unsqueeze(1)).item() / y.size(0)
                    )
                loss_value = float(loss.detach().cpu())
                accuracy_value = float(accuracy)

        with torch.no_grad():
            post_update_logits = model.combined_logits_from_z(
                z,
                online_z=online_z,
                expert_id=int(self.task_id),
            ).detach()
            score_drift = self._score_drift_from_logits(
                pre_update_logits,
                post_update_logits,
            )
            route_logits = model.combined_logits_from_z(
                route_z,
                online_z=route_online_z,
                expert_id=int(self.task_id),
            ).detach()

        self._race_replay.add(
            route_online_z,
            route_base_logits,
            y.detach().cpu(),
            regime_id=int(self.task_id),
        )
        if self.task_id > 0:
            self._update_threshold_buffer(
                int(self.task_id),
                route_logits,
                y.detach(),
                score_drift=score_drift,
            )
            if self._route_statistics_enabled():
                self._update_route_stats(int(self.task_id), route_online_z)
        self._append_race_candidates(
            route_online_z,
            route_base_logits,
            y.detach().cpu(),
        )
        self._race_pending_backfill = None
        return loss_value, accuracy_value

    def _sample_race_replay(self):
        if self._race_pending_backfill is not None:
            pending = self._race_pending_backfill
            count = min(self._feature_replay_batch_size(), len(pending))
            if count > 0:
                selected = slice(len(pending) - count, len(pending))
                replay = ReplayBatch(
                    pending.features[selected].to(self.device, non_blocking=True),
                    pending.base_logits[selected].to(self.device, non_blocking=True),
                    pending.labels[selected].to(self.device, non_blocking=True),
                )
                self._race_replay_sampled_batches += 1
                zero_count = int((replay.labels == 0).sum().item())
                one_count = int((replay.labels == 1).sum().item())
                if abs(zero_count - one_count) <= 1 and zero_count and one_count:
                    self._race_replay_balanced_batches += 1
                return replay
        regime_id = None
        if bool(self._cfg("replay_active_only", False)):
            regime_id = int(self.task_id)
        replay = self._race_replay.sample(
            self._feature_replay_batch_size(),
            device=self.device,
            regime_id=regime_id,
        )
        if replay is not None:
            self._race_replay_sampled_batches += 1
            zero_count = int((replay.labels == 0).sum().item())
            one_count = int((replay.labels == 1).sum().item())
            if abs(zero_count - one_count) <= 1 and zero_count and one_count:
                self._race_replay_balanced_batches += 1
        return replay

    def _race_replay_logits(self, features, base_logits):
        model = self.model_without_ddp
        if bool(getattr(model, "direct_classifier", False)):
            return model.current_head()(features)
        alpha = model.current_alpha().to(dtype=base_logits.dtype)
        return base_logits + alpha * model.current_head()(features)

    def _append_race_candidates(self, features, base_logits, labels):
        if (
            self._race_candidate.maxlen == 0
            or not bool(self._cfg("change_candidate_backfill", True))
            or self._race_controller is None
            or self._race_controller.mode == "none"
        ):
            return
        features = features.detach().float().cpu()
        base_logits = base_logits.detach().float().cpu()
        labels = labels.detach().long().cpu()
        for row in range(int(labels.numel())):
            self._race_candidate.append(
                (
                    features[row].clone(),
                    base_logits[row].clone(),
                    int(labels[row].item()),
                )
            )

    def _method_summary_payload(self):
        model = getattr(self, "model_without_ddp", None)
        residual_count = 0 if model is None else len(model.residual_heads)
        controller = (
            {} if self._race_controller is None else self._race_controller.diagnostics()
        )
        replay = {} if self._race_replay is None else self._race_replay.diagnostics()
        detector = (
            {} if self._race_detector is None else self._race_detector.diagnostics()
        )
        expert_parameters = []
        if model is not None:
            expert_parameters = [
                int(sum(parameter.numel() for parameter in head.parameters()))
                for head in model.residual_heads
            ]
        sampled = int(self._race_replay_sampled_batches)
        return {
            "method": "RACE",
            "code_method": "RIGEv2",
            "continuous_learner_stream": bool(self._uses_continuous_online_stream()),
            "uses_boundary_signal": bool(self._uses_oracle_boundaries()),
            "stream_started": bool(self._race_stream_started),
            "stream_finished": bool(self._race_stream_finished),
            "residual_expert_count": int(residual_count),
            "final_expert_count": int(residual_count + 1),
            "allocation": controller,
            "allocation_events": list(self._race_allocation_events),
            "detector": detector,
            "detection_events": list(self._race_detection_events),
            "replay": replay,
            "replay_balanced_batch_fraction": (
                None
                if sampled == 0
                else float(self._race_replay_balanced_batches / sampled)
            ),
            "selected_feature_dim": int(self._expected_route_feature_dim()),
            "selected_feature_indices": (
                []
                if model is None
                else model.online_feature_indices.detach().cpu().tolist()
            ),
            "residual_expert_parameters": expert_parameters,
        }

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
        selector = str(self._cfg("feature_selector", "headweight") or "headweight").lower()
        block_dim = int(self._cfg("feature_block_dim", 768) or 0)
        reuse_checkpoint_selector = (
            selector in {"headweight", "block_topk"} and block_dim == 768
        )
        if torch.is_tensor(selected_indices) and reuse_checkpoint_selector:
            try:
                self.model_without_ddp.set_online_feature_indices(selected_indices.cpu())
                logger.info(
                    "Loaded RIGEv2 online feature indices from checkpoint | dim=%s",
                    int(selected_indices.numel()),
                )
            except (AttributeError, ValueError) as exc:
                logger.warning("Ignored incompatible RIGEv2 feature indices: %s", exc)
        elif torch.is_tensor(selected_indices):
            logger.info(
                "Ignoring checkpoint feature indices for selector ablation | "
                "selector=%s block_dim=%s",
                selector,
                block_dim,
            )

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

        selector = str(self._cfg("feature_selector", "headweight") or "headweight").lower()
        if selector == "identity":
            if selected_dim != feature_dim:
                raise ValueError(
                    "RIGEv2 identity feature selector requires replay_dim == raw feature dim"
                )
            return torch.arange(feature_dim, dtype=torch.long)
        if selector == "random":
            generator = torch.Generator().manual_seed(
                int(self._cfg("feature_selector_seed", getattr(self, "rnd_seed", 1)) or 0)
            )
            return torch.sort(
                torch.randperm(feature_dim, generator=generator)[:selected_dim]
            ).values.long()
        if selector not in {"headweight", "global_topk", "block_topk"}:
            raise ValueError(f"Unsupported RIGEv2 feature selector: {selector!r}")

        block_dim = int(self._cfg("feature_block_dim", 768) or 0)
        if selector == "global_topk":
            block_dim = 0
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

    def _feature_gaussian_scores(self, online_z, expert_count):
        metric = str(self._cfg("route_metric", "gaussian") or "gaussian").lower()
        if metric == "gaussian":
            return super()._feature_gaussian_scores(online_z, expert_count)
        if metric != "nearest_mean":
            raise ValueError(f"Unsupported RIGEv2 route metric: {metric!r}")

        features = self._route_features(online_z)
        scores = torch.full(
            (features.size(0), int(expert_count)),
            -torch.inf,
            dtype=features.dtype,
            device=features.device,
        )
        min_count = max(1, int(self._cfg("route_min_count", 2) or 1))
        for expert_id in range(int(expert_count)):
            stats = self._route_stats_by_expert.get(expert_id)
            if not stats or int(stats.get("count", 0)) < min_count:
                continue
            mean = stats["mean"].to(features.device, dtype=features.dtype)
            scores[:, expert_id] = -(features - mean.view(1, -1)).pow(2).mean(dim=1)
        missing = torch.isneginf(scores).all(dim=1)
        if torch.any(missing):
            fallback = min(max(int(self.task_id), 0), int(expert_count) - 1)
            scores[missing, fallback] = 0.0
        return scores

    def _calibrated_threshold(self, scores, labels):
        mode = str(self._cfg("threshold_mode", "gaussian_midpoint") or "gaussian_midpoint").lower()
        if mode in {"fixed", "fixed_margin", "fixed_margin0"}:
            return float(self._cfg("classification_fixed_threshold", 0.0) or 0.0)
        return super()._calibrated_threshold(scores, labels)

    def _threshold_for_expert(self, expert_id):
        scope = str(self._cfg("threshold_scope", "per_expert") or "per_expert").lower()
        if scope == "global":
            return float(getattr(self, "_decision_threshold", 0.0))
        if scope != "per_expert":
            raise ValueError(f"Unsupported RIGEv2 threshold scope: {scope!r}")
        return super()._threshold_for_expert(expert_id)
