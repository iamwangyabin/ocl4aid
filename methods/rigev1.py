from collections import Counter
import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import safe_collate_drop_bad
from methods._trainer import _Trainer
from protocol_metrics import compute_binary_detection_metrics


logger = logging.getLogger()


class RIGEv1(_Trainer):
    """Residual Incremental Gaussian Experts v1."""

    def __init__(self, *args, **kwargs):
        super(RIGEv1, self).__init__(*args, **kwargs)
        self.task_id = 0
        self._current_head_optimizer = None
        self._threshold_scores = []
        self._threshold_labels = []
        self._threshold_scores_by_expert = {}
        self._threshold_labels_by_expert = {}
        self._decision_thresholds = {}
        self._decision_threshold = 0.0
        self._gate_scores_by_expert = {}
        self._gate_labels_by_expert = {}
        self._gate_thresholds = {}
        self._calibration_score_drift_ema_by_expert = {}
        self._route_stats_by_expert = {}
        self._score_dump_counter = 0
        self._replay_online_z = None
        self._replay_base_logits = None
        self._replay_labels = None

    def _cfg(self, name, default):
        return getattr(self, f"rigev1_{name}", default)

    def online_before_task(self, task_id):
        self.task_id = int(task_id)
        model = self.model_without_ddp
        model.active_stage = self.task_id
        if hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(False)

        if self.task_id == 0:
            model.set_train_stage(0)
            self._current_head_optimizer = self.optimizer
            return

        self._reset_feature_replay()
        head = model.add_residual_head(self.task_id).to(self.device)
        if self.distributed:
            logger.warning(
                "RIGEv1 dynamically adds heads and is intended for single-GPU runs; "
                "DDP parameter synchronization is not implemented for new heads."
            )
        optimizer_params = list(head.parameters())
        alpha = model.current_alpha() if hasattr(model, "current_alpha") else None
        if alpha is not None:
            optimizer_params.append(alpha)
        self._current_head_optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=float(getattr(self, "lr", 1e-3)),
            weight_decay=float(self._cfg("weight_decay", 1e-4) or 0.0),
        )
        logger.info(
            "RIGEv1 stage %s | base_head=%s | online_head=%s | online_dim=%s | alpha=%s | inner_steps=%s | replay_window=%s | replay_batch=%s | threshold_mode=%s | update_all_calibration=%s | route_eval_transform=%s | route_normalize=%s | route_var_floor=%s | drift_threshold=%s | threshold_window=%s",
            self.task_id,
            getattr(model, "head_type", None),
            getattr(model, "online_head_type", None),
            getattr(model, "online_feature_dim", None),
            None if alpha is None else float(alpha.detach().cpu()),
            self._cfg("inner_steps", 1),
            self._cfg("replay_window", 0),
            self._cfg("replay_batch_size", 0),
            self._cfg("threshold_mode", "online_f1"),
            self._cfg("update_all_expert_calibration", False),
            self._cfg("route_use_eval_transform", True),
            self._cfg("route_normalize_features", True),
            self._cfg("route_variance_floor", 1e-4),
            self._cfg("calibration_score_drift_threshold", 0.2),
            self._cfg("calibration_score_window", 5000),
        )

    def online_step(self, images, labels, idx):
        del idx
        self.add_new_class(labels)
        y = labels.clone()
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        raw_x = images.to(self.device, non_blocking=True)
        y = y.to(self.device)

        model = self.model_without_ddp
        model.backbone.eval()
        model.current_head().train()
        with torch.no_grad():
            z, online_z = model.extract_base_and_online_z(self.train_transform(raw_x))
            z = z.detach()
            online_z = online_z.detach()
            route_online_z = online_z
            if bool(self._cfg("route_use_eval_transform", True)):
                _route_z, route_online_z = model.extract_base_and_online_z(
                    self.test_transform_tensor(raw_x)
                )
                route_online_z = route_online_z.detach()

        optimizer = self._current_head_optimizer or self.optimizer
        inner_steps = max(1, int(self._cfg("inner_steps", 1) or 1))
        loss_value = 0.0
        acc_value = 0.0
        with torch.no_grad():
            pre_update_logits = self._active_stage_logits_from_z(model, z, online_z)
            current_base_logits = model.base_head(z).detach() if self.task_id > 0 else None

        for _ in range(inner_steps):
            optimizer.zero_grad(set_to_none=True)
            # Selected residual features can exceed FP16's stable linear range.
            # Keep online expert optimization in FP32; the supervised base head
            # may still use AMP through the framework setting.
            with torch.cuda.amp.autocast(enabled=self.use_amp and self.task_id == 0):
                if self.task_id == 0:
                    logits = model.base_head(z)
                    train_y = y
                else:
                    train_online_z, train_base_logits, train_y = self._build_replay_training_batch(
                        online_z,
                        current_base_logits,
                        y,
                    )
                    alpha = model.current_alpha().to(dtype=train_base_logits.dtype)
                    logits = train_base_logits + alpha * model.current_head()(train_online_z)
                loss = self.criterion(logits, train_y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "RIGE produced a non-finite training loss "
                        f"at stage={self.task_id}; refusing to emit invalid metrics."
                    )

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            with torch.no_grad():
                current_logits = self._active_stage_logits_from_z(model, z, online_z)
                if not torch.isfinite(current_logits).all():
                    raise FloatingPointError(
                        "RIGE produced non-finite logits after an optimizer step "
                        f"at stage={self.task_id}."
                    )
                _, preds = current_logits.topk(self.topk, 1, True, True)
                acc = torch.sum(preds == y.unsqueeze(1)).item() / y.size(0)

            loss_value = float(loss.detach().cpu())
            acc_value = float(acc)

        score_drift = self._score_drift_from_logits(pre_update_logits, current_logits.detach())
        if self.task_id == 0:
            self.update_schedule()
        else:
            self._append_feature_replay(current_base_logits, online_z, y)
        self._update_expert_calibration(
            route_online_z,
            y,
            current_logits=current_logits.detach(),
            score_drift=score_drift,
        )
        return loss_value, acc_value

    def _update_expert_calibration(
        self,
        route_online_z,
        labels,
        *,
        current_logits,
        score_drift,
    ):
        update_all = bool(self._cfg("update_all_expert_calibration", False))
        expert_ids = (
            range(len(self.model_without_ddp.residual_heads) + 1)
            if update_all
            else [self.task_id]
        )
        stage_id = int(self.task_id)
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id == stage_id:
                logits = current_logits
                expert_score_drift = score_drift
            else:
                # Keep old classifier calibration frozen. Older score thresholds
                # represent their own stage, not future negative examples.
                continue
            self._update_threshold_buffer(
                expert_id,
                logits,
                labels.detach(),
                score_drift=expert_score_drift,
            )
        self._update_route_stats(stage_id, route_online_z.detach())

    def _reset_feature_replay(self):
        self._replay_online_z = None
        self._replay_base_logits = None
        self._replay_labels = None

    def _feature_replay_window(self):
        return max(0, int(self._cfg("replay_window", 0) or 0))

    def _feature_replay_batch_size(self):
        return max(0, int(self._cfg("replay_batch_size", 0) or 0))

    def _build_replay_training_batch(self, online_z, base_logits, labels):
        replay = self._sample_feature_replay(online_z.device)
        if replay is None:
            return online_z, base_logits, labels

        replay_online_z, replay_base_logits, replay_labels = replay
        train_online_z = torch.cat([online_z, replay_online_z.to(dtype=online_z.dtype)], dim=0)
        train_base_logits = torch.cat(
            [base_logits, replay_base_logits.to(dtype=base_logits.dtype)],
            dim=0,
        )
        train_labels = torch.cat([labels, replay_labels], dim=0)
        return train_online_z, train_base_logits, train_labels

    def _sample_feature_replay(self, device):
        replay_batch_size = self._feature_replay_batch_size()
        if replay_batch_size <= 0 or self._replay_online_z is None:
            return None
        replay_count = int(self._replay_labels.size(0))
        if replay_count <= 0:
            return None
        sample_count = min(replay_batch_size, replay_count)
        replay_ids = torch.randint(replay_count, (sample_count,))
        return (
            self._replay_online_z[replay_ids].to(device, non_blocking=True),
            self._replay_base_logits[replay_ids].to(device, non_blocking=True),
            self._replay_labels[replay_ids].to(device, non_blocking=True),
        )

    def _append_feature_replay(self, base_logits, online_z, labels):
        window = self._feature_replay_window()
        if window <= 0:
            return
        new_online_z = online_z.detach().float().cpu()
        new_base_logits = base_logits.detach().float().cpu()
        new_labels = labels.detach().long().cpu()
        if self._replay_online_z is None:
            self._replay_online_z = new_online_z
            self._replay_base_logits = new_base_logits
            self._replay_labels = new_labels
        else:
            self._replay_online_z = torch.cat([self._replay_online_z, new_online_z], dim=0)
            self._replay_base_logits = torch.cat(
                [self._replay_base_logits, new_base_logits],
                dim=0,
            )
            self._replay_labels = torch.cat([self._replay_labels, new_labels], dim=0)

        if self._replay_labels.size(0) > window:
            self._replay_online_z = self._replay_online_z[-window:]
            self._replay_base_logits = self._replay_base_logits[-window:]
            self._replay_labels = self._replay_labels[-window:]

    def _route_features(self, features):
        features = features.detach().float()
        if bool(self._cfg("route_normalize_features", True)):
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return features

    def _update_route_stats(self, expert_id, features):
        features = self._route_features(features).cpu()
        if features.numel() == 0:
            return
        expert_id = int(expert_id)
        batch_count = int(features.size(0))
        batch_mean = features.mean(dim=0)
        centered = features - batch_mean
        batch_m2 = centered.pow(2).sum(dim=0)

        current = self._route_stats_by_expert.get(expert_id)
        if current is None:
            self._route_stats_by_expert[expert_id] = {
                "count": batch_count,
                "mean": batch_mean,
                "m2": batch_m2,
            }
            return

        count = int(current["count"])
        mean = current["mean"].float()
        m2 = current["m2"].float()
        total = count + batch_count
        delta = batch_mean - mean
        new_mean = mean + delta * (batch_count / total)
        new_m2 = m2 + batch_m2 + delta.pow(2) * (count * batch_count / total)
        self._route_stats_by_expert[expert_id] = {
            "count": total,
            "mean": new_mean,
            "m2": new_m2,
        }

    def _feature_gaussian_scores(self, online_z, expert_count):
        features = self._route_features(online_z)
        scores = torch.full(
            (features.size(0), int(expert_count)),
            -torch.inf,
            dtype=features.dtype,
            device=features.device,
        )
        variance_floor = float(
            self._cfg("route_variance_floor", 1e-4) or 1e-4
        )
        min_count = max(1, int(self._cfg("route_min_count", 2) or 1))
        for expert_id in range(int(expert_count)):
            stats = self._route_stats_by_expert.get(expert_id)
            if not stats or int(stats.get("count", 0)) < min_count:
                continue
            count = int(stats["count"])
            mean = stats["mean"].to(features.device, dtype=features.dtype)
            m2 = stats["m2"].to(features.device, dtype=features.dtype)
            denom = max(count - 1, 1)
            variance = (m2 / denom).clamp_min(variance_floor)
            diff = features - mean.view(1, -1)
            scores[:, expert_id] = -0.5 * (
                diff.pow(2) / variance.view(1, -1) + torch.log(variance).view(1, -1)
            ).mean(dim=1)

        missing_rows = torch.isneginf(scores).all(dim=1)
        if torch.any(missing_rows):
            fallback_expert = min(
                max(int(getattr(self.model_without_ddp, "active_stage", self.task_id)), 0),
                int(expert_count) - 1,
            )
            scores[missing_rows, fallback_expert] = 0.0
        return scores

    def _checkpoint_method_state(self):
        return {
            "decision_thresholds": {
                int(expert_id): float(threshold)
                for expert_id, threshold in self._decision_thresholds.items()
            },
            "route_stats_by_expert": {
                int(expert_id): {
                    "count": int(stats["count"]),
                    "mean": stats["mean"].detach().cpu(),
                    "m2": stats["m2"].detach().cpu(),
                }
                for expert_id, stats in self._route_stats_by_expert.items()
            },
        }

    def _load_checkpoint_method_state(self, state):
        self._decision_thresholds = {
            int(expert_id): float(threshold)
            for expert_id, threshold in state.get("decision_thresholds", {}).items()
        }
        if self._decision_thresholds:
            self._decision_threshold = self._decision_thresholds.get(
                int(getattr(self, "task_id", 0) or 0),
                next(iter(self._decision_thresholds.values())),
            )
        self._route_stats_by_expert = {}
        for expert_id, stats in state.get("route_stats_by_expert", {}).items():
            mean = stats.get("mean")
            m2 = stats.get("m2")
            if not torch.is_tensor(mean) or not torch.is_tensor(m2):
                continue
            self._route_stats_by_expert[int(expert_id)] = {
                "count": int(stats.get("count", 0)),
                "mean": mean.detach().cpu().float(),
                "m2": m2.detach().cpu().float(),
            }

    def _after_base_checkpoint_loaded(self, checkpoint):
        del checkpoint
        base_stage_id = self._base_stage_id()
        if base_stage_id is None:
            return
        if base_stage_id in self._route_stats_by_expert and base_stage_id in self._decision_thresholds:
            return
        self._rebuild_stage_statistics_from_train_data(base_stage_id)

    @torch.no_grad()
    def _rebuild_stage_statistics_from_train_data(self, stage_id):
        indices = list(self.train_dataset.stage_indices.get(int(stage_id), []))
        if not indices:
            return
        max_samples = int(self._cfg("route_rebuild_max_samples", 20000) or 0)
        if max_samples > 0 and len(indices) > max_samples:
            stride = max(1, len(indices) // max_samples)
            indices = indices[::stride][:max_samples]

        model = self.model_without_ddp
        was_training = self.model.training
        model.eval()
        subset = self.train_dataset.make_eval_subset(indices)
        loader = DataLoader(
            subset,
            batch_size=max(1, min(64, self.batchsize * 4)),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=safe_collate_drop_bad,
        )
        try:
            for batch in loader:
                if batch is None:
                    continue
                images, _targets, binary_targets = batch
                images = self.test_transform_tensor(images.to(self.device, non_blocking=True))
                z, online_z = model.extract_base_and_online_z(images)
                logits = model.combined_logits_from_z(
                    z,
                    online_z=online_z,
                    expert_id=int(stage_id),
                ).detach()
                labels = torch.as_tensor(binary_targets, dtype=torch.long, device=self.device)
                self._update_route_stats(stage_id, online_z.detach())
                self._update_threshold_buffer(stage_id, logits, labels, score_drift=None)
        finally:
            if was_training:
                self.model.train()

        stats = self._route_stats_by_expert.get(int(stage_id))
        if stats:
            logger.info(
                "Rebuilt RIGEv1 route stats for stage %s from %s samples",
                stage_id,
                stats["count"],
            )

    def _active_stage_logits_from_z(self, model, z, online_z):
        if self.task_id == 0:
            return model.base_head(z)
        base_logits = model.base_head(z)
        alpha = model.current_alpha().to(dtype=base_logits.dtype)
        return base_logits + alpha * model.current_head()(online_z)

    def _fake_scores_from_logits(self, logits):
        probabilities = torch.softmax(logits.float(), dim=-1)
        fake_class_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        for logit_index, original_class in enumerate(self.exposed_classes[: logits.size(-1)]):
            if original_class != 0:
                fake_class_mask[logit_index] = True
        if not torch.any(fake_class_mask):
            return torch.zeros(logits.size(0), dtype=probabilities.dtype, device=logits.device)
        return probabilities[:, fake_class_mask].sum(dim=-1)

    def _decision_scores_from_logits(self, logits):
        logits = logits.float()
        real_index = None
        fake_indices = []
        for logit_index, original_class in enumerate(self.exposed_classes[: logits.size(-1)]):
            if original_class == 0:
                real_index = logit_index
            else:
                fake_indices.append(logit_index)
        if real_index is None or not fake_indices:
            if logits.size(-1) >= 2:
                return logits[:, 1] - logits[:, 0]
            return torch.zeros(logits.size(0), dtype=logits.dtype, device=logits.device)
        real_logits = logits[:, real_index]
        fake_logits = logits[:, fake_indices]
        if fake_logits.dim() == 2 and fake_logits.size(1) > 1:
            fake_logits = torch.logsumexp(fake_logits, dim=1)
        else:
            fake_logits = fake_logits.reshape(logits.size(0))
        return fake_logits - real_logits

    def _score_drift_from_logits(self, before_logits, after_logits):
        before_scores = self._decision_scores_from_logits(before_logits).detach().float()
        after_scores = self._decision_scores_from_logits(after_logits).detach().float()
        if before_scores.numel() == 0:
            return 0.0
        scale = torch.std(after_scores, unbiased=False).clamp_min(1e-6)
        return float(torch.mean(torch.abs(after_scores - before_scores)) / scale)

    def _update_threshold_buffer(self, expert_id, logits, labels, score_drift=None):
        scores = self._decision_scores_from_logits(logits).detach().cpu().tolist()
        binary_labels = [
            0 if self.exposed_classes[int(label)] == 0 else 1
            for label in labels.detach().cpu().tolist()
        ]
        expert_id = int(expert_id)
        if not self._should_update_calibration_window(expert_id, score_drift):
            return

        self._threshold_scores_by_expert.setdefault(expert_id, []).extend(
            float(score) for score in scores
        )
        self._threshold_labels_by_expert.setdefault(expert_id, []).extend(
            int(label) for label in binary_labels
        )

        window = int(self._cfg("calibration_score_window", 5000) or 0)
        if window > 0 and len(self._threshold_scores_by_expert[expert_id]) > window:
            self._threshold_scores_by_expert[expert_id] = self._threshold_scores_by_expert[expert_id][-window:]
            self._threshold_labels_by_expert[expert_id] = self._threshold_labels_by_expert[expert_id][-window:]

        self._decision_thresholds[expert_id] = self._calibrated_threshold(
            self._threshold_scores_by_expert[expert_id],
            self._threshold_labels_by_expert[expert_id],
        )

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
        self._decision_threshold = self._calibrated_threshold(
            self._threshold_scores,
            self._threshold_labels,
        )

    def _update_gate_buffer(self, expert_id, logits, *, is_owner):
        scores = self._gate_scores_from_logits(logits, expert_id).detach().cpu().tolist()
        labels = [1 if is_owner else 0 for _ in scores]
        expert_id = int(expert_id)
        self._gate_scores_by_expert.setdefault(expert_id, []).extend(
            float(score) for score in scores
        )
        self._gate_labels_by_expert.setdefault(expert_id, []).extend(labels)

        window = int(self._cfg("calibration_score_window", 5000) or 0)
        if window > 0 and len(self._gate_scores_by_expert[expert_id]) > window:
            self._gate_scores_by_expert[expert_id] = self._gate_scores_by_expert[expert_id][-window:]
            self._gate_labels_by_expert[expert_id] = self._gate_labels_by_expert[expert_id][-window:]

        self._gate_thresholds[expert_id] = self._calibrated_threshold(
            self._gate_scores_by_expert[expert_id],
            self._gate_labels_by_expert[expert_id],
        )

    def _gate_scores_from_logits(self, logits, expert_id):
        decision_scores = self._decision_scores_from_logits(logits)
        threshold = self._threshold_for_expert(int(expert_id))
        return torch.abs(decision_scores - threshold)

    def _should_update_calibration_window(self, expert_id, score_drift):
        if score_drift is None:
            return True
        beta = float(self._cfg("calibration_drift_ema_beta", 0.9) or 0.0)
        beta = min(max(beta, 0.0), 0.999)
        score_drift = float(score_drift)
        previous = self._calibration_score_drift_ema_by_expert.get(expert_id)
        drift_ema = score_drift if previous is None else beta * previous + (1.0 - beta) * score_drift
        self._calibration_score_drift_ema_by_expert[expert_id] = drift_ema
        threshold = float(
            self._cfg("calibration_score_drift_threshold", 0.2) or 0.0
        )
        return drift_ema <= threshold

    def _calibrated_threshold(self, scores, labels):
        threshold_mode = str(
            self._cfg("threshold_mode", "online_f1") or "online_f1"
        ).lower()
        if threshold_mode in {"gaussian_midpoint", "gaussian_mean_midpoint"}:
            return self._gaussian_midpoint_threshold(scores, labels)
        if threshold_mode == "gaussian_intersection":
            return self._gaussian_intersection_threshold(scores, labels)
        return self._best_f1_threshold(scores, labels)

    @staticmethod
    def _best_f1_threshold(scores, labels):
        if not scores or len(set(labels)) < 2:
            return 0.0
        pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
        positives = sum(labels)
        tp = 0
        fp = 0
        best_f1 = -1.0
        best_threshold = 0.0
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            fn = positives - tp
            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else (2 * tp) / denom
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(score)
        return best_threshold

    @staticmethod
    def _gaussian_intersection_threshold(scores, labels):
        real_scores = [float(score) for score, label in zip(scores, labels) if int(label) == 0]
        fake_scores = [float(score) for score, label in zip(scores, labels) if int(label) == 1]
        if not real_scores or not fake_scores:
            return 0.0

        def moments(values):
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            return mean, max(variance, 1e-6)

        mu_real, var_real = moments(real_scores)
        mu_fake, var_fake = moments(fake_scores)
        midpoint = 0.5 * (mu_real + mu_fake)
        if abs(var_real - var_fake) < 1e-8:
            return float(midpoint)

        a = (1.0 / var_real) - (1.0 / var_fake)
        b = (-2.0 * mu_real / var_real) + (2.0 * mu_fake / var_fake)
        c = (mu_real * mu_real / var_real) - (mu_fake * mu_fake / var_fake) + math.log(var_real / var_fake)
        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return float(midpoint)
            return float(-c / b)

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return float(midpoint)
        root_delta = math.sqrt(discriminant)
        roots = [(-b - root_delta) / (2.0 * a), (-b + root_delta) / (2.0 * a)]
        lower = min(mu_real, mu_fake)
        upper = max(mu_real, mu_fake)
        in_between = [root for root in roots if lower <= root <= upper]
        if in_between:
            return float(min(in_between, key=lambda root: abs(root - midpoint)))
        return float(min(roots, key=lambda root: abs(root - midpoint)))

    @staticmethod
    def _gaussian_midpoint_threshold(scores, labels):
        real_scores = [float(score) for score, label in zip(scores, labels) if int(label) == 0]
        fake_scores = [float(score) for score, label in zip(scores, labels) if int(label) == 1]
        if not real_scores or not fake_scores:
            return 0.0
        mu_real = sum(real_scores) / len(real_scores)
        mu_fake = sum(fake_scores) / len(fake_scores)
        return float(0.5 * (mu_real + mu_fake))

    def _evaluate_protocol_slice(self, indices):
        return self._evaluate_protocol_slice_on_dataset(
            self.test_dataset,
            indices,
            normalize_loaded_tensors=False,
            oracle_stage_id=self._infer_stage_id_from_indices(self.test_dataset, indices),
        )

    def _evaluate_protocol_slice_on_dataset(
        self,
        dataset,
        indices,
        *,
        normalize_loaded_tensors: bool = False,
        oracle_stage_id=None,
    ):
        if oracle_stage_id is None:
            oracle_stage_id = self._infer_stage_id_from_indices(dataset, indices)
        subset = dataset.make_eval_subset(indices)
        loader = DataLoader(
            subset,
            batch_size=self.batchsize * 2,
            shuffle=False,
            num_workers=self.n_worker,
            collate_fn=safe_collate_drop_bad,
        )
        binary_predictions = []
        binary_target_values = []
        fake_scores = []
        decision_scores = []
        sample_thresholds = []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    logger.warning("Skipping empty protocol eval batch after dropping unreadable samples.")
                    continue
                images, _targets, binary_targets = batch
                if normalize_loaded_tensors:
                    images = self.test_transform_tensor(images)
                images = images.to(self.device)
                eval_mode = getattr(self.model_without_ddp, "eval_mode", "max_fake")
                force_expert_id = None
                if eval_mode == "oracle_current":
                    force_expert_id = oracle_stage_id
                elif eval_mode == "latest":
                    force_expert_id = int(getattr(self.model_without_ddp, "active_stage", self.task_id))
                logits, expert_ids, aggregate_scores = self._protocol_eval_logits_and_experts(
                    images,
                    force_expert_id=force_expert_id,
                )

                if aggregate_scores is not None:
                    decision_score_tensor = aggregate_scores.detach().cpu()
                    batch_decision_scores = decision_score_tensor.tolist()
                    thresholds = [0.0 for _ in batch_decision_scores]
                    batch_fake_scores = batch_decision_scores
                else:
                    decision_score_tensor = self._decision_scores_from_logits(logits).detach().cpu()
                    batch_decision_scores = decision_score_tensor.tolist()
                    if expert_ids is None:
                        thresholds = [
                            float(getattr(self, "_decision_threshold", 0.0))
                            for _ in batch_decision_scores
                        ]
                    else:
                        thresholds = [
                            self._threshold_for_expert(int(expert_id))
                            for expert_id in expert_ids.detach().cpu().tolist()
                        ]
                    if eval_mode in {"max_margin", "gate_margin", "feature_gaussian"}:
                        threshold_tensor = torch.tensor(
                            thresholds,
                            dtype=decision_score_tensor.dtype,
                        )
                        batch_fake_scores = (decision_score_tensor - threshold_tensor).tolist()
                    else:
                        batch_fake_scores = self._fake_scores_from_logits(logits).detach().cpu().tolist()
                binary_predictions.extend(
                    1 if score >= threshold else 0
                    for score, threshold in zip(batch_decision_scores, thresholds)
                )
                sample_thresholds.extend(float(threshold) for threshold in thresholds)
                binary_target_values.extend(int(item) for item in binary_targets.tolist())
                fake_scores.extend(batch_fake_scores)
                decision_scores.extend(batch_decision_scores)

        threshold_mode = str(
            self._cfg("threshold_mode", "online_f1") or "online_f1"
        ).lower()
        if threshold_mode == "balanced_prior":
            binary_predictions = self._balanced_prior_predictions(fake_scores)
        elif threshold_mode == "fixed_0_5":
            binary_predictions = [1 if score >= 0.5 else 0 for score in fake_scores]
        elif len(binary_predictions) != len(fake_scores):
            binary_predictions = [
                1 if score >= threshold else 0
                for score, threshold in zip(decision_scores, sample_thresholds)
            ]
        self._maybe_dump_score_diagnostics(
            dataset=dataset,
            oracle_stage_id=oracle_stage_id,
            binary_targets=binary_target_values,
            fake_scores=fake_scores,
            decision_scores=decision_scores,
            thresholds=sample_thresholds,
            binary_predictions=binary_predictions,
        )
        return compute_binary_detection_metrics(
            binary_target_values,
            binary_predictions,
            fake_scores,
        )

    def _maybe_dump_score_diagnostics(
        self,
        *,
        dataset,
        oracle_stage_id,
        binary_targets,
        fake_scores,
        decision_scores,
        thresholds,
        binary_predictions,
    ):
        if not bool(self._cfg("dump_score_data", False)):
            return
        if not decision_scores or not binary_targets:
            return

        split = "train" if getattr(dataset, "train", False) else "test"
        generator_name = self._generator_name_for_stage(dataset, oracle_stage_id)
        eval_stage_id = int(getattr(self, "task_id", -1))
        expert_id = -1 if oracle_stage_id is None else int(oracle_stage_id)
        online_sample = int(getattr(self, "online_samples_seen", 0))
        dump_id = self._score_dump_counter
        self._score_dump_counter += 1
        root = Path(self.log_dir) / "rigev1_score_diagnostics"
        root.mkdir(parents=True, exist_ok=True)

        summary = self._score_threshold_summary(
            binary_targets,
            decision_scores,
            thresholds,
        )
        summary.update(
            {
                "dump_id": dump_id,
                "eval_stage_id": eval_stage_id,
                "online_sample": online_sample,
                "split": split,
                "slice_stage_id": expert_id,
                "generator": generator_name,
            }
        )
        with (root / "threshold_summary.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")

    def _generator_name_for_stage(self, dataset, stage_id):
        if stage_id is None:
            return "unknown"
        try:
            return str(dataset.generator_order[int(stage_id)]["generator_name"])
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return f"stage{stage_id}"

    def _score_threshold_summary(self, labels, scores, current_thresholds):
        labels = [int(label) for label in labels]
        scores = [float(score) for score in scores]
        current_threshold = (
            sum(float(threshold) for threshold in current_thresholds) / len(current_thresholds)
            if current_thresholds
            else 0.0
        )
        thresholds = {
            "current": current_threshold,
            "fixed_margin0": 0.0,
            "balanced_prior_0_5": self._balanced_prior_threshold(scores, 0.5),
            "label_prior": self._balanced_prior_threshold(scores, sum(labels) / len(labels)),
            "gaussian_midpoint": self._gaussian_midpoint_threshold(scores, labels),
            "gaussian_intersection": self._gaussian_intersection_threshold(scores, labels),
            "best_acc": self._best_threshold_for_metric(scores, labels, metric="accuracy"),
            "best_f1": self._best_threshold_for_metric(scores, labels, metric="f1"),
        }
        threshold_metrics = {
            name: {
                "threshold": float(threshold),
                **self._binary_metrics_at_threshold(scores, labels, threshold),
            }
            for name, threshold in thresholds.items()
        }
        real_scores = [score for score, label in zip(scores, labels) if label == 0]
        fake_scores = [score for score, label in zip(scores, labels) if label == 1]
        return {
            "n": len(labels),
            "positive_count": int(sum(labels)),
            "negative_count": int(len(labels) - sum(labels)),
            "score_mean": self._mean(scores),
            "score_std": self._std(scores),
            "real_mean": self._mean(real_scores),
            "real_std": self._std(real_scores),
            "fake_mean": self._mean(fake_scores),
            "fake_std": self._std(fake_scores),
            "threshold_metrics": threshold_metrics,
        }

    @staticmethod
    def _mean(values):
        return None if not values else float(sum(values) / len(values))

    @staticmethod
    def _std(values):
        if not values:
            return None
        mean = sum(values) / len(values)
        return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))

    @staticmethod
    def _binary_metrics_at_threshold(scores, labels, threshold):
        predictions = [1 if score >= threshold else 0 for score in scores]
        total = len(labels)
        correct = sum(int(prediction == label) for prediction, label in zip(predictions, labels))
        tp = sum(1 for prediction, label in zip(predictions, labels) if prediction == 1 and label == 1)
        fp = sum(1 for prediction, label in zip(predictions, labels) if prediction == 1 and label == 0)
        fn = sum(1 for prediction, label in zip(predictions, labels) if prediction == 0 and label == 1)
        denom = 2 * tp + fp + fn
        return {
            "accuracy": 0.0 if total == 0 else float(correct / total),
            "f1": 0.0 if denom == 0 else float((2 * tp) / denom),
            "predicted_positive_rate": 0.0 if total == 0 else float(sum(predictions) / total),
        }

    def _best_threshold_for_metric(self, scores, labels, *, metric):
        if not scores:
            return 0.0
        pairs = sorted(
            ((float(score), int(label)) for score, label in zip(scores, labels)),
            key=lambda item: item[0],
            reverse=True,
        )
        total = len(pairs)
        positives = sum(label for _, label in pairs)
        tp = 0
        fp = 0
        fn = positives
        tn = total - positives
        best_threshold = pairs[0][0] + 1e-6
        best_value = -1.0

        def update_best(threshold):
            nonlocal best_threshold, best_value
            if metric == "accuracy":
                value = (tp + tn) / total
            else:
                denom = 2 * tp + fp + fn
                value = 0.0 if denom == 0 else (2 * tp) / denom
            if value > best_value:
                best_value = value
                best_threshold = threshold

        update_best(best_threshold)
        index = 0
        while index < total:
            score = pairs[index][0]
            while index < total and pairs[index][0] == score:
                label = pairs[index][1]
                if label == 1:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                    tn -= 1
                index += 1
            if index < total:
                threshold = 0.5 * (score + pairs[index][0])
            else:
                threshold = score - 1e-6
            update_best(float(threshold))
        return float(best_threshold)

    @staticmethod
    def _balanced_prior_threshold(scores, fake_prior):
        if not scores:
            return 0.0
        fake_prior = min(max(float(fake_prior), 0.0), 1.0)
        fake_count = int(round(len(scores) * fake_prior))
        if fake_count <= 0:
            return float(max(scores) + 1e-6)
        if fake_count >= len(scores):
            return float(min(scores) - 1e-6)
        ranked = sorted(scores, reverse=True)
        return float(ranked[fake_count - 1])

    def _infer_stage_id_from_indices(self, dataset, indices):
        indices = list(indices or [])
        if not indices:
            return None

        metadata = getattr(dataset, "metadata", None)
        if metadata is not None and "_online_stage_id" in getattr(metadata, "columns", []):
            stage_ids = []
            for index in indices[:128]:
                try:
                    stage_ids.append(int(metadata.iloc[int(index)]["_online_stage_id"]))
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
            if stage_ids:
                return Counter(stage_ids).most_common(1)[0][0]

        stage_indices = getattr(dataset, "stage_indices", None) or {}
        if stage_indices:
            first_indices = set(indices[:128])
            best_stage_id = None
            best_overlap = 0
            for stage_id, candidate_indices in stage_indices.items():
                overlap = len(first_indices.intersection(candidate_indices))
                if overlap > best_overlap:
                    best_stage_id = int(stage_id)
                    best_overlap = overlap
            if best_stage_id is not None:
                return best_stage_id
        return None

    def _balanced_prior_predictions(self, scores):
        if not scores:
            return []
        fake_prior = float(self._cfg("eval_fake_prior", 0.5) or 0.5)
        fake_prior = min(max(fake_prior, 0.0), 1.0)
        fake_count = int(round(len(scores) * fake_prior))
        fake_count = min(max(fake_count, 0), len(scores))
        ranked_ids = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        predictions = [0 for _ in scores]
        for idx in ranked_ids[:fake_count]:
            predictions[idx] = 1
        return predictions

    def _threshold_for_expert(self, expert_id):
        expert_id = int(expert_id)
        if expert_id <= 0:
            return float(self._decision_thresholds.get(0, 0.0))
        return float(
            self._decision_thresholds.get(
                expert_id,
                getattr(self, "_decision_threshold", 0.0),
            )
        )

    def _gate_threshold_for_expert(self, expert_id):
        expert_id = int(expert_id)
        return float(self._gate_thresholds.get(expert_id, 0.0))

    def _protocol_eval_logits_and_experts(self, images, force_expert_id=None):
        model = self.model_without_ddp
        z, online_z = model.extract_base_and_online_z(images)
        if force_expert_id is not None:
            force_expert_id = int(force_expert_id)
            if force_expert_id <= 0 or force_expert_id <= len(model.residual_heads):
                logits = model.combined_logits_from_z(
                    z,
                    online_z=online_z,
                    expert_id=max(force_expert_id, 0),
                ) + self.mask
                expert_ids = torch.full(
                    (z.size(0),),
                    max(force_expert_id, 0),
                    dtype=torch.long,
                    device=z.device,
                )
                return logits, expert_ids, None

        if not model.residual_heads:
            logits = model.base_head(z) + self.mask
            expert_ids = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            return logits, expert_ids, None

        expert_logits = model.expert_logits_from_z(z, online_z=online_z) + self.mask.view(1, 1, -1)
        eval_mode = getattr(model, "eval_mode", "max_fake")
        if eval_mode in {"oracle_current", "latest"}:
            eval_mode = "max_fake"
        if eval_mode == "max_confidence":
            expert_scores = torch.softmax(expert_logits, dim=-1).max(dim=-1).values
        elif eval_mode in {"max_margin", "mean_margin", "top2_margin"}:
            batch_size, expert_count, class_count = expert_logits.shape
            decision_scores = self._decision_scores_from_logits(
                expert_logits.reshape(batch_size * expert_count, class_count)
            ).view(batch_size, expert_count)
            thresholds = torch.tensor(
                [self._threshold_for_expert(expert_id) for expert_id in range(expert_count)],
                dtype=decision_scores.dtype,
                device=decision_scores.device,
            )
            margins = decision_scores - thresholds.view(1, -1)
            if eval_mode == "mean_margin":
                aggregate_scores = margins.mean(dim=1)
                return expert_logits[:, 0], None, aggregate_scores
            if eval_mode == "top2_margin":
                topk = min(2, int(margins.size(1)))
                aggregate_scores = margins.topk(topk, dim=1).values.mean(dim=1)
                return expert_logits[:, 0], None, aggregate_scores
            expert_scores = margins
        elif eval_mode == "gate_margin":
            batch_size, expert_count, class_count = expert_logits.shape
            flat_logits = expert_logits.reshape(batch_size * expert_count, class_count)
            decision_scores = self._decision_scores_from_logits(flat_logits).view(
                batch_size,
                expert_count,
            )
            decision_thresholds = torch.tensor(
                [self._threshold_for_expert(expert_id) for expert_id in range(expert_count)],
                dtype=decision_scores.dtype,
                device=decision_scores.device,
            )
            gate_scores = torch.abs(decision_scores - decision_thresholds.view(1, -1))
            gate_thresholds = torch.tensor(
                [self._gate_threshold_for_expert(expert_id) for expert_id in range(expert_count)],
                dtype=gate_scores.dtype,
                device=gate_scores.device,
            )
            expert_scores = gate_scores - gate_thresholds.view(1, -1)
        elif eval_mode == "feature_gaussian":
            expert_scores = self._feature_gaussian_scores(
                online_z,
                expert_logits.size(1),
            )
        else:
            expert_scores = torch.softmax(expert_logits, dim=-1)[:, :, 1]
        expert_ids = torch.argmax(expert_scores, dim=1)
        batch_ids = torch.arange(z.size(0), device=z.device)
        return expert_logits[batch_ids, expert_ids], expert_ids, None
