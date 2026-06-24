import logging

import torch
from torch.utils.data import DataLoader

from datasets import safe_collate_drop_bad
from methods._trainer import _Trainer
from protocol_metrics import compute_binary_detection_metrics


logger = logging.getLogger()


class RINEResidual(_Trainer):
    """Frozen CAID features with one binary expert head per protocol stage."""

    def __init__(self, *args, **kwargs):
        super(RINEResidual, self).__init__(*args, **kwargs)
        self.task_id = 0
        self._current_head_optimizer = None
        self._threshold_scores = []
        self._threshold_labels = []
        self._threshold_scores_by_expert = {}
        self._threshold_labels_by_expert = {}
        self._decision_thresholds = {}
        self._decision_threshold = 0.5

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

        head = model.add_residual_head(self.task_id).to(self.device)
        if self.distributed:
            logger.warning(
                "RINE-Residual dynamically adds heads and is intended for single-GPU runs; "
                "DDP parameter synchronization is not implemented for new heads."
            )
        self._current_head_optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(getattr(self, "lr", 1e-3)),
            weight_decay=float(getattr(self, "rine_residual_weight_decay", 1e-4) or 0.0),
        )
        logger.info(
            "RINE-Residual stage %s | base_head=%s | online_head=%s | online_dim=%s | inner_steps=%s",
            self.task_id,
            getattr(model, "head_type", None),
            getattr(model, "online_head_type", None),
            getattr(model, "online_feature_dim", None),
            getattr(self, "rine_residual_inner_steps", 1),
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

        optimizer = self._current_head_optimizer or self.optimizer
        inner_steps = max(1, int(getattr(self, "rine_residual_inner_steps", 1) or 1))
        loss_value = 0.0
        acc_value = 0.0

        for _ in range(inner_steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self.task_id == 0:
                    logits = model.base_head(z)
                else:
                    logits = model.current_head()(online_z)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            with torch.no_grad():
                if self.task_id == 0:
                    current_logits = model.base_head(z)
                else:
                    current_logits = model.current_head()(online_z)
                _, preds = current_logits.topk(self.topk, 1, True, True)
                acc = torch.sum(preds == y.unsqueeze(1)).item() / y.size(0)

            loss_value = float(loss.detach().cpu())
            acc_value = float(acc)

        if self.task_id == 0:
            self.update_schedule()
            self._update_threshold_buffer(0, current_logits.detach(), y.detach())
        else:
            self._update_threshold_buffer(self.task_id, current_logits.detach(), y.detach())
        return loss_value, acc_value

    def _fake_scores_from_logits(self, logits):
        probabilities = torch.softmax(logits.float(), dim=-1)
        fake_class_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        for logit_index, original_class in enumerate(self.exposed_classes[: logits.size(-1)]):
            if original_class != 0:
                fake_class_mask[logit_index] = True
        if not torch.any(fake_class_mask):
            return torch.zeros(logits.size(0), dtype=probabilities.dtype, device=logits.device)
        return probabilities[:, fake_class_mask].sum(dim=-1)

    def _update_threshold_buffer(self, expert_id, logits, labels):
        scores = self._fake_scores_from_logits(logits).detach().cpu().tolist()
        binary_labels = [
            0 if self.exposed_classes[int(label)] == 0 else 1
            for label in labels.detach().cpu().tolist()
        ]
        expert_id = int(expert_id)
        self._threshold_scores_by_expert.setdefault(expert_id, []).extend(
            float(score) for score in scores
        )
        self._threshold_labels_by_expert.setdefault(expert_id, []).extend(
            int(label) for label in binary_labels
        )
        self._threshold_scores.extend(float(score) for score in scores)
        self._threshold_labels.extend(int(label) for label in binary_labels)
        self._decision_thresholds[expert_id] = self._best_f1_threshold(
            self._threshold_scores_by_expert[expert_id],
            self._threshold_labels_by_expert[expert_id],
        )
        self._decision_threshold = self._best_f1_threshold(
            self._threshold_scores,
            self._threshold_labels,
        )

    @staticmethod
    def _best_f1_threshold(scores, labels):
        if not scores or len(set(labels)) < 2:
            return 0.5
        pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
        positives = sum(labels)
        tp = 0
        fp = 0
        best_f1 = -1.0
        best_threshold = 0.5
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

    def _evaluate_protocol_slice(self, indices):
        return self._evaluate_protocol_slice_on_dataset(
            self.test_dataset,
            indices,
            normalize_loaded_tensors=False,
        )

    def _evaluate_protocol_slice_on_dataset(
        self,
        dataset,
        indices,
        *,
        normalize_loaded_tensors: bool = False,
    ):
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
                logits, expert_ids = self._protocol_eval_logits_and_experts(images)

                batch_fake_scores = self._fake_scores_from_logits(logits).detach().cpu().tolist()
                if expert_ids is None:
                    thresholds = [
                        float(getattr(self, "_decision_threshold", 0.5))
                        for _ in batch_fake_scores
                    ]
                else:
                    thresholds = [
                        self._threshold_for_expert(int(expert_id))
                        for expert_id in expert_ids.detach().cpu().tolist()
                    ]
                binary_predictions.extend(
                    1 if score >= threshold else 0
                    for score, threshold in zip(batch_fake_scores, thresholds)
                )
                sample_thresholds.extend(float(threshold) for threshold in thresholds)
                binary_target_values.extend(int(item) for item in binary_targets.tolist())
                fake_scores.extend(batch_fake_scores)

        threshold_mode = str(
            getattr(self, "rine_residual_threshold_mode", "online_f1") or "online_f1"
        ).lower()
        if threshold_mode == "balanced_prior":
            binary_predictions = self._balanced_prior_predictions(fake_scores)
        elif threshold_mode == "fixed_0_5":
            binary_predictions = [1 if score >= 0.5 else 0 for score in fake_scores]
        elif len(binary_predictions) != len(fake_scores):
            binary_predictions = [
                1 if score >= threshold else 0
                for score, threshold in zip(fake_scores, sample_thresholds)
            ]
        return compute_binary_detection_metrics(
            binary_target_values,
            binary_predictions,
            fake_scores,
        )

    def _balanced_prior_predictions(self, scores):
        if not scores:
            return []
        fake_prior = float(getattr(self, "rine_residual_eval_fake_prior", 0.5) or 0.5)
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
            return 0.5
        return float(
            self._decision_thresholds.get(
                expert_id,
                getattr(self, "_decision_threshold", 0.5),
            )
        )

    def _protocol_eval_logits_and_experts(self, images):
        model = self.model_without_ddp
        z, online_z = model.extract_base_and_online_z(images)
        if not model.residual_heads:
            logits = model.base_head(z) + self.mask
            expert_ids = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
            return logits, expert_ids

        expert_logits = model.expert_logits_from_z(z, online_z=online_z) + self.mask.view(1, 1, -1)
        if getattr(model, "eval_mode", "max_fake") == "max_confidence":
            expert_scores = torch.softmax(expert_logits, dim=-1).max(dim=-1).values
        else:
            expert_scores = torch.softmax(expert_logits, dim=-1)[:, :, 1]
        expert_ids = torch.argmax(expert_scores, dim=1)
        batch_ids = torch.arange(z.size(0), device=z.device)
        return expert_logits[batch_ids, expert_ids], expert_ids
