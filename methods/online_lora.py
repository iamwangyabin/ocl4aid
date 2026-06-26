import logging
import math
from collections import deque

import torch
import torch.nn.functional as F

from methods._trainer import _Trainer


logger = logging.getLogger()


class OnlineLoRA(_Trainer):
    """Official-style Online-LoRA trainer adapted to CAID binary detection."""

    def __init__(self, *args, **kwargs):
        super(OnlineLoRA, self).__init__(*args, **kwargs)
        self._loss_window = deque(maxlen=max(1, int(self.online_lora_loss_window)))
        self._new_peak_detected = True
        self._last_loss_window_mean = 0.0
        self._last_loss_window_variance = 0.0
        self._online_lora_steps = 0
        self._hard_buffer = []

    def online_step(self, images, labels, idx):
        del idx
        self.add_new_class(labels)
        total_loss, total_acc, total_iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            total_loss += loss
            total_acc += acc
            total_iter += 1

        return total_loss / total_iter, total_acc / total_iter

    def online_train(self, data):
        self.model.train()
        x, y = data

        for idx in range(len(y)):
            y[idx] = self.exposed_classes.index(y[idx].item())

        x = self.train_transform(x.to(self.device))
        y = y.to(self.device)
        buffer_x, buffer_y = self._hard_buffer_tensors()
        train_mask = self._training_logit_mask(y, buffer_y)

        self.optimizer.zero_grad()
        logit, loss, ce_loss, mas_loss, current_losses, buffer_losses = self.model_forward(
            x,
            y,
            buffer_x=buffer_x,
            buffer_y=buffer_y,
            mask=train_mask,
        )

        _, preds = logit.topk(self.topk, 1, True, True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()
        self._online_lora_steps += 1

        consolidated = self._update_loss_dynamics_and_maybe_consolidate(float(ce_loss.detach().item()))
        self._update_hard_buffer(x, y, current_losses, buffer_losses)

        if consolidated and self.is_main_process():
            logger.info(
                "Online-LoRA consolidated new LoRA | step=%s | ce=%.6f | mas=%.6f | buffer=%s",
                self._online_lora_steps,
                float(ce_loss.detach().item()),
                float(mas_loss.detach().item()),
                len(self._hard_buffer),
            )

        correct = torch.sum(preds == y.unsqueeze(1)).item()
        return loss.item(), correct / y.size(0)

    def model_forward(self, x, y, *, buffer_x=None, buffer_y=None, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self._apply_mask(self.model(x), mask)
            current_losses = F.cross_entropy(logit, y, reduction="none")
            ce_loss = current_losses.mean()

            buffer_losses = None
            if buffer_x is not None and buffer_y is not None and buffer_y.numel() > 0:
                buffer_logit = self._apply_mask(self.model(buffer_x), mask)
                buffer_losses = F.cross_entropy(buffer_logit, buffer_y, reduction="none")
                ce_loss = ce_loss + buffer_losses.mean()

            mas_loss = self.model_without_ddp.regularization_loss()
            loss = ce_loss + 0.5 * float(self.online_lora_mas_weight) * mas_loss

        return logit, loss, ce_loss, mas_loss, current_losses, buffer_losses

    def _apply_mask(self, logits, mask):
        if mask is not None:
            return logits + mask
        return logits + self.mask

    def _training_logit_mask(self, y, buffer_y):
        if self.no_batchmask:
            return self.mask
        logit_mask = torch.zeros_like(self.mask) - torch.inf
        labels = y if buffer_y is None else torch.cat([y, buffer_y.to(y.device)], dim=0)
        for cls_id in torch.unique(labels):
            logit_mask[cls_id] = 0
        return logit_mask

    def _hard_buffer_tensors(self):
        if not self._hard_buffer:
            return None, None
        images = torch.stack([item["image"] for item in self._hard_buffer], dim=0).to(
            self.device,
            non_blocking=True,
        )
        labels = torch.tensor(
            [int(item["label"]) for item in self._hard_buffer],
            dtype=torch.long,
            device=self.device,
        )
        return images, labels

    def _update_hard_buffer(self, current_x, current_y, current_losses, buffer_losses):
        buffer_size = max(0, int(self.online_lora_hard_buffer_size))
        if buffer_size <= 0:
            self._hard_buffer = []
            return

        candidates = []
        current_loss_values = current_losses.detach().float().cpu()
        current_images = current_x.detach().cpu()
        current_labels = current_y.detach().cpu()
        for idx in range(current_images.size(0)):
            candidates.append(
                {
                    "loss": float(current_loss_values[idx].item()),
                    "image": current_images[idx].clone(),
                    "label": int(current_labels[idx].item()),
                }
            )

        if self._hard_buffer:
            if buffer_losses is not None:
                replay_loss_values = buffer_losses.detach().float().cpu()
            else:
                replay_loss_values = torch.tensor(
                    [float(item.get("loss", 0.0)) for item in self._hard_buffer],
                    dtype=torch.float32,
                )
            for idx, item in enumerate(self._hard_buffer):
                candidates.append(
                    {
                        "loss": float(replay_loss_values[idx].item()),
                        "image": item["image"].detach().cpu().clone(),
                        "label": int(item["label"]),
                    }
                )

        candidates.sort(key=lambda item: item["loss"], reverse=True)
        self._hard_buffer = candidates[:buffer_size]

    def _update_loss_dynamics_and_maybe_consolidate(self, loss_value: float) -> bool:
        self._loss_window.append(float(loss_value))
        if len(self._loss_window) < self._loss_window.maxlen:
            return False

        losses = torch.tensor(list(self._loss_window), dtype=torch.float32)
        loss_mean = float(losses.mean().item())
        loss_variance = float(losses.var(unbiased=False).item())

        if (
            not self._new_peak_detected
            and loss_mean > self._last_loss_window_mean + math.sqrt(max(self._last_loss_window_variance, 0.0))
        ):
            self._new_peak_detected = True

        stable_after_peak = (
            self._new_peak_detected
            and loss_mean < float(self.online_lora_loss_mean_threshold)
            and loss_variance < float(self.online_lora_loss_variance_threshold)
        )
        if not stable_after_peak or not self._hard_buffer:
            return False

        gradients = self._estimate_importance_from_hard_buffer()
        updated = self.model_without_ddp.update_omega_from_gradients(gradients)
        if updated <= 0:
            return False

        self.model_without_ddp.merge_and_reset_lora()
        self._last_loss_window_mean = loss_mean
        self._last_loss_window_variance = loss_variance
        self._new_peak_detected = False
        return True

    def _estimate_importance_from_hard_buffer(self):
        if not self._hard_buffer:
            return {}

        model = self.model_without_ddp
        was_training = model.training
        model.eval()
        gradients = {
            name: torch.zeros_like(param.detach())
            for name, param in model.wnew_named_parameters()
        }
        batch_size = max(1, int(self.online_lora_importance_batch_size))

        for start in range(0, len(self._hard_buffer), batch_size):
            batch = self._hard_buffer[start : start + batch_size]
            images = torch.stack([item["image"] for item in batch], dim=0).to(self.device)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(images) + self.mask
                pseudo_labels = torch.argmax(logits.detach(), dim=1)
                omega_loss = F.nll_loss(F.log_softmax(logits, dim=1), pseudo_labels)
                omega_loss.backward()

            for name, param in model.wnew_named_parameters():
                if param.grad is None:
                    continue
                gradients[name] = gradients[name].to(param.device) + torch.nan_to_num(
                    param.grad.detach().float().abs().pow(2),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

        model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        return gradients

    def _checkpoint_method_state(self):
        return {
            "online_lora_importance": self.model_without_ddp.export_importance_state(),
            "online_lora_steps": int(self._online_lora_steps),
            "online_lora_loss_window": list(self._loss_window),
            "online_lora_new_peak_detected": bool(self._new_peak_detected),
            "online_lora_last_loss_window_mean": float(self._last_loss_window_mean),
            "online_lora_last_loss_window_variance": float(self._last_loss_window_variance),
            "online_lora_hard_buffer": [
                {
                    "loss": float(item["loss"]),
                    "image": item["image"].detach().cpu(),
                    "label": int(item["label"]),
                }
                for item in self._hard_buffer
            ],
        }

    def _load_checkpoint_method_state(self, state):
        importance_state = state.get("online_lora_importance", {})
        self.model_without_ddp.load_importance_state(importance_state)
        self._online_lora_steps = int(state.get("online_lora_steps", 0))
        self._new_peak_detected = bool(state.get("online_lora_new_peak_detected", True))
        self._last_loss_window_mean = float(state.get("online_lora_last_loss_window_mean", 0.0))
        self._last_loss_window_variance = float(
            state.get("online_lora_last_loss_window_variance", 0.0)
        )
        self._loss_window.clear()
        for value in state.get("online_lora_loss_window", []):
            self._loss_window.append(float(value))
        self._hard_buffer = []
        for item in state.get("online_lora_hard_buffer", []):
            image = item.get("image")
            if torch.is_tensor(image):
                self._hard_buffer.append(
                    {
                        "loss": float(item.get("loss", 0.0)),
                        "image": image.detach().cpu(),
                        "label": int(item.get("label", 0)),
                    }
                )
