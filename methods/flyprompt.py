import gc

import torch

from methods._trainer import _Trainer


class FlyPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(FlyPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        self.collect(images.clone(), labels.clone())

        # Update internal session schedule based only on the number of samples
        # seen during the online phase.
        if hasattr(self, "_maybe_advance_internal_session"):
            batch_size_global = images.size(0) * self.world_size
            self._maybe_advance_internal_session(batch_size_global)

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def collect(self, images, labels):
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        images = images.to(self.device)
        labels = labels.to(self.device)

        images = self.test_transform_tensor(images)

        with torch.no_grad():
            self.model.eval()
            self.model_without_ddp.collect(images, labels)

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x,y,mask=logit_mask)
        else:
            logit, loss = self.model_forward(x,y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        # Update EMA heads for the expert corresponding to the current
        # internal session. This avoids using benchmark task ids.
        if hasattr(self.model_without_ddp, "update_ema_fc"):
            self.model_without_ddp.update_ema_fc()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def _ensemble_logits(self, logit_ls):
        if not hasattr(self, 'ensemble_method'):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # Shape: [batch_size, n_classes, n_experts]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)  # [batch_size, n_experts]
            min_entropy_indices = torch.argmin(entropies, dim=-1)  # [batch_size]
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        """Hook called after each benchmark task.

        We keep ``task_id`` for task bookkeeping only; the underlying model's
        internal session state is advanced exclusively via the task-free
        online scheduler.
        """
        self.task_id += 1
