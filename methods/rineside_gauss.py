import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from datasets import safe_collate_drop_bad
from methods._trainer import _Trainer


class RineSideGauss(_Trainer):
    def __init__(self, *args, **kwargs):
        super(RineSideGauss, self).__init__(*args, **kwargs)
        self.task_id = 0

    def online_before_task(self, task_id):
        self.task_id = int(task_id)
        if getattr(self.model_without_ddp, "projector_dim", 0) > 0:
            self.model_without_ddp.begin_stage(self.task_id)

    def online_step(self, images, labels, idx):
        del idx
        self.add_new_class(labels)

        y = labels.clone()
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = images.to(self.device, non_blocking=True)
        y = y.to(self.device)
        x = self.test_transform_tensor(x)

        if getattr(self.model_without_ddp, "projector_dim", 0) > 0:
            return self._online_projection_step(x, y)

        self.model.eval()
        with torch.no_grad():
            z = self.model_without_ddp.extract_z(x)
            z_all, y_all = self._gather_batch_for_stats(z, y)
            self.model_without_ddp.update_statistics(self.task_id, z_all, y_all)

            logits = self.model_without_ddp.gaussian_logits_from_z(z)
            logits = logits + self._batch_logit_mask(y)
            loss = self.criterion(logits, y)
            _, preds = logits.topk(self.topk, 1, True, True)
            acc = torch.sum(preds == y.unsqueeze(1)).item() / y.size(0)

        return loss.item(), acc

    def _online_projection_step(self, x, y):
        model = self.model_without_ddp
        model.train()
        model.backbone.eval()

        with torch.no_grad():
            z = model.extract_z(x)
            z_all, y_all = self._gather_batch_for_stats(z, y)

        replay_per_class = int(getattr(self, "rine_gauss_replay_per_class", 0) or 0)
        replay_weight = float(getattr(self, "rine_gauss_replay_weight", 1.0) or 0.0)
        active_heads = model.active_head_ids()
        old_heads = [head_id for head_id in active_heads if head_id != self.task_id]

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits = model.head_logits_from_z(self.task_id, z)
            logits = logits + self._batch_logit_mask(y)
            current_loss = self.criterion(logits, y)
            loss = current_loss

            replay_losses = []
            if replay_per_class > 0 and replay_weight > 0:
                for head_id in old_heads:
                    if not model.has_projected_replay(head_id):
                        continue
                    h_current = model.project_z(head_id, z.detach())
                    replay_x, replay_y = model.sample_projected_replay(
                        head_id,
                        replay_per_class,
                    )
                    if replay_x.numel() == 0:
                        continue
                    mix_x = torch.cat([h_current, replay_x.to(h_current.device)], dim=0)
                    mix_y = torch.cat([y, replay_y.to(y.device)], dim=0)
                    replay_logits = model.detector_logits_from_h(head_id, mix_x)
                    replay_losses.append(self.criterion(replay_logits, mix_y))
                if replay_losses:
                    loss = loss + replay_weight * torch.stack(replay_losses).mean()

        _, preds = logits.topk(self.topk, 1, True, True)
        acc = torch.sum(preds == y.unsqueeze(1)).item() / y.size(0)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        with torch.no_grad():
            for head_id in old_heads:
                if not bool(model.head_active[head_id].item()):
                    continue
                model.update_projected_statistics_from_z(head_id, z_all, y_all)

        return loss.item(), acc

    def online_after_task(self, task_id):
        del task_id
        if getattr(self.model_without_ddp, "projector_dim", 0) <= 0:
            return
        self.model_without_ddp.set_train_stage(None)
        self._rebuild_current_projected_statistics()

    @torch.no_grad()
    def _rebuild_current_projected_statistics(self):
        indices = getattr(self.train_dataset, "stage_indices", {}).get(self.task_id, [])
        if not indices:
            return

        self.model_without_ddp.eval()
        self.model_without_ddp.reset_projected_statistics(self.task_id)
        worker_count = min(int(getattr(self, "n_worker", 0) or 0), 4)
        loader = DataLoader(
            Subset(self.online_iter_dataset, indices),
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=False,
            persistent_workers=worker_count > 0,
            collate_fn=safe_collate_drop_bad,
        )
        for batch in loader:
            if self._skip_empty_batch(batch, f"rineside_stats_stage_{self.task_id}"):
                continue
            images, labels, _idx = batch
            y = labels.clone()
            for j in range(len(y)):
                y[j] = self.exposed_classes.index(y[j].item())
            x = images.to(self.device, non_blocking=True)
            y = y.to(self.device)
            x = self.test_transform_tensor(x)
            z = self.model_without_ddp.extract_z(x)
            z_all, y_all = self._gather_batch_for_stats(z, y)
            self.model_without_ddp.update_projected_statistics_from_z(
                self.task_id,
                z_all,
                y_all,
            )

    def _batch_logit_mask(self, y):
        if self.no_batchmask:
            return self.mask
        logit_mask = torch.zeros_like(self.mask) - torch.inf
        for cls_idx in torch.unique(y):
            logit_mask[cls_idx] = 0
        return logit_mask

    def _gather_batch_for_stats(self, z, y):
        if not self.distributed:
            return z, y

        world_size = dist.get_world_size()
        local_size = torch.tensor(z.size(0), device=z.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        max_size = int(torch.stack(sizes).max().item())

        z_pad = z.new_zeros((max_size,) + tuple(z.shape[1:]))
        y_pad = y.new_zeros((max_size,))
        z_pad[: z.size(0)] = z
        y_pad[: y.size(0)] = y

        z_parts = [torch.zeros_like(z_pad) for _ in range(world_size)]
        y_parts = [torch.zeros_like(y_pad) for _ in range(world_size)]
        dist.all_gather(z_parts, z_pad)
        dist.all_gather(y_parts, y_pad)

        z_all = []
        y_all = []
        for size, z_part, y_part in zip(sizes, z_parts, y_parts):
            n = int(size.item())
            z_all.append(z_part[:n])
            y_all.append(y_part[:n])
        return torch.cat(z_all, dim=0), torch.cat(y_all, dim=0)
