import torch
import torch.distributed as dist

from methods._trainer import _Trainer


class RineSideGauss(_Trainer):
    def __init__(self, *args, **kwargs):
        super(RineSideGauss, self).__init__(*args, **kwargs)
        self.task_id = 0

    def online_before_task(self, task_id):
        self.task_id = int(task_id)

    def online_step(self, images, labels, idx):
        del idx
        self.add_new_class(labels)

        y = labels.clone()
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        x = images.to(self.device, non_blocking=True)
        y = y.to(self.device)
        x = self.test_transform_tensor(x)

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
