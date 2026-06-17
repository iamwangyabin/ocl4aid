import torch

from methods._trainer import _Trainer


class CodaPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(CodaPrompt, self).__init__(*args, **kwargs)

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

        return _loss / _iter, _acc / _iter

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

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        ortho_loss = None
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            res = self.model(x)
            if isinstance(res, tuple):
                logit, ortho_loss = res
            else:
                logit = res
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)
            if ortho_loss is not None:
                loss += ortho_loss

        return logit, loss

    def online_after_task(self, cur_iter):
        del cur_iter
        if self.task_id + 1 < getattr(self, "n_tasks", 1):
            self._advance_model_task_count()
        self.task_id += 1
