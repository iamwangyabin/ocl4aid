import gc
import logging

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class SinglePromptTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super(SinglePromptTrainer, self).__init__(*args, **kwargs)

    def online_step(self, images, labels, idx):
        del idx
        self.add_new_class(labels)
        total_loss, total_acc, total_iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            total_loss += loss
            total_acc += acc
            total_iter += 1

        del images, labels
        gc.collect()
        return total_loss / total_iter, total_acc / total_iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data
        for idx in range(len(y)):
            y[idx] = self.exposed_classes.index(y[idx].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        for cls_idx in torch.unique(y):
            logit_mask[cls_idx] = 0

        x = x.to(self.device)
        y = y.to(self.device)
        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x, y, mask=logit_mask)
        else:
            logit, loss = self.model_forward(x, y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask
            loss = self.criterion(logit, y)
        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        del task_id
        del end
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                x, y = data
                for idx in range(len(y)):
                    y[idx] = self.exposed_classes.index(y[idx].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x) + self.mask
                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        return {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

    def online_before_task(self, task_id):
        del task_id

    def online_after_task(self, cur_iter):
        del cur_iter
