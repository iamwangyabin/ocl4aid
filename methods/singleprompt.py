import torch

from methods._trainer import _Trainer


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
