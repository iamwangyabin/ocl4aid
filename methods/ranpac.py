import logging

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class RanPAC(_Trainer):
    def __init__(self, *args, **kwargs):
        super(RanPAC, self).__init__(*args, **kwargs)

        self.task_id = 0

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)

        if self.task_id == 0:
            return self._train_first_task(images, labels)
        else:
            return self._collect_features_for_statistics(images, labels)

    def _train_first_task(self, images, labels):
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        return _loss / _iter, _acc / _iter

    def after_base_stage_train(self, base_stage_id):
        if self.task_id != 0:
            return

        logger.info("Collecting final base-stage features for RanPAC statistics")
        model_obj = self.model.module if self.distributed else self.model
        model_obj.setup_rp()
        model_obj.freeze_all_except_classifier()
        self.train_sampler.set_task(base_stage_id)
        self.train_sampler.set_epoch(self.base_stage_epochs)

        collected = 0
        for batch in self.train_dataloader:
            if self._skip_empty_batch(batch, "base_feature_collection"):
                continue
            images, labels, _idx = batch
            collected += images.size(0) * self.world_size
            self._collect_features_for_statistics(images, labels)

        logger.info("Collected %s final base-stage samples for RanPAC statistics", collected)

    def _collect_features_for_statistics(self, images, labels):
        images_copy = images.clone()
        labels_copy = labels.clone()
        
        # Map labels to exposed class indices
        for j in range(len(labels_copy)):
            labels_copy[j] = self.exposed_classes.index(labels_copy[j].item())

        images_copy = images_copy.to(self.device)
        labels_copy = labels_copy.to(self.device)

        # images_copy = self.test_transform_tensor(images_copy)
        images_copy = self.train_transform(images_copy)

        with torch.no_grad():
            self.model.eval()
            if self.distributed:
                self.model.module.collect_features_labels(images_copy, labels_copy)
            else:
                self.model.collect_features_labels(images_copy, labels_copy)

        return 0.0, 0.0  # No training loss/acc for subsequent tasks

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
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_before_task(self, task_id):
        if task_id == 0:
            if not self.distributed:
                if self.model.use_g_prompt:
                    self.model.freeze_backbone_except_prompts()
                    logger.info("First task: g-prompts and classifier enabled for training")
                else:
                    self.model.freeze_backbone_except_adapters()
                    logger.info("First task: adapters and classifier enabled for training")
            else:
                if self.model.module.use_g_prompt:
                    self.model.module.freeze_backbone_except_prompts()
                    logger.info("First task: g-prompts and classifier enabled for training")
                else:
                    self.model.module.freeze_backbone_except_adapters()
                    logger.info("First task: adapters and classifier enabled for training")
        else:
            if not self.distributed:
                self.model.freeze_all_except_classifier()
            else:
                self.model.module.freeze_all_except_classifier()
            
            if not self.distributed:
                mode = "g-prompt" if self.model.use_g_prompt else "adapter"
            else:
                mode = "g-prompt" if self.model.module.use_g_prompt else "adapter"
            logger.info(f"Task {task_id} ({mode} mode): collecting features for RanPAC statistics")

    def online_after_task(self, cur_iter):
        model_obj = self.model.module if self.distributed else self.model
        if self.task_id == 0:
            logger.info("Completing first task training, setting up random projection")

            model_obj.setup_rp()
            model_obj.freeze_all_except_classifier()
            model_obj.update_statistics_and_classifier()

            logger.info("Random projection initialized, adapters frozen")
        else:
            logger.info("Updating RanPAC statistics and classifier for task %s", self.task_id)
            model_obj.update_statistics_and_classifier()

        if self.task_id + 1 < getattr(self, "n_tasks", 1):
            self._advance_model_task_count()

        logger.info(f"Task {self.task_id} completed, statistics updated")
        self.task_id += 1
