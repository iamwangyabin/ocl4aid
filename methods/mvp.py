import copy
import datetime
import gc
import logging
import time

import torch
import torch.nn.functional as F

from methods._trainer import _Trainer

logger = logging.getLogger()


class MVP(_Trainer):
    def __init__(self, **kwargs):
        super(MVP, self).__init__(**kwargs)

        self.use_afs  = True
        self.use_mcr  = True
        self.use_mask = True

        self.alpha  = 0.5
        self.gamma  = 2.0
        self.margin  = 0.5

        self.task_id = 0
    @torch.no_grad()
    def _collect_rp_features(self, images, labels):
        """Collect features for RPFC gating (if enabled).

        Uses backbone CLS features and current internal session id as regression
        targets, following the FlyPrompt RPFC design.
        """
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        rp_head = getattr(self.model_without_ddp, "rp_head", None)
        if not use_rp_gate or rp_head is None:
            return

        # Map labels to seen-class indices, consistent with training.
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device)

        images = self.test_transform_tensor(images)

        self.model_without_ddp.backbone.eval()
        if hasattr(self.model_without_ddp.backbone, "forward_features"):
            feats = self.model_without_ddp.backbone.forward_features(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            cls_feat = feats[:, 0]
        else:
            x = self.model_without_ddp.backbone.patch_embed(images)
            B, N, D = x.size()
            cls_token = self.model_without_ddp.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.model_without_ddp.backbone.pos_drop(token_appended + self.model_without_ddp.backbone.pos_embed)
            x = self.model_without_ddp.backbone.blocks(x)
            x = self.model_without_ddp.backbone.norm(x)
            cls_feat = x[:, 0]

        session_id = getattr(self, "current_session", 0)
        session_labels = torch.full(
            (labels.size(0),),
            session_id,
            device=labels.device,
            dtype=torch.long,
        )
        self.model_without_ddp.rp_head.collect(cls_feat, session_labels)



    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        # Collect RPFC features once per online update (per internal session), using
        # the original images/labels (before deletion).
        self._collect_rp_features(images.clone(), labels.clone())

        # Update internal session schedule based only on the number of samples
        # seen during the online phase.
        if hasattr(self, "_maybe_advance_internal_session"):
            batch_size_global = images.size(0) * self.world_size
            self._maybe_advance_internal_session(batch_size_global)

        del (images, labels)
        gc.collect()
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

        # EMA classifier head update (optional, per-prompt-slot experts)
        if getattr(self.model_without_ddp, "use_ema_head", False) and hasattr(self.model_without_ddp, "update_ema_fc"):
            expert_ids = getattr(self.model_without_ddp, "last_expert_ids", None)
            if expert_ids is not None:
                self.model_without_ddp.update_ema_fc(expert_ids=expert_ids)


        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            feature, mvp_mask = self.model_without_ddp.forward_features(x)
            logit = self.model_without_ddp.forward_head(feature)
            if mask is not None: # batchmask
                logit += mask
            elif self.use_mask:
                logit = logit * mvp_mask
                logit = logit + self.mask
            loss = self.loss_fn(feature, mvp_mask, y)
        return logit, loss

    def _ensemble_logits(self, logit_ls):
        """Ensemble a list of logits from online and EMA heads.

        The behavior mirrors FlyPrompt's implementation and is controlled by
        ``self.ensemble_method`` (default: softmax_max_prob).
        """
        if not hasattr(self, "ensemble_method"):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # [batch_size, n_classes, n_heads]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)
            min_entropy_indices = torch.argmin(entropies, dim=-1)
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

    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.model_without_ddp.backbone.fc)
        head.zero_grad()
        logit = head(feature.detach())
        if self.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask

        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim

        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim

        return sample_grad, batch_grad

    def _get_ignore(self, sample_grad, batch_grad):
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1))#B
        return ign_score

    def _get_compensation(self, y, feat):
        head_w = self.model_without_ddp.backbone.fc.weight[y].clone().detach()
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score

    def loss_fn(self, feature, mask, y):
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)

        if self.use_afs:
            logit = self.model_without_ddp.forward_head(feature)
            logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.model_without_ddp.forward_head(feature)
        if self.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(log_p, y)

        if self.use_mcr:
            loss = (1-self.alpha)* loss + self.alpha * (ign_score ** self.gamma) * loss
        return loss.mean() + self.model_without_ddp.get_similarity_loss()

    def report_training(self, sample_num, train_loss, train_acc):
        fallback_total = getattr(self, "total_samples", sample_num)
        total_training_samples = max(int(getattr(self, "total_training_samples", fallback_total)), sample_num)
        elapsed = time.time() - self.start_time
        remaining = max(total_training_samples - sample_num, 0)
        eta_seconds = int(elapsed * remaining / sample_num) if sample_num > 0 else 0
        logger.info(
             f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
             f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
             f"running_time {datetime.timedelta(seconds=int(elapsed))} | "
             f"ETA {datetime.timedelta(seconds=eta_seconds)} | "
             f"N_Prompts {self.model_without_ddp.e_prompts.size(0)} | "
             f"N_Exposed {len(self.exposed_classes)} | "
             f"Counts {self.model_without_ddp.count.to(torch.int64).tolist()}"
             )
