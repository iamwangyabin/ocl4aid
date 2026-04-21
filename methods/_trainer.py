import datetime
import json
import logging
import os
import random
import sys
import time
import math
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import *
from protocol_metrics import StageMetrics, compute_online_metrics
from utils.augment import Cutout
from utils.data_loader import get_statistics
from utils.onlinesampler import ManifestStageSampler, OnlineSampler, OnlineTestSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
mp.set_sharing_strategy('file_system')


class _Trainer():
    def __init__(self, *args, **kwargs) -> None:

        self.kwargs = kwargs
        self.__dict__.update(kwargs)

        self.start_time = time.time()
        self.eval_period = np.inf if self.eval_period < 0 else self.eval_period
        self.is_protocol_dataset = getattr(self, "dataset", None) == "openfake_protocol"

        # Internal step-based schedule (task-boundary-free) for selected methods.
        method_name = getattr(self, "method", None)
        step_aware_methods = {"dualprompt", "mvp", "flyprompt"}
        if method_name in step_aware_methods:
            # step_num > 1; if not provided or <=0, default to n_tasks.
            self.step_num = getattr(self, "step_num", None)
            if self.step_num is None or self.step_num <= 0:
                if hasattr(self, "n_tasks"):
                    self.step_num = self.n_tasks
            if self.step_num is not None and self.step_num <= 1:
                raise ValueError(f"step_num must be > 1, got {self.step_num}")
        else:
            # Other methods keep using the original task-id based schedule.
            self.step_num = None

        # These will be fully initialized once dataset size is known.
        self.current_step = 0
        self.current_step_seen_samples = 0
        self.samples_per_step = None

        # Distributed training setup
        self.world_size = 1
        self.ngpus_per_nodes = torch.cuda.device_count()
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size  = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size  = self.world_size * self.ngpus_per_nodes

        self.distributed = self.world_size > 1
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        if self.distributed:
            self.batchsize = self.batchsize // self.world_size

        self.log_dir = f"{self.log_path}/logs/{self.dataset}/{self.note}"

        os.makedirs(self.log_dir, exist_ok=True)

        return

    def setup_distributed_dataset(self):

        self.datasets = DATASETS
        if self.is_protocol_dataset:
            self._setup_protocol_dataset()
            return

        mean, std, n_classes, inp_size, in_channels = get_statistics(dataset=self.dataset)
        inp_size = 224 # override for ViT
        self.n_classes = n_classes
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        train_transform = []
        self.cutmix = "cutmix" in self.transforms
        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
        if "autoaug" in self.transforms:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))

        self.train_transform = transforms.Compose([
                lambda x: (x * 255).to(torch.uint8),
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                lambda x: x.float() / 255,
                # transforms.ToTensor(),
                transforms.Normalize(mean, std),])
        logger.info(f"Using train-transforms {train_transform}")
        self.test_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),])

        # Create tensor-compatible test transform for cases where input is already a tensor
        self.test_transform_tensor = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                # No ToTensor() since input is already a tensor
                transforms.Normalize(mean, std),])

        if 'imagenet' in self.dataset or 'cub' in self.dataset or 'car' in self.dataset:
            self.load_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor()])
        else:
            self.load_transform = transforms.ToTensor()

        self.train_dataset = self.datasets[self.dataset](root=self.data_dir, train=True,  download=True, transform=self.load_transform)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)
        self.test_dataset = self.datasets[self.dataset](root=self.data_dir, train=False, download=True, transform=self.test_transform)

        _r = dist.get_rank() if self.distributed else None       # means that it is not distributed
        _w = dist.get_world_size() if self.distributed else None # means that it is not distributed
        self.train_sampler = OnlineSampler(self.online_iter_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, 0, self.rnd_NM, _w, _r)
        self.train_dataloader = DataLoader(self.online_iter_dataset, batch_size=self.batchsize, sampler=self.train_sampler, pin_memory=False, num_workers=0)
        self.test_sampler = OnlineTestSampler(self.test_dataset, [], _w, _r)

        self.seen = 0
        self.exposed_classes = []
        self.disjoint_classes = self.train_sampler.disjoint_classes
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf

    def _setup_protocol_dataset(self):
        if not getattr(self, "protocol_manifest", None):
            raise ValueError("--protocol_manifest is required for dataset=openfake_protocol")

        mean, std, n_classes, inp_size, in_channels = get_statistics(dataset=self.dataset)
        del in_channels
        inp_size = 224
        self.n_classes = n_classes
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        train_transform = []
        self.cutmix = "cutmix" in self.transforms
        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
        if "autoaug" in self.transforms:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy("imagenet")))

        self.train_transform = transforms.Compose([
            lambda x: (x * 255).to(torch.uint8),
            transforms.Resize((inp_size, inp_size)),
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            lambda x: x.float() / 255,
            transforms.Normalize(mean, std),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.test_transform_tensor = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.Normalize(mean, std),
        ])
        self.load_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
        ])

        self.train_dataset = self.datasets[self.dataset](
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.load_transform,
            protocol_manifest=self.protocol_manifest,
        )
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset, 1)
        self.test_dataset = self.datasets[self.dataset](
            root=self.data_dir,
            train=False,
            download=False,
            transform=self.test_transform,
            protocol_manifest=self.protocol_manifest,
        )

        _r = dist.get_rank() if self.distributed else None
        _w = dist.get_world_size() if self.distributed else None
        self.train_sampler = ManifestStageSampler(
            self.online_iter_dataset,
            self.train_dataset.stage_indices,
            _w,
            _r,
        )
        self.train_dataloader = DataLoader(
            self.online_iter_dataset,
            batch_size=self.batchsize,
            sampler=self.train_sampler,
            pin_memory=False,
            num_workers=0,
        )
        self.test_sampler = None
        self.n_tasks = len(self.train_dataset.stage_indices)
        self.protocol_generator_order = self.train_dataset.generator_order
        if self.method in {"dualprompt", "mvp", "flyprompt"}:
            raw_step_num = self.kwargs.get("step_num", None)
            if raw_step_num is None or raw_step_num <= 0:
                self.step_num = self.n_tasks

        self.seen = 0
        self.exposed_classes = []
        self.disjoint_classes = []
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf

    def setup_distributed_model(self):

        logger.info(f"Building model: {self.method}")
        self.model = select_model(self.method, self.backbone, self.n_classes, self.n_tasks, self.kwargs).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.model.to(self.device)
        self.model_without_ddp = self.model

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = getattr(self.model_without_ddp, "loss_fn", nn.CrossEntropyLoss(reduction="mean"))
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.lr_gamma = 0.99995 if 'imagenet' in self.dataset else 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        logger.info(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        learnables = [n for n, p in self.model_without_ddp.named_parameters() if p.requires_grad]
        logger.info(f"Learnable Parameters :\t{n_params}")
        logger.info(learnables)
        logger.info("")

    def run(self):
        if self.profile:
            self.profile_worker(0)
        else:
            # Distributed Launch
            if self.ngpus_per_nodes > 1:
                mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
            else:
                self.main_worker(0)

    def _init_internal_step_scheduler(self):
        """Initialize internal step schedule based on training set size.

        The step schedule is intentionally decoupled from benchmark tasks:
        step boundaries are determined only by how many training samples have
        been seen in total.
        """
        if getattr(self, "step_num", None) is None:
            return
        if self.step_num <= 1:
            # Already validated in __init__, but guard for safety.
            raise ValueError(f"step_num must be > 1, got {self.step_num}")
        if not hasattr(self, "total_samples"):
            return
        if self.total_samples <= 0:
            return

        # Use training set size to determine how many samples belong to each
        # internal step (approximate, sampler may re-order samples).
        self.samples_per_step = max(1, self.total_samples // self.step_num)
        self.current_step = 0
        self.current_step_seen_samples = 0

    def _maybe_advance_internal_step(self, batch_size: int):
        """Advance internal step counter purely based on seen samples.

        This does not use any ground-truth task boundary information. When a
        new step begins, the underlying model is notified via
        ``process_task_count()``, if implemented.
        """
        if getattr(self, "step_num", None) is None:
            return
        if getattr(self, "samples_per_step", None) is None:
            return
        if self.step_num <= 1 or batch_size <= 0:
            return

        self.current_step_seen_samples += batch_size
        while self.current_step < self.step_num - 1 and self.current_step_seen_samples >= self.samples_per_step:
            self.current_step_seen_samples -= self.samples_per_step
            self.current_step += 1

            model_obj = getattr(self, "model_without_ddp", None)
            if model_obj is None:
                model_obj = getattr(self, "model", None)
            if model_obj is not None and hasattr(model_obj, "process_task_count"):
                model_obj.process_task_count()

    def _run_protocol_loop(self):
        logger.info(f"[2] Incrementally training protocol stages ({self.n_tasks})")
        samples_cnt = 0
        num_report = 2000
        report_period = 500
        stage_metrics = []

        for task_id in range(self.n_tasks):
            stage_name = self.protocol_generator_order[task_id]["generator_name"]
            logger.info("\n")
            logger.info("#" * 50)
            logger.info(f"# Stage {task_id}: {stage_name}")
            logger.info("#" * 50 + "\n")

            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
                for images, labels, idx in self.train_dataloader:
                    samples_cnt += images.size(0) * self.world_size
                    loss, acc = self.online_step(images, labels, idx)
                    if samples_cnt + images.size(0) * self.world_size > num_report:
                        self.report_training(samples_cnt, loss, acc)
                        num_report += report_period
                    sys.stdout.flush()

            if self.is_main_process():
                stage_metric = self._evaluate_protocol_stage(task_id)
                stage_metrics.append(stage_metric)
                internal_avg = (
                    sum(stage_metric.internal_accuracy_by_generator.values()) / len(stage_metric.internal_accuracy_by_generator)
                    if stage_metric.internal_accuracy_by_generator else 0.0
                )
                external_avg = (
                    sum(stage_metric.external_accuracy_by_subset.values()) / len(stage_metric.external_accuracy_by_subset)
                    if stage_metric.external_accuracy_by_subset else 0.0
                )
                logger.info(
                    "Protocol Eval | stage %s | avg_internal_acc %.4f | plasticity %.4f | external_avg %.4f",
                    task_id,
                    internal_avg,
                    stage_metric.internal_accuracy_by_generator.get(stage_name, 0.0),
                    external_avg,
                )

            self.online_after_task(task_id)

        if self.is_main_process():
            metrics = compute_online_metrics(stage_metrics)
            summary = {
                "stage_metrics": [
                    {
                        "stage_id": item.stage_id,
                        "new_generators": item.new_generators,
                        "internal_accuracy_by_generator": item.internal_accuracy_by_generator,
                        "external_accuracy_by_subset": item.external_accuracy_by_subset,
                    }
                    for item in stage_metrics
                ],
                "metrics": metrics,
            }
            output_path = os.path.join(self.log_dir, f"seed_{self.rnd_seed}_ocl_metrics.json")
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            logger.info("Saved protocol metrics to %s", output_path)

    def _evaluate_protocol_stage(self, stage_id: int) -> StageMetrics:
        self.model.eval()
        seen_generators = [
            entry["generator_name"]
            for entry in self.protocol_generator_order[: stage_id + 1]
        ]
        current_generator = self.protocol_generator_order[stage_id]["generator_name"]
        self._prepare_protocol_eval()

        internal_scores = {}
        for generator_name in seen_generators:
            if generator_name not in self.test_dataset.internal_slices:
                continue
            internal_scores[generator_name] = self._evaluate_protocol_slice(
                self.test_dataset.internal_slices[generator_name]
            )

        external_scores = {}
        for subset_name, indices in self.test_dataset.external_slices.items():
            external_scores[subset_name] = self._evaluate_protocol_slice(indices)

        return StageMetrics(
            stage_id=stage_id,
            internal_accuracy_by_generator=internal_scores,
            external_accuracy_by_subset=external_scores,
            new_generators=[current_generator],
        )

    def _prepare_protocol_eval(self):
        if self.method == "flyprompt" and hasattr(self.model_without_ddp, "update"):
            self.model_without_ddp.update()
        elif self.method == "sprompt":
            if getattr(self.model_without_ddp, "use_rp_gate", False) and hasattr(self.model_without_ddp, "update"):
                self.model_without_ddp.update()
            elif hasattr(self, "_cur_task_features") and len(self._cur_task_features) > 0:
                self._build_prototypes_for_task(self.task_id)
        elif self.method in {"hide", "hide_lora", "hide_adapter", "norga"}:
            if getattr(self.model_without_ddp, "use_rp_gate", False) and hasattr(self.model_without_ddp, "update"):
                self.model_without_ddp.update()

    def _evaluate_protocol_slice(self, indices):
        subset = self.test_dataset.make_eval_subset(indices)
        loader = DataLoader(
            subset,
            batch_size=self.batchsize * 2,
            shuffle=False,
            num_workers=self.n_worker,
        )
        total_correct = 0
        total_num = 0
        with torch.no_grad():
            for images, _targets, binary_targets in loader:
                images = images.to(self.device)
                logits = self._protocol_eval_logits(images)
                pred_indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                binary_targets = binary_targets.tolist()
                for pred_index, binary_target in zip(pred_indices, binary_targets):
                    original_class = self.exposed_classes[pred_index]
                    pred_binary = 0 if original_class == 0 else 1
                    total_correct += int(pred_binary == binary_target)
                    total_num += 1
        return total_correct / total_num if total_num > 0 else 0.0

    def _protocol_eval_logits(self, images):
        if self.method == "flyprompt":
            logit_raw = self.model_without_ddp.forward_with_rp(images)
            expert_ids = torch.argmax(logit_raw, dim=-1)
            logit_ls = self.model_without_ddp.forward_with_ema(images, expert_ids=expert_ids)
            logit_ls = [logit + self.mask for logit in logit_ls]
            return self._ensemble_logits(logit_ls)

        if self.method == "sprompt":
            use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
            use_ema_head = getattr(self.model_without_ddp, "use_ema_head", False)
            if use_rp_gate:
                logit_task = self.model_without_ddp.forward_with_rp(images)
                expert_count = self.task_id + 1
                logit_task = logit_task[:, :expert_count]
                expert_ids = torch.argmax(logit_task, dim=-1)
            else:
                expert_ids = self._route_batch_by_prototypes(images)
            if use_ema_head:
                logit_ls = self.model_without_ddp.forward_with_ema(images, expert_ids=expert_ids)
                logit_ls = [logit + self.mask for logit in logit_ls]
                return self._ensemble_logits(logit_ls)
            return self.model(images, expert_ids=expert_ids) + self.mask

        if self.method in {"hide", "hide_lora", "hide_adapter", "norga"}:
            task_hat = self._predict_task_from_gate(images)
            use_ema_head = getattr(self.model_without_ddp, "use_ema_head", False)
            if use_ema_head and hasattr(self.model_without_ddp, "forward_prompt_with_ema"):
                logit_ls = self.model_without_ddp.forward_prompt_with_ema(images, task_id=task_hat)
                logit_ls = [logit + self.mask for logit in logit_ls]
                return self._ensemble_logits(logit_ls)
            logit_prompt, _ = self.model_without_ddp.forward_prompt(images, task_id=task_hat)
            return logit_prompt + self.mask

        if self.method in {"dualprompt", "mvp"} and getattr(self.model_without_ddp, "use_ema_head", False):
            logit_ls = self.model_without_ddp.forward_with_ema(images)
            logit_ls = [logit + self.mask for logit in logit_ls]
            return self._ensemble_logits(logit_ls)

        if self.method == "codaprompt":
            result = self.model(images)
            logits = result[0] if isinstance(result, tuple) else result
            return logits + self.mask

        if self.method in {"l2p", "dualprompt", "mvp", "ranpac", "singleprompt", "slca", "sdlora"}:
            return self.model(images) + self.mask

        raise NotImplementedError(
            f"Protocol evaluation is not implemented for method={self.method}"
        )


    def main_worker(self, gpu) -> None:
        # ========= Distributed training setup =========
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                logger.info(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                logger.info(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12702'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
            logger.info(
                'You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.'
            )
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)
        self._init_internal_step_scheduler()

        logger.info(f"[1] Select a GCL method ({self.method})")
        self.setup_distributed_model()

        if self.is_protocol_dataset:
            self._run_protocol_loop()
            return

        # =========== Incrementally training ===========
        logger.info(f"[2] Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period
        num_report = 2000
        report_period = 500

        for task_id in range(self.n_tasks):

            logger.info("\n")
            logger.info("#" * 50)
            logger.info(f"# Task {task_id} iteration")
            logger.info("#" * 50 + "\n")
            logger.info("[2-1] Prepare a datalist for the current task")

            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
                for i, (images, labels, idx) in enumerate(self.train_dataloader):
                    samples_cnt += images.size(0) * self.world_size

                    loss, acc = self.online_step(images, labels, idx)

                    if samples_cnt + images.size(0) * self.world_size > num_report:
                        self.report_training(samples_cnt, loss, acc)
                        num_report += report_period

                    if samples_cnt + images.size(0) * self.world_size > num_eval:
                        with torch.no_grad():
                            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                            eval_dict = self.online_evaluate(test_dataloader)
                            if self.distributed:
                                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                                eval_dict = eval_dict.cpu().numpy()
                                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
                            if self.is_main_process():
                                eval_results["test_acc"].append(eval_dict['avg_acc'])
                                eval_results["avg_acc"].append(eval_dict['cls_acc'])
                                eval_results["data_cnt"].append(num_eval)
                                self.report_test(num_eval, eval_dict["avg_loss"], eval_dict['avg_acc'])
                            num_eval += self.eval_period

                    sys.stdout.flush()

                test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                eval_dict = self.online_evaluate(test_dataloader, task_id=task_id, end=True)

            self.online_after_task(task_id)

            if self.distributed:
                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                eval_dict = eval_dict.cpu().numpy()
                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
            task_acc = eval_dict['avg_acc']

            logger.info("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_acc)
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            logger.info("[2-5] Report task result")
            logger.info(task_records['task_acc'])

        # ================== Summary ===================
        if self.is_main_process():

            # Accuracy (A)
            A_auc = np.mean(eval_results["test_acc"])
            A_avg = np.mean(task_records["task_acc"])
            A_last = task_records["task_acc"][self.n_tasks - 1]

            # Forgetting (F)
            cls_acc = np.array(task_records["cls_acc"])
            acc_diff = []
            if self.n_tasks > 1:
                for j in range(self.n_classes):
                    if np.max(cls_acc[:-1, j]) > 0:
                        acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
                F_last = np.mean(acc_diff)
            else:
                F_last = -999

            # Backward Transfer (BWT), class-level: last accuracy minus accuracy
            # when the class was first learned (first non-zero accuracy before last task)
            if self.n_tasks > 1:
                bwt_vals = []
                for j in range(self.n_classes):
                    per_cls_prev = cls_acc[:-1, j]
                    seen_indices = np.where(per_cls_prev > 0)[0]
                    if len(seen_indices) == 0:
                        continue
                    first_acc = per_cls_prev[seen_indices[0]]
                    last_acc = cls_acc[-1, j]
                    bwt_vals.append(last_acc - first_acc)
                if len(bwt_vals) > 0:
                    BWT_last = np.mean(bwt_vals)
                else:
                    BWT_last = -999
            else:
                BWT_last = -999

            logger.info(f"======== Summary =======")
            logger.info(self.note)
            logger.info(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
            logger.info(f"BWT_last {BWT_last}")
            logger.info(f"="*24)
            logger.info(eval_results['test_acc'])

            np.save(f"{self.log_dir}/seed_{self.rnd_seed}.npy", task_records["task_acc"])

            if self.eval_period != np.inf:
                np.save(f'{self.log_dir}/seed_{self.rnd_seed}_eval.npy', eval_results['test_acc'])
                np.save(f'{self.log_dir}/seed_{self.rnd_seed}_eval_time.npy', eval_results['data_cnt'])

            # Optional post-hoc expert representation analysis (e.g., FlyPrompt/DualPrompt/MVP).
            if getattr(self, "analysis_expert_similarity", False):
                if hasattr(self, "analyze_expert_features"):
                    logger.info("[Post] Running expert feature similarity / CKA analysis ...")
                    try:
                        self.analyze_expert_features()
                    except Exception as e:
                        logger.exception("[Post] Expert feature analysis failed: %s", e)
                else:
                    logger.info(
                        "[Post] analysis_expert_similarity=True but method has no "
                        "analyze_expert_features; skipping expert analysis."
                    )

    def profile_worker(self, gpu) -> None:
        # ============ Toy experiment setup ============
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                logger.info(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                logger.info(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12702'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass

        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
        cudnn.benchmark = False

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)
        self._init_internal_step_scheduler()

        self.setup_distributed_model()

        samples_cnt = 0
        self.train_sampler.set_task(0)
        self.online_before_task(0)
        for i, (images, labels, idx) in enumerate(self.train_dataloader):
            samples_cnt += images.size(0) * self.world_size
            loss, acc = self.online_step(images, labels, idx)
            self.report_training(samples_cnt, loss, acc)
            break
        self.online_after_task(0)

    def add_new_class(self, class_name):
        exposed_classes = []
        new = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
                new.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        self.mask[:len(self.exposed_classes)] = 0

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, samples_cnt):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_after_task(self, task_id):
        raise NotImplementedError()

    def online_evaluate(self, test_loader, samples_cnt, task_id=None, end=False):
        raise NotImplementedError()

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables print, logging when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

        class MasterOnlyFilter(logging.Filter):
            def __init__(self, is_master):
                super().__init__()
                self.is_master = is_master

            def filter(self, record):
                return self.is_master or record.levelno < logging.INFO

        for h in logging.getLogger().handlers:
            h.addFilter(MasterOnlyFilter(is_master))

    def report_training(self, sample_num, train_loss, train_acc):
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples*self.num_epochs-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=self.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs

    def train_data_config(self, n_task, train_dataset, train_sampler):
        for t_i in range(n_task):
            train_sampler.set_task(t_i)
            train_dataloader = DataLoader(train_dataset,batch_size=self.batchsize,sampler=train_sampler,num_workers=4)
            data_info={}
            for i,data in enumerate(train_dataloader):
                _,label = data
                label = label.to(self.device)
                for b in range(len(label)):
                    if 'Class_'+str(label[b].item()) in data_info.keys():
                        data_info['Class_'+str(label[b].item())] += 1
                    else:
                        data_info['Class_'+str(label[b].item())] = 1
            logger.info(f"[Train] Task{t_i} Data Info")
            logger.info(data_info)
            convert_data_info = self.convert_class_label(data_info)
            np.save(f"{self.log_dir}/seed_{self.rnd_seed}_task{t_i}_train_data.npy", convert_data_info)
            logger.info(f"[Train] Task{t_i} Converted Data Info")
            logger.info(convert_data_info)
            logger.info("")

    def test_data_config(self, test_dataloader, task_id):
        data_info={}
        for i,data in enumerate(test_dataloader):
            _,label = data
            label = label.to(self.device)
            for b in range(len(label)):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())]+=1
                else:
                    data_info['Class_'+str(label[b].item())]=1
        logger.info("[Test] Exposed Classes:")
        logger.info(self.exposed_classes)
        logger.info(f"[Test] Task {task_id} Data Info")
        logger.info(data_info)
        logger.info(f"[Test] Task{task_id} Converted Data Info")
        convert_data_info = self.convert_class_label(data_info)
        logger.info(convert_data_info)
        logger.info("")

    def convert_class_label(self,data_info):
        #* self.class_list => original class label
        self.class_list = self.train_dataset.classes
        for key in list(data_info.keys()):
            old_key= int(key[6:])
            data_info[self.class_list[old_key]] = data_info.pop(key)
        return data_info

    def current_task_data(self,train_loader):
        data_info={}
        for i,data in enumerate(train_loader):
            _,label = data
            for b in range(label.shape[0]):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())] +=1
                else:
                    data_info['Class_'+str(label[b].item())] =1
        logger.info("[Current Task] Data Info")
        logger.info(data_info)
        logger.info("[Current Task] Converted Data Info")
        convert_data_info = self.convert_class_label(data_info)
        logger.info(convert_data_info)
        logger.info("")
