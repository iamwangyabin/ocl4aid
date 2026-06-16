import datetime
import atexit
import json
import logging
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CAIDBenchmarkProtocol, OnlineIterDataset
from protocol_metrics import StageMetrics, compute_online_metrics
from utils.augment import Cutout
from utils.onlinesampler import ManifestStageSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
mp.set_sharing_strategy('file_system')

DATASET_NAME = "caidbench_protocol"
CAIDBENCH_MEAN = (0.485, 0.456, 0.406)
CAIDBENCH_STD = (0.229, 0.224, 0.225)
CAIDBENCH_INPUT_SIZE = 224


class _Trainer():
    def __init__(self, *args, **kwargs) -> None:

        self.kwargs = kwargs
        self.__dict__.update(kwargs)

        self.start_time = time.time()
        self.base_epochs = int(getattr(self, "base_epochs", 1))
        if self.base_epochs < 1:
            raise ValueError(f"base_epochs must be >= 1, got {self.base_epochs}")
        self.base_batchsize = getattr(self, "base_batchsize", None)
        if self.base_batchsize is not None:
            self.base_batchsize = int(self.base_batchsize)
            if self.base_batchsize < 1:
                raise ValueError(f"base_batchsize must be >= 1, got {self.base_batchsize}")

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
        self._swanlab = None
        self._swanlab_run = None
        self._swanlab_enabled = False
        self._swanlab_atexit_registered = False
        self._swanlab_resolved_experiment_name = None

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
            if self.base_batchsize is not None:
                self.base_batchsize = max(1, self.base_batchsize // self.world_size)

        self.log_dir = f"{self.log_path}/logs/{DATASET_NAME}/{self.note}"

        os.makedirs(self.log_dir, exist_ok=True)

        return

    def _sanitize_swanlab_value(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._sanitize_swanlab_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._sanitize_swanlab_value(item) for key, item in value.items()}
        return str(value)

    def _swanlab_config(self):
        config = {
            key: self._sanitize_swanlab_value(value)
            for key, value in sorted(self.kwargs.items())
        }
        config.update({
            "dataset": DATASET_NAME,
            "log_dir": self.log_dir,
            "world_size": self.world_size,
            "distributed": self.distributed,
            "swanlab_resolved_project": self.swanlab_project,
            "swanlab_resolved_experiment_name": self._swanlab_experiment_name(),
        })
        return config

    def _swanlab_experiment_name(self):
        if getattr(self, "swanlab_experiment_name", None):
            return self.swanlab_experiment_name
        if self._swanlab_resolved_experiment_name is not None:
            return self._swanlab_resolved_experiment_name
        base_name = self.note or self.method or "run"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._swanlab_resolved_experiment_name = f"{base_name}_{timestamp}"
        return self._swanlab_resolved_experiment_name

    def _metric_slug(self, value):
        slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
        return slug or "unknown"

    def _to_swanlab_scalar(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return value.detach().cpu().item()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (int, float, bool)):
            return value
        return None

    def _init_swanlab(self):
        if not getattr(self, "use_swanlab", False):
            return
        if getattr(self, "swanlab_mode", "cloud") == "disabled":
            return
        if not self.is_main_process():
            return
        try:
            import swanlab
        except ModuleNotFoundError:
            logger.warning(
                "SwanLab logging requested but package 'swanlab' is not installed. "
                "Install it or run with --no_swanlab."
            )
            return

        init_kwargs = {
            "project": self.swanlab_project,
            "experiment_name": self._swanlab_experiment_name(),
            "description": self.swanlab_description,
            "group": self.swanlab_group,
            "tags": self.swanlab_tags,
            "config": self._swanlab_config(),
            "logdir": self.swanlab_logdir or os.path.join(self.log_dir, "swanlab"),
            "mode": self.swanlab_mode,
            "public": self.swanlab_public,
        }
        if getattr(self, "swanlab_workspace", None):
            init_kwargs["workspace"] = self.swanlab_workspace
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

        try:
            self._swanlab = swanlab
            try:
                self._swanlab_run = swanlab.init(**init_kwargs)
            except TypeError as e:
                logger.warning(
                    "SwanLab init rejected optional arguments, retrying with basic arguments: %s",
                    e,
                )
                basic_keys = {"project", "experiment_name", "config", "logdir", "mode", "workspace"}
                basic_kwargs = {
                    key: value
                    for key, value in init_kwargs.items()
                    if key in basic_keys
                }
                self._swanlab_run = swanlab.init(**basic_kwargs)
            self._swanlab_enabled = True
            if not self._swanlab_atexit_registered:
                atexit.register(self._finish_swanlab)
                self._swanlab_atexit_registered = True
            logger.info(
                "SwanLab logging enabled: project=%s experiment=%s mode=%s",
                init_kwargs.get("project"),
                init_kwargs.get("experiment_name"),
                init_kwargs.get("mode"),
            )
        except Exception as e:
            self._swanlab = None
            self._swanlab_run = None
            self._swanlab_enabled = False
            logger.exception("Failed to initialize SwanLab logging: %s", e)

    def _log_swanlab(self, metrics, step=None):
        if not self._swanlab_enabled or self._swanlab is None:
            return
        if not self.is_main_process():
            return
        payload = {}
        for key, value in metrics.items():
            scalar = self._to_swanlab_scalar(value)
            if scalar is not None:
                payload[key] = scalar
        if not payload:
            return
        try:
            if step is None:
                self._swanlab.log(payload)
            else:
                self._swanlab.log(payload, step=int(step))
        except Exception as e:
            self._swanlab_enabled = False
            logger.exception("SwanLab logging failed and has been disabled: %s", e)

    def _finish_swanlab(self):
        if not self._swanlab_enabled or self._swanlab is None:
            return
        try:
            self._swanlab.finish()
        except Exception as e:
            logger.exception("Failed to finish SwanLab run cleanly: %s", e)
        finally:
            self._swanlab_enabled = False

    def setup_distributed_dataset(self):
        mean = CAIDBENCH_MEAN
        std = CAIDBENCH_STD
        inp_size = CAIDBENCH_INPUT_SIZE
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        train_transform = []
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

        self.train_dataset = CAIDBenchmarkProtocol(
            root=self.caidbench_data_dir,
            train=True,
            download=False,
            transform=self.load_transform,
            protocol_path=self.caidbench_protocol,
            index_path=self.caidbench_index_path,
            label_mode=self.caidbench_label_mode,
            image_column=self.caidbench_image_column,
        )
        self.n_classes = len(self.train_dataset.label_space)
        self.online_iter_dataset = OnlineIterDataset(self.train_dataset)
        self.test_dataset = CAIDBenchmarkProtocol(
            root=self.caidbench_data_dir,
            train=False,
            download=False,
            transform=self.test_transform,
            protocol_path=self.caidbench_protocol,
            index_path=self.caidbench_index_path,
            label_mode=self.caidbench_label_mode,
            image_column=self.caidbench_image_column,
        )

        _r = dist.get_rank() if self.distributed else None
        _w = dist.get_world_size() if self.distributed else None
        self.train_sampler = ManifestStageSampler(
            self.online_iter_dataset,
            self.train_dataset.stage_indices,
            _w,
            _r,
            seed=self.rnd_seed,
        )
        self.train_dataloader = DataLoader(
            self.online_iter_dataset,
            batch_size=self.batchsize,
            sampler=self.train_sampler,
            pin_memory=False,
            num_workers=self.n_worker,
            persistent_workers=self.n_worker > 0,
        )
        base_batchsize = self.base_batchsize or self.batchsize
        self.base_train_dataloader = DataLoader(
            self.online_iter_dataset,
            batch_size=base_batchsize,
            sampler=self.train_sampler,
            pin_memory=False,
            num_workers=self.n_worker,
            persistent_workers=self.n_worker > 0,
        )
        self.test_sampler = None
        self.protocol_stage_ids = list(self.train_dataset.active_stage_ids)
        if not self.protocol_stage_ids:
            raise ValueError("CAIDBenchmark protocol has no non-empty training stages.")
        self.n_tasks = len(self.protocol_stage_ids)
        self.protocol_generator_order = self.train_dataset.generator_order
        if self.method in {"dualprompt", "mvp", "flyprompt"}:
            raw_step_num = self.kwargs.get("step_num", None)
            if raw_step_num is None or raw_step_num <= 0:
                self.step_num = self.n_tasks

        self.exposed_classes = []
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
        self.lr_gamma = 0.9999
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
        """Initialize internal step schedule based on dataloader-observed samples.

        The step schedule is intentionally decoupled from benchmark tasks:
        step boundaries are determined only by how many training samples have
        been seen in total, including repeated base-session passes.
        """
        if getattr(self, "step_num", None) is None:
            return
        if self.step_num <= 1:
            # Already validated in __init__, but guard for safety.
            raise ValueError(f"step_num must be > 1, got {self.step_num}")

        schedule_total = int(getattr(self, "total_training_samples", getattr(self, "total_samples", 0)))
        if schedule_total <= 0:
            return

        # Use the actual number of dataloader-observed samples so base_epochs > 1
        # does not advance internal prompt steps too early.
        self.samples_per_step = max(1, schedule_total // self.step_num)
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

        for task_pos, stage_id in enumerate(self.protocol_stage_ids):
            stage_name = self.protocol_generator_order[stage_id]["generator_name"]
            logger.info("\n")
            logger.info("#" * 50)
            logger.info(f"# Stage {stage_id}: {stage_name}")
            logger.info("#" * 50 + "\n")

            self.train_sampler.set_task(stage_id)
            self.online_before_task(stage_id)
            stage_epochs = self.base_epochs if task_pos == 0 else 1
            if task_pos == 0:
                logger.info(f"Base session epochs: {stage_epochs} | batch_size {self.base_batchsize or self.batchsize}")
            else:
                logger.info(f"Online stage: single pass | batch_size {self.batchsize}")
            train_dataloader = self.base_train_dataloader if task_pos == 0 else self.train_dataloader
            for epoch in range(stage_epochs):
                logger.info(f"Pass {epoch + 1}/{stage_epochs}")
                for images, labels, idx in train_dataloader:
                    samples_cnt += images.size(0) * self.world_size
                    loss, acc = self.online_step(images, labels, idx)
                    if samples_cnt + images.size(0) * self.world_size > num_report:
                        self.report_training(samples_cnt, loss, acc)
                        num_report += report_period
                    sys.stdout.flush()

            if self.is_main_process():
                stage_metric = self._evaluate_protocol_stage(stage_id)
                stage_metrics.append(stage_metric)
                internal_avg = (
                    sum(stage_metric.internal_accuracy_by_generator.values()) / len(stage_metric.internal_accuracy_by_generator)
                    if stage_metric.internal_accuracy_by_generator else 0.0
                )
                logger.info(
                    "Protocol Eval | stage %s | avg_internal_acc %.4f | plasticity %.4f",
                    stage_id,
                    internal_avg,
                    stage_metric.internal_accuracy_by_generator.get(stage_name, 0.0),
                )
                swanlab_metrics = {
                    "protocol/stage": stage_id,
                    "protocol/internal_avg_acc": internal_avg,
                    "protocol/current_generator_acc": stage_metric.internal_accuracy_by_generator.get(stage_name, 0.0),
                }
                for generator_name, score in stage_metric.internal_accuracy_by_generator.items():
                    swanlab_metrics[f"protocol/internal/{self._metric_slug(generator_name)}"] = score
                self._log_swanlab(swanlab_metrics, step=stage_id)

            self.online_after_task(stage_id)

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
            if stage_metrics:
                last_stage_id = stage_metrics[-1].stage_id
                final_metrics = {}
                for key, values_by_stage in metrics.items():
                    if not isinstance(values_by_stage, dict):
                        continue
                    value = values_by_stage.get(last_stage_id)
                    if isinstance(value, (int, float, np.generic)) and value is not None:
                        final_metrics[f"protocol/final/{self._metric_slug(key)}"] = value
                self._log_swanlab(final_metrics, step=last_stage_id)

    def _expected_training_samples(self):
        if not hasattr(self, "train_dataset") or not hasattr(self, "protocol_stage_ids"):
            return getattr(self, "total_samples", 0)
        stage_indices = getattr(self.train_dataset, "stage_indices", {})
        total = sum(len(indices) for indices in stage_indices.values())
        if not self.protocol_stage_ids:
            return total
        base_stage_id = self.protocol_stage_ids[0]
        base_count = len(stage_indices.get(base_stage_id, []))
        return total + max(self.base_epochs - 1, 0) * base_count

    def _evaluate_protocol_stage(self, stage_id: int) -> StageMetrics:
        self.model.eval()
        seen_generators = [
            entry["generator_name"]
            for entry in self.protocol_generator_order[: stage_id + 1]
        ]
        current_generator = self.protocol_generator_order[stage_id]["generator_name"]
        stage_generators = getattr(self.train_dataset, "stage_generators", {}).get(stage_id, [])
        self._prepare_protocol_eval()

        internal_scores = {}
        for generator_name in seen_generators:
            if generator_name not in self.test_dataset.internal_slices:
                continue
            internal_scores[generator_name] = self._evaluate_protocol_slice(
                self.test_dataset.internal_slices[generator_name]
            )

        return StageMetrics(
            stage_id=stage_id,
            internal_accuracy_by_generator=internal_scores,
            external_accuracy_by_subset={},
            new_generators=(
                [current_generator] if current_generator in stage_generators else []
            ),
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
        self._init_swanlab()

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)
        self.total_training_samples = self._expected_training_samples()
        self._init_internal_step_scheduler()

        logger.info(f"[1] Select a GCL method ({self.method})")
        self.setup_distributed_model()

        self._run_protocol_loop()
        self._finish_swanlab()

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
        self._init_swanlab()

        self.setup_distributed_dataset()
        self.total_samples = len(self.train_dataset)
        self.total_training_samples = self._expected_training_samples()
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
        self._finish_swanlab()

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

    def online_step(self, images, labels, idx):
        raise NotImplementedError()

    def online_before_task(self, task_id):
        raise NotImplementedError()

    def online_after_task(self, task_id):
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
        fallback_total = getattr(self, "total_samples", sample_num)
        total_training_samples = max(int(getattr(self, "total_training_samples", fallback_total)), sample_num)
        elapsed = time.time() - self.start_time
        remaining = max(total_training_samples - sample_num, 0)
        eta_seconds = int(elapsed * remaining / sample_num) if sample_num > 0 else 0
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(elapsed))} | "
            f"ETA {datetime.timedelta(seconds=eta_seconds)}"
        )
        self._log_swanlab({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "train/num_classes": len(self.exposed_classes),
            "time/elapsed_sec": int(time.time() - self.start_time),
        }, step=sample_num)

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
