import datetime
import atexit
import json
import logging
import math
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

from datasets import CAIDBenchmarkProtocol, ConditionalJPEGCompress, OnlineIterDataset, safe_collate_drop_bad
from protocol_metrics import (
    DETECTION_METRICS,
    StageMetrics,
    compute_binary_detection_metrics,
    compute_online_metrics,
)
from utils.augment import Cutout
from utils.onlinesampler import ManifestStageSampler
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
mp.set_sharing_strategy('file_system')

DATASET_NAME = "caidbench_protocol"
CAIDBENCH_MEAN = (0.485, 0.456, 0.406)
CAIDBENCH_STD = (0.229, 0.224, 0.225)
CAIDBENCH_INPUT_SIZE = 224


_LEARNER_KWARG_DENYLIST = {
    "config",
    "caidbench_data_dir",
    "caidbench_protocol",
    "caidbench_index_path",
    "caidbench_image_column",
    "eval_interval",
    "base_stage_epochs",
    "save_base_checkpoint",
    "base_checkpoint_dir",
    "load_base_checkpoint",
    "base_checkpoint_only",
    "stage_blurry_n",
    "stage_blurry_m",
    "batchsize",
    "n_worker",
    "log_path",
    "note",
    "seeds",
    "rnd_seed",
    "use_swanlab",
    "swanlab_project",
    "swanlab_workspace",
    "swanlab_experiment_name",
    "swanlab_description",
    "swanlab_group",
    "swanlab_tags",
    "swanlab_mode",
    "swanlab_logdir",
    "swanlab_public",
}


class _Trainer():
    def __init__(self, *args, **kwargs) -> None:

        self.kwargs = kwargs
        self.__dict__.update(kwargs)

        self.start_time = time.time()
        self.eval_interval = int(getattr(self, "eval_interval", 20000) or 0)
        self.base_stage_epochs = int(getattr(self, "base_stage_epochs", 1) or 0)
        if self.base_stage_epochs < 0:
            raise ValueError(
                f"--base_stage_epochs must be non-negative, got {self.base_stage_epochs}"
            )
        self.save_base_checkpoint = bool(getattr(self, "save_base_checkpoint", False))
        self.load_base_checkpoint = getattr(self, "load_base_checkpoint", None)
        if self.load_base_checkpoint == "":
            self.load_base_checkpoint = None
        self.base_checkpoint_dir = getattr(self, "base_checkpoint_dir", None)
        self.base_checkpoint_only = bool(getattr(self, "base_checkpoint_only", False))

        # These will be fully initialized once dataset size is known.
        self.phase = "init"
        self.online_samples_seen = 0
        self._next_stream_eval_at = self.eval_interval if self.eval_interval > 0 else None
        self._base_checkpoint_loaded = False
        self._loaded_base_samples_seen = 0
        self._loaded_base_stage_metrics_payload = []
        self._swanlab = None
        self._swanlab_run = None
        self._swanlab_enabled = False
        self._swanlab_atexit_registered = False
        self._swanlab_resolved_experiment_name = None
        self._file_log_handler = None
        self.train_log_path = None

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
        self.global_batchsize = int(self.batchsize)
        if self.global_batchsize <= 0:
            raise ValueError(f"--batchsize must be positive, got {self.global_batchsize}")
        if self.distributed:
            if self.global_batchsize % self.world_size != 0:
                raise ValueError(
                    "--batchsize is the global online batch size and must be "
                    f"divisible by world_size={self.world_size}; got "
                    f"{self.global_batchsize}."
                )
            self.batchsize = self.global_batchsize // self.world_size
        self.local_batchsize = int(self.batchsize)

        run_name = self.note or self.method or "run"
        self.log_dir = os.path.join(self.log_path, run_name)

        os.makedirs(self.log_dir, exist_ok=True)

        return

    def _init_file_logging(self):
        if not self.is_main_process():
            return

        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if getattr(handler, "_ocl4aid_train_file", False):
                root_logger.removeHandler(handler)
                handler.close()

        self.train_log_path = os.path.join(self.log_dir, f"seed_{self.rnd_seed}_train.log")
        formatter = None
        if root_logger.handlers:
            formatter = root_logger.handlers[0].formatter
        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d > %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        handler = logging.FileHandler(self.train_log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler._ocl4aid_train_file = True
        root_logger.addHandler(handler)
        self._file_log_handler = handler
        logger.info("Writing training log to %s", self.train_log_path)

    def _close_file_logging(self):
        handler = getattr(self, "_file_log_handler", None)
        if handler is None:
            return
        root_logger = logging.getLogger()
        root_logger.removeHandler(handler)
        handler.close()
        self._file_log_handler = None

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
            "global_online_batchsize": self.global_batchsize,
            "local_online_batchsize": self.local_batchsize,
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
            ConditionalJPEGCompress(quality=80, recompress_if_jpeg_quality_above=80),
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
            image_column=self.caidbench_image_column,
        )

        self.protocol_stage_ids = list(self.train_dataset.active_stage_ids)
        if not self.protocol_stage_ids:
            raise ValueError("CAIDBenchmark protocol has no non-empty training stages.")
        self.protocol_stage_count = len(self.protocol_stage_ids)
        self.n_tasks = self.protocol_stage_count
        self.protocol_generator_order = self.train_dataset.generator_order

        _r = dist.get_rank() if self.distributed else None
        _w = dist.get_world_size() if self.distributed else None
        self.train_sampler = ManifestStageSampler(
            self.online_iter_dataset,
            self.train_dataset.stage_indices,
            _w,
            _r,
            seed=self.rnd_seed,
            stage_blurry_n=getattr(self, "stage_blurry_n", 100),
            stage_blurry_m=getattr(self, "stage_blurry_m", 0),
            stage_blurry_start_pos=1 if self._base_stage_enabled() else 0,
        )
        self.train_dataloader = DataLoader(
            self.online_iter_dataset,
            batch_size=self.batchsize,
            sampler=self.train_sampler,
            pin_memory=False,
            num_workers=self.n_worker,
            persistent_workers=self.n_worker > 0,
            collate_fn=safe_collate_drop_bad,
        )
        self.test_sampler = None
        self._log_protocol_stream_metadata()

        self.exposed_classes = []
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf

    def _base_stage_enabled(self):
        return self.base_stage_epochs > 0 and bool(getattr(self, "protocol_stage_ids", []))

    def _base_stage_id(self):
        if not self._base_stage_enabled():
            return None
        return self.protocol_stage_ids[0]

    def _online_stage_ids(self):
        stage_ids = list(getattr(self, "protocol_stage_ids", []))
        if self._base_stage_enabled():
            return stage_ids[1:]
        return stage_ids

    def _log_protocol_stream_metadata(self):
        stage_indices = getattr(self.train_dataset, "stage_indices", {})
        base_stage_id = self._base_stage_id()
        base_samples = 0
        if base_stage_id is not None:
            base_samples = len(stage_indices.get(base_stage_id, [])) * self.base_stage_epochs
        online_stage_ids = self._online_stage_ids()
        online_samples = sum(
            len(stage_indices.get(stage_id, []))
            for stage_id in online_stage_ids
        )
        logger.info(
            "Protocol stream | base_stage=%s | base_epochs=%s | online_stages=%s | learner labels: binary | task slots: %s | base samples: %s | online samples: %s | temporal blurry n=%s m=%s",
            base_stage_id,
            self.base_stage_epochs if base_stage_id is not None else 0,
            len(online_stage_ids),
            self.protocol_stage_count,
            base_samples,
            online_samples,
            getattr(self, "stage_blurry_n", 100),
            getattr(self, "stage_blurry_m", 0),
        )

    def _skip_empty_batch(self, batch, context):
        local_empty = batch is None
        if self.distributed:
            empty_flag = torch.tensor(
                1 if local_empty else 0,
                dtype=torch.int32,
                device=self.device,
            )
            dist.all_reduce(empty_flag, op=dist.ReduceOp.MAX)
            if int(empty_flag.item()) > 0:
                logger.warning(
                    "Skipping %s batch because at least one distributed rank dropped all unreadable samples.",
                    context,
                )
                return True
        elif local_empty:
            logger.warning("Skipping empty %s batch after dropping unreadable samples.", context)
            return True
        return False

    def setup_distributed_model(self):

        logger.info(f"Building model: {self.method}")
        logger.info(
            "Learner-visible setup | num_classes=%s | task_slots=%s | protocol_stages=%s",
            self.n_classes,
            self.n_tasks,
            getattr(self, "protocol_stage_count", None),
        )
        self.model = select_model(self.method, self.backbone, self.n_classes, self.n_tasks, self._learner_kwargs()).to(self.device)
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
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self._scheduler_hparams())
        logger.info(
            "Optimizer/Scheduler | optimizer=%s | lr=%s | scheduler=%s | t_max=%s",
            self.opt_name,
            self.lr,
            self.sched_name,
            self._scheduler_step_budget(),
        )

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        logger.info(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        learnables = [n for n, p in self.model_without_ddp.named_parameters() if p.requires_grad]
        logger.info(f"Learnable Parameters :\t{n_params}")
        logger.info(learnables)
        logger.info("")

    def _learner_kwargs(self):
        return {
            key: value
            for key, value in self.kwargs.items()
            if key not in _LEARNER_KWARG_DENYLIST
        }

    def run(self):
        if self.profile:
            self.profile_worker(0)
        else:
            # Distributed Launch
            if self.ngpus_per_nodes > 1:
                mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
            else:
                self.main_worker(0)

    def _protocol_eval_average(self, stage_metric: StageMetrics, metric_name="accuracy"):
        values = [
            metrics.get(metric_name)
            for metrics in stage_metric.internal_metrics_by_generator.values()
            if metrics.get(metric_name) is not None
        ]
        if not values:
            return 0.0 if metric_name in {"accuracy", "f1"} else None
        return sum(values) / len(values)

    def _protocol_metric_payload(self, stage_metric: StageMetrics):
        return {
            "stage_id": stage_metric.stage_id,
            "new_generators": stage_metric.new_generators,
            "internal_metrics_by_generator": stage_metric.internal_metrics_by_generator,
            "external_metrics_by_subset": stage_metric.external_metrics_by_subset,
            "internal_accuracy_by_generator": stage_metric.internal_accuracy_by_generator,
            "external_accuracy_by_subset": stage_metric.external_accuracy_by_subset,
        }

    def _stage_metric_from_payload(self, payload):
        return StageMetrics(
            stage_id=int(payload["stage_id"]),
            internal_metrics_by_generator=dict(
                payload.get("internal_metrics_by_generator", {})
            ),
            external_metrics_by_subset=dict(
                payload.get("external_metrics_by_subset", {})
            ),
            new_generators=list(payload.get("new_generators", [])),
        )

    def _base_checkpoint_directory(self):
        configured = getattr(self, "base_checkpoint_dir", None)
        if configured:
            return os.path.abspath(os.path.expanduser(configured))
        return os.path.abspath(
            os.path.join(getattr(self, "log_path", "run_logs"), "base_checkpoints")
        )

    def _default_base_checkpoint_path(self):
        base_stage_id = self._base_stage_id()
        if base_stage_id is None:
            raise ValueError(
                "Base checkpointing requires --base_stage_epochs > 0 and a non-empty protocol."
            )
        protocol_name = os.path.splitext(os.path.basename(self.caidbench_protocol))[0]
        filename = (
            f"base_{self._metric_slug(self.method)}"
            f"_{self._metric_slug(self.backbone)}"
            f"_{self._metric_slug(protocol_name)}"
            f"_seed{self.rnd_seed}"
            f"_stage{base_stage_id}"
            f"_epochs{self.base_stage_epochs}.pt"
        )
        return os.path.join(self._base_checkpoint_directory(), filename)

    def _resolve_load_base_checkpoint_path(self):
        configured = getattr(self, "load_base_checkpoint", None)
        if not configured:
            return None
        configured = str(configured)
        if configured.lower() == "auto":
            return self._default_base_checkpoint_path()
        return os.path.abspath(os.path.expanduser(configured))

    def _collect_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, rng_state):
        if not isinstance(rng_state, dict):
            return
        try:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"].cpu())
        except Exception as e:
            logger.warning("Failed to restore CPU RNG state from base checkpoint: %s", e)

        cuda_states = rng_state.get("cuda_all")
        if not torch.cuda.is_available() or not cuda_states:
            return
        try:
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
            else:
                local_index = int(getattr(self, "gpu", 0) or 0)
                cuda_state = cuda_states[min(local_index, len(cuda_states) - 1)].cpu()
                torch.cuda.set_rng_state(cuda_state, device=self.device)
        except Exception as e:
            logger.warning("Failed to restore CUDA RNG state from base checkpoint: %s", e)

    def _torch_load_checkpoint(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _protocol_generator_names(self):
        return [
            entry["generator_name"]
            for entry in getattr(self, "protocol_generator_order", [])
        ]

    def _validate_base_checkpoint(self, checkpoint, path):
        metadata = checkpoint.get("metadata", {})
        if metadata.get("format_version") != 1:
            raise ValueError(
                f"Unsupported base checkpoint format in {path}: "
                f"{metadata.get('format_version')!r}"
            )

        checks = [
            ("method", metadata.get("method"), self.method),
            ("backbone", metadata.get("backbone"), self.backbone),
            ("n_classes", metadata.get("n_classes"), self.n_classes),
            ("n_tasks", metadata.get("n_tasks"), self.n_tasks),
            ("base_stage_id", metadata.get("base_stage_id"), self._base_stage_id()),
            (
                "base_stage_epochs",
                metadata.get("base_stage_epochs"),
                self.base_stage_epochs,
            ),
        ]
        if self.rnd_seed is not None:
            checks.append(("rnd_seed", metadata.get("rnd_seed"), self.rnd_seed))

        mismatches = []
        for name, saved, expected in checks:
            if saved is None:
                mismatches.append(f"{name}: missing != {expected!r}")
            elif saved != expected:
                mismatches.append(f"{name}: {saved!r} != {expected!r}")

        saved_generators = metadata.get("protocol_generators")
        current_generators = self._protocol_generator_names()
        if saved_generators != current_generators:
            mismatches.append("protocol generator order differs")

        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(f"Base checkpoint {path} does not match this run: {details}")

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _save_base_checkpoint(self, stage_metrics, samples_cnt):
        if not self.save_base_checkpoint:
            return None

        output_path = self._default_base_checkpoint_path()
        if self.is_main_process():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            checkpoint = {
                "metadata": {
                    "format_version": 1,
                    "created_at": datetime.datetime.now().isoformat(),
                    "method": self.method,
                    "backbone": self.backbone,
                    "rnd_seed": self.rnd_seed,
                    "caidbench_protocol": self.caidbench_protocol,
                    "protocol_generators": self._protocol_generator_names(),
                    "base_stage_id": self._base_stage_id(),
                    "base_stage_epochs": self.base_stage_epochs,
                    "n_classes": self.n_classes,
                    "n_tasks": self.n_tasks,
                    "global_batchsize": self.global_batchsize,
                    "input_size": self.inp_size,
                },
                "model_state": self.model_without_ddp.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "model_attrs": {
                    "task_count": getattr(self.model_without_ddp, "task_count", None),
                },
                "trainer_state": {
                    "task_id": getattr(self, "task_id", None),
                    "exposed_classes": list(self.exposed_classes),
                    "mask": self.mask.detach().cpu(),
                    "phase": self.phase,
                    "samples_cnt": int(samples_cnt),
                    "online_samples_seen": int(self.online_samples_seen),
                    "next_stream_eval_at": self._next_stream_eval_at,
                },
                "stage_metrics": [
                    self._protocol_metric_payload(item)
                    for item in stage_metrics
                ],
                "rng_state": self._collect_rng_state(),
            }
            torch.save(checkpoint, output_path)
            logger.info("Saved reusable base-stage checkpoint to %s", output_path)

        if self.distributed:
            dist.barrier()
        return output_path

    def _load_base_checkpoint_if_requested(self):
        path = self._resolve_load_base_checkpoint_path()
        if path is None:
            return
        if self._base_stage_id() is None:
            raise ValueError("--load_base_checkpoint requires --base_stage_epochs > 0.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Base checkpoint not found: {path}")

        checkpoint = self._torch_load_checkpoint(path)
        self._validate_base_checkpoint(checkpoint, path)

        self.model_without_ddp.load_state_dict(checkpoint["model_state"])
        model_task_count = checkpoint.get("model_attrs", {}).get("task_count")
        if model_task_count is not None and hasattr(self.model_without_ddp, "task_count"):
            self.model_without_ddp.task_count = int(model_task_count)

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._move_optimizer_state_to_device()
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

        trainer_state = checkpoint.get("trainer_state", {})
        if "task_id" in trainer_state and trainer_state["task_id"] is not None:
            self.task_id = int(trainer_state["task_id"])
        self.exposed_classes = [
            int(item)
            for item in trainer_state.get("exposed_classes", self.exposed_classes)
        ]
        mask = trainer_state.get("mask")
        if torch.is_tensor(mask):
            self.mask = mask.to(self.device)
        self.online_samples_seen = int(trainer_state.get("online_samples_seen", 0))
        next_stream_eval_at = trainer_state.get(
            "next_stream_eval_at",
            self._next_stream_eval_at,
        )
        self._next_stream_eval_at = (
            None if next_stream_eval_at is None else int(next_stream_eval_at)
        )
        self._loaded_base_samples_seen = int(trainer_state.get("samples_cnt", 0))
        self._loaded_base_stage_metrics_payload = list(
            checkpoint.get("stage_metrics", [])
        )
        self._restore_rng_state(checkpoint.get("rng_state"))
        self._base_checkpoint_loaded = True
        self.phase = "base_loaded"
        logger.info(
            "Loaded reusable base-stage checkpoint from %s | samples=%s | task_id=%s | model_task_count=%s",
            path,
            self._loaded_base_samples_seen,
            getattr(self, "task_id", None),
            getattr(self.model_without_ddp, "task_count", None),
        )
        if self.distributed:
            dist.barrier()

    def _format_metric_value(self, value):
        return "n/a" if value is None else f"{value:.4f}"

    def _log_protocol_eval(self, stage_metric: StageMetrics, stage_name: str, *, stream_sample=None):
        internal_avg = {
            metric_name: self._protocol_eval_average(stage_metric, metric_name)
            for metric_name in DETECTION_METRICS
        }
        current_metrics = stage_metric.internal_metrics_by_generator.get(stage_name, {})
        current_by_metric = {
            metric_name: current_metrics.get(metric_name)
            for metric_name in DETECTION_METRICS
        }
        if stream_sample is None:
            logger.info(
                "Protocol Eval | stage %s | avg acc %s | f1 %s | ap %s | auc %s | current acc %s | f1 %s | ap %s | auc %s",
                stage_metric.stage_id,
                self._format_metric_value(internal_avg["accuracy"]),
                self._format_metric_value(internal_avg["f1"]),
                self._format_metric_value(internal_avg["ap"]),
                self._format_metric_value(internal_avg["auc"]),
                self._format_metric_value(current_by_metric["accuracy"]),
                self._format_metric_value(current_by_metric["f1"]),
                self._format_metric_value(current_by_metric["ap"]),
                self._format_metric_value(current_by_metric["auc"]),
            )
            prefix = "protocol"
            step = stage_metric.stage_id
        else:
            logger.info(
                "Protocol Stream Eval | online_sample %s | stage %s | avg acc %s | f1 %s | ap %s | auc %s | current acc %s | f1 %s | ap %s | auc %s",
                stream_sample,
                stage_metric.stage_id,
                self._format_metric_value(internal_avg["accuracy"]),
                self._format_metric_value(internal_avg["f1"]),
                self._format_metric_value(internal_avg["ap"]),
                self._format_metric_value(internal_avg["auc"]),
                self._format_metric_value(current_by_metric["accuracy"]),
                self._format_metric_value(current_by_metric["f1"]),
                self._format_metric_value(current_by_metric["ap"]),
                self._format_metric_value(current_by_metric["auc"]),
            )
            prefix = "protocol_stream"
            step = stream_sample

        swanlab_metrics = {
            f"{prefix}/stage": stage_metric.stage_id,
            f"{prefix}/internal_avg_acc": internal_avg["accuracy"],
            f"{prefix}/current_generator_acc": current_by_metric["accuracy"],
        }
        for metric_name, score in internal_avg.items():
            swanlab_metrics[f"{prefix}/internal_avg_{metric_name}"] = score
        for metric_name, score in current_by_metric.items():
            swanlab_metrics[f"{prefix}/current_generator_{metric_name}"] = score
        for generator_name, generator_metrics in stage_metric.internal_metrics_by_generator.items():
            generator_slug = self._metric_slug(generator_name)
            swanlab_metrics[f"{prefix}/internal/{generator_slug}"] = generator_metrics.get("accuracy")
            for metric_name, score in generator_metrics.items():
                swanlab_metrics[f"{prefix}/internal/{generator_slug}/{metric_name}"] = score
        self._log_swanlab(swanlab_metrics, step=step)

    def _stage_id_for_seen_samples(self, sample_count: int) -> int:
        stage_ids = self._online_stage_ids()
        if not stage_ids:
            return self.protocol_stage_ids[-1]
        offset = 0
        sampler_indices = getattr(self.train_sampler, "indices", {})
        for stage_id in stage_ids:
            offset += len(sampler_indices.get(stage_id, []))
            if sample_count <= offset:
                return stage_id
        return stage_ids[-1]

    def _maybe_run_stream_eval(self, stream_metrics):
        if self.eval_interval <= 0 or self._next_stream_eval_at is None:
            return

        while self.online_samples_seen >= self._next_stream_eval_at:
            stream_sample = self._next_stream_eval_at
            eval_stage_id = self._stage_id_for_seen_samples(stream_sample)
            if self.distributed:
                dist.barrier()
            if self.is_main_process():
                stage_name = self.protocol_generator_order[eval_stage_id]["generator_name"]
                stage_metric = self._evaluate_protocol_stage(eval_stage_id)
                self._log_protocol_eval(stage_metric, stage_name, stream_sample=stream_sample)
                stream_payload = self._protocol_metric_payload(stage_metric)
                stream_payload["online_sample"] = stream_sample
                stream_metrics.append(stream_payload)
            if self.distributed:
                dist.barrier()
            self._next_stream_eval_at += self.eval_interval

    def _maybe_report_training(self, samples_cnt, loss, acc, next_report_at, report_period):
        if samples_cnt >= next_report_at:
            self.report_training(samples_cnt, loss, acc)
            while next_report_at <= samples_cnt:
                next_report_at += report_period
        return next_report_at

    def _evaluate_and_log_stage(self, stage_id, stage_metrics):
        if self.distributed:
            dist.barrier()
        if self.is_main_process():
            stage_name = self.protocol_generator_order[stage_id]["generator_name"]
            stage_metric = self._evaluate_protocol_stage(stage_id)
            stage_metrics.append(stage_metric)
            self._log_protocol_eval(stage_metric, stage_name)
        if self.distributed:
            dist.barrier()

    def _run_base_stage(self, stage_metrics, samples_cnt, num_report, report_period):
        base_stage_id = self._base_stage_id()
        if base_stage_id is None:
            return samples_cnt, num_report

        stage_name = self.protocol_generator_order[base_stage_id]["generator_name"]
        logger.info("\n")
        logger.info("#" * 50)
        logger.info(
            "# Base Stage: %s | supervised epochs %s",
            stage_name,
            self.base_stage_epochs,
        )
        logger.info("#" * 50 + "\n")

        self.phase = "base"
        self.train_sampler.set_task(base_stage_id)
        self.online_before_task(base_stage_id)

        for epoch in range(self.base_stage_epochs):
            logger.info("Base epoch %s/%s", epoch + 1, self.base_stage_epochs)
            self.train_sampler.set_epoch(epoch)
            for batch in self.train_dataloader:
                if self._skip_empty_batch(batch, "base"):
                    continue
                images, labels, _idx = batch
                samples_cnt += images.size(0) * self.world_size
                loss, acc = self.online_step(images, labels, None)
                num_report = self._maybe_report_training(
                    samples_cnt,
                    loss,
                    acc,
                    num_report,
                    report_period,
                )
                sys.stdout.flush()

        self.after_base_stage_train(base_stage_id)
        self._evaluate_and_log_stage(base_stage_id, stage_metrics)
        self.online_after_task(base_stage_id)
        self._save_base_checkpoint(stage_metrics, samples_cnt)
        return samples_cnt, num_report

    def _run_protocol_loop(self):
        if self.save_base_checkpoint and self._base_stage_id() is None:
            raise ValueError("--save_base_checkpoint requires --base_stage_epochs > 0.")

        logger.info(
            "[2] Base stage training followed by online continual learning (%s stages, binary labels)",
            self.protocol_stage_count,
        )
        samples_cnt = 0
        online_samples_cnt = 0
        num_report = 2000
        report_period = 500
        stage_metrics = [
            self._stage_metric_from_payload(payload)
            for payload in self._loaded_base_stage_metrics_payload
        ]
        stream_metrics = []
        samples_cnt = int(self._loaded_base_samples_seen)
        while num_report <= samples_cnt:
            num_report += report_period

        if self._base_checkpoint_loaded:
            logger.info(
                "Skipping supervised base stage because a reusable base checkpoint was loaded."
            )
        else:
            samples_cnt, num_report = self._run_base_stage(
                stage_metrics,
                samples_cnt,
                num_report,
                report_period,
            )

        if self.base_checkpoint_only:
            logger.info("Base checkpoint-only mode requested; stopping before online stages.")
            self.phase = "done"
            self._save_protocol_summary(stage_metrics, stream_metrics)
            return

        self.phase = "stream"
        online_stage_ids = self._online_stage_ids()
        if not online_stage_ids:
            logger.info("No online continual stages remain after base stage.")

        for task_pos, stage_id in enumerate(online_stage_ids):
            stage_name = self.protocol_generator_order[stage_id]["generator_name"]
            logger.info("\n")
            logger.info("#" * 50)
            logger.info(
                "# Online Stage %s/%s: %s",
                task_pos + 1,
                len(online_stage_ids),
                stage_name,
            )
            logger.info("#" * 50 + "\n")

            self.train_sampler.set_task(stage_id)
            self.online_before_task(stage_id)

            for batch in self.train_dataloader:
                if self._skip_empty_batch(batch, f"online_stage_{stage_id}"):
                    continue
                images, labels, _idx = batch
                batch_size_global = images.size(0) * self.world_size
                online_samples_cnt += batch_size_global
                samples_cnt += batch_size_global
                self.online_samples_seen = online_samples_cnt

                # Framework stages are task boundaries, but the learner still
                # receives only images and binary labels.
                loss, acc = self.online_step(images, labels, None)
                num_report = self._maybe_report_training(
                    samples_cnt,
                    loss,
                    acc,
                    num_report,
                    report_period,
                )

                self._maybe_run_stream_eval(stream_metrics)
                sys.stdout.flush()

            self._evaluate_and_log_stage(stage_id, stage_metrics)
            self.online_after_task(stage_id)

        self.phase = "done"
        self._save_protocol_summary(stage_metrics, stream_metrics)

    def _save_protocol_summary(self, stage_metrics, stream_metrics):
        if not self.is_main_process():
            return
        metrics = compute_online_metrics(stage_metrics)
        summary = {
            "stage_metrics": [
                self._protocol_metric_payload(item)
                for item in stage_metrics
            ],
            "stream_metrics": stream_metrics,
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
        sampler_indices = getattr(getattr(self, "train_sampler", None), "indices", None)
        if sampler_indices is None:
            sampler_indices = getattr(self.train_dataset, "stage_indices", {})
        if not self.protocol_stage_ids:
            return sum(len(indices) for indices in sampler_indices.values())

        total = 0
        base_stage_id = self._base_stage_id()
        if base_stage_id is not None:
            total += len(sampler_indices.get(base_stage_id, [])) * self.base_stage_epochs
        total += sum(
            len(sampler_indices.get(stage_id, []))
            for stage_id in self._online_stage_ids()
        )
        return total

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
            internal_metrics_by_generator=internal_scores,
            external_metrics_by_subset={},
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
            collate_fn=safe_collate_drop_bad,
        )
        binary_predictions = []
        binary_target_values = []
        fake_scores = []
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    logger.warning("Skipping empty protocol eval batch after dropping unreadable samples.")
                    continue
                images, _targets, binary_targets = batch
                images = images.to(self.device)
                logits = self._protocol_eval_logits(images)
                pred_indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
                batch_fake_scores = self._protocol_fake_scores(logits).detach().cpu().tolist()
                binary_targets = [int(item) for item in binary_targets.tolist()]
                for pred_index in pred_indices:
                    original_class = self.exposed_classes[pred_index]
                    binary_predictions.append(0 if original_class == 0 else 1)
                binary_target_values.extend(binary_targets)
                fake_scores.extend(batch_fake_scores)
        return compute_binary_detection_metrics(
            binary_target_values,
            binary_predictions,
            fake_scores,
        )

    def _protocol_fake_scores(self, logits):
        probabilities = torch.softmax(logits, dim=-1)
        fake_class_mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
        for logit_index, original_class in enumerate(self.exposed_classes[: logits.size(-1)]):
            if original_class != 0:
                fake_class_mask[logit_index] = True
        if not torch.any(fake_class_mask):
            return torch.zeros(logits.size(0), dtype=probabilities.dtype, device=logits.device)
        return probabilities[:, fake_class_mask].sum(dim=-1)

    def _protocol_eval_logits(self, images):
        if self.method == "flyprompt":
            logit_raw = self.model_without_ddp.forward_with_rp(images)
            expert_count = min(
                int(getattr(self.model_without_ddp, "task_count", 0)) + 1,
                logit_raw.size(1),
            )
            logit_raw = logit_raw[:, :expert_count]
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

        if self.method in {"l2p", "dualprompt", "mvp", "ranpac", "singleprompt", "slca", "sdlora", "rineside_gauss"}:
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
        self._init_file_logging()
        try:
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

            logger.info(f"[1] Select a GCL method ({self.method})")
            self.setup_distributed_model()
            self._load_base_checkpoint_if_requested()

            self._run_protocol_loop()
        finally:
            self._finish_swanlab()
            self._close_file_logging()

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
        self._init_file_logging()
        try:
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

            self.setup_distributed_model()

            samples_cnt = 0
            for i, batch in enumerate(self.train_dataloader):
                if self._skip_empty_batch(batch, "smoke"):
                    continue
                images, labels, idx = batch
                samples_cnt += images.size(0) * self.world_size
                loss, acc = self.online_step(images, labels, None)
                self.report_training(samples_cnt, loss, acc)
                break
        finally:
            self._finish_swanlab()
            self._close_file_logging()

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

    def after_base_stage_train(self, base_stage_id):
        return None

    def _advance_model_task_count(self):
        model_obj = self.model.module if self.distributed else self.model
        if hasattr(model_obj, "process_task_count"):
            model_obj.process_task_count()

    @torch.no_grad()
    def _collect_rp_features_for_task_slot(self, images, labels):
        """Collect backbone CLS features for RPFC task-slot gating."""
        model_obj = self.model_without_ddp
        use_rp_gate = getattr(model_obj, "use_rp_gate", False)
        rp_head = getattr(model_obj, "rp_head", None)
        if not use_rp_gate or rp_head is None:
            return

        images = images.to(self.device, non_blocking=True)
        images = self.test_transform_tensor(images)

        model_obj.backbone.eval()
        if hasattr(model_obj.backbone, "forward_features"):
            feats = model_obj.backbone.forward_features(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            cls_feat = feats[:, 0]
        else:
            x = model_obj.backbone.patch_embed(images)
            batch_size = x.size(0)
            cls_token = model_obj.backbone.cls_token.expand(batch_size, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = model_obj.backbone.pos_drop(token_appended + model_obj.backbone.pos_embed)
            x = model_obj.backbone.blocks(x)
            x = model_obj.backbone.norm(x)
            cls_feat = x[:, 0]

        session_id = int(getattr(model_obj, "task_count", getattr(self, "task_id", 0)))
        session_labels = torch.full(
            (labels.size(0),),
            session_id,
            device=self.device,
            dtype=torch.long,
        )
        rp_head.collect(cls_feat, session_labels)

    def online_before_task(self, task_id):
        del task_id

    def online_after_task(self, task_id):
        del task_id

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self._scheduler_hparams())
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def _estimated_optimizer_steps(self, sample_count):
        if sample_count <= 0:
            return 1
        updates_per_batch = max(1, int(float(getattr(self, "online_iter", 1) or 1)))
        return max(1, math.ceil(sample_count / max(1, self.global_batchsize)) * updates_per_batch)

    def _scheduler_step_budget(self):
        configured = int(getattr(self, "scheduler_t_max", 0) or 0)
        if configured > 0:
            return configured
        sample_count = int(getattr(self, "total_training_samples", 0) or 0)
        if sample_count <= 0:
            sample_count = int(getattr(self, "total_samples", 0) or 0)
        return self._estimated_optimizer_steps(sample_count)

    def _scheduler_hparams(self):
        return {
            "gamma": self.lr_gamma,
            "t_max": self._scheduler_step_budget(),
            "eta_min": float(getattr(self, "scheduler_eta_min", 0.0) or 0.0),
        }

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
