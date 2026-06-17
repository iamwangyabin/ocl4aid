from typing import Sized

import torch
from torch.utils.data.distributed import DistributedSampler


class ManifestStreamSampler(DistributedSampler):
    """Sampler for a learner-blind protocol stream.

    The protocol stage metadata is used only to construct a deterministic
    stream order and evaluator checkpoints. The learner sees ordinary dataset
    items from this flattened stream, not task ids or generator names.
    """

    def __init__(
        self,
        data_source: Sized,
        stage_indices,
        num_replicas=None,
        rank=None,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.classes = self.data_source.classes
        self.targets = self.data_source.targets
        self.seed = int(seed or 0)

        if (num_replicas is None) != (rank is None):
            raise ValueError("num_replicas and rank must be provided together.")

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.stage_order = sorted(int(stage_id) for stage_id in stage_indices)
        self.stage_indices = {
            stage_id: self._interleave_stage_indices(stage_id, list(stage_indices[stage_id]))
            for stage_id in self.stage_order
        }

        self.ordered_indices = []
        self.stage_end_offsets = {}
        offset = 0
        for stage_id in self.stage_order:
            indices = self.stage_indices[stage_id]
            self.ordered_indices.extend(indices)
            offset += len(indices)
            self.stage_end_offsets[stage_id] = offset

        if self.distributed:
            self.num_samples = int(len(self.ordered_indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = self.num_samples
        else:
            self.num_samples = int(len(self.ordered_indices))
            self.total_size = self.num_samples
            self.num_selected_samples = self.num_samples

    def _interleave_stage_indices(self, stage_id: int, indices: list[int]) -> list[int]:
        if len(indices) <= 1:
            return indices

        grouped = {}
        for index in indices:
            label = int(self.targets[index])
            grouped.setdefault(label, []).append(index)

        generator = torch.Generator().manual_seed(self.seed + stage_id)
        labels = sorted(grouped)
        label_perm = torch.randperm(len(labels), generator=generator).tolist()
        labels = [labels[i] for i in label_perm]

        for label in labels:
            group = grouped[label]
            perm = torch.randperm(len(group), generator=generator).tolist()
            grouped[label] = [group[i] for i in perm]

        cursors = {label: 0 for label in labels}
        ordered = []
        while len(ordered) < len(indices):
            progressed = False
            for label in labels:
                cursor = cursors[label]
                group = grouped[label]
                if cursor >= len(group):
                    continue
                ordered.append(group[cursor])
                cursors[label] = cursor + 1
                progressed = True
            if not progressed:
                break
        return ordered

    def __iter__(self):
        if self.distributed:
            indices = self.ordered_indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        return iter(self.ordered_indices)

    def __len__(self):
        return self.num_selected_samples


class ManifestStageSampler(ManifestStreamSampler):
    """Task sampler backed by explicit CAIDBenchmark protocol stages."""

    def __init__(
        self,
        data_source: Sized,
        stage_indices,
        num_replicas=None,
        rank=None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            data_source,
            stage_indices,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )
        self.indices = self.stage_indices
        self.task = self.stage_order[0]
        self._update_task_metadata()

    def _update_task_metadata(self):
        current = self.indices[self.task]
        if self.distributed:
            self.num_samples = int(len(current) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = self.num_samples
        else:
            self.num_samples = int(len(current))
            self.total_size = self.num_samples
            self.num_selected_samples = self.num_samples

    def __iter__(self):
        current = self.indices[self.task]
        if self.distributed:
            indices = current[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        return iter(current)

    def __len__(self):
        return self.num_selected_samples

    def set_task(self, cur_iter):
        if cur_iter not in self.indices:
            raise ValueError("task out of range")
        self.task = cur_iter
        self._update_task_metadata()
