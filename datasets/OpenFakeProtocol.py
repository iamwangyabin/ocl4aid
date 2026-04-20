import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class _ProtocolEvalSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        image, target = self.base_dataset[base_index]
        record = self.base_dataset.records[base_index]
        binary_target = 0 if record["binary_label"] == "real" else 1
        return image, target, binary_target


class OpenFakeProtocol(Dataset):
    """Manifest-backed dataset for the OpenFake + AIGIBench protocol."""

    def __init__(
        self,
        root,
        train=True,
        download=False,
        transform=None,
        protocol_manifest=None,
    ):
        super().__init__()
        if protocol_manifest is None:
            raise ValueError("protocol_manifest is required for openfake_protocol")
        del download

        self.root = root
        self.train = train
        self.transform = transform

        payload = json.loads(Path(protocol_manifest).read_text(encoding="utf-8"))
        self.generator_order = payload["generator_order"]
        self.label_space = payload["label_space"]
        self.classes = list(range(len(self.label_space)))

        record_map = {record["record_id"]: record for record in payload["records"]}
        if train:
            selected_ids = []
            for stage_id in sorted(payload["train_by_stage"], key=int):
                selected_ids.extend(payload["train_by_stage"][str(stage_id)]["sample_ids"])
        else:
            selected_ids = []
            for test_group in (payload["internal_tests"], payload["external_tests"]):
                for test_slice in test_group.values():
                    selected_ids.extend(test_slice["sample_ids"])
            selected_ids = list(dict.fromkeys(selected_ids))

        self.records = []
        self.record_id_to_index = {}
        self.targets = []
        for record_id in selected_ids:
            record = dict(record_map[record_id])
            record["resolved_path"] = self._resolve_path(record["path"])
            self.record_id_to_index[record_id] = len(self.records)
            self.records.append(record)
            class_id = record["class_id"]
            self.targets.append(class_id if class_id >= 0 else 0)

        self.stage_indices = {}
        if train:
            for stage_id, stage_info in payload["train_by_stage"].items():
                self.stage_indices[int(stage_id)] = [
                    self.record_id_to_index[record_id]
                    for record_id in stage_info["sample_ids"]
                ]

        self.internal_slices = {}
        self.external_slices = {}
        if not train:
            for name, test_slice in payload["internal_tests"].items():
                self.internal_slices[name] = [
                    self.record_id_to_index[record_id]
                    for record_id in test_slice["sample_ids"]
                ]
            for name, test_slice in payload["external_tests"].items():
                self.external_slices[name] = [
                    self.record_id_to_index[record_id]
                    for record_id in test_slice["sample_ids"]
                ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["resolved_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = record["class_id"] if record["class_id"] >= 0 else 0
        return image, target

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    def make_eval_subset(self, indices):
        return _ProtocolEvalSubset(self, indices)
