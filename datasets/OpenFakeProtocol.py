import json
import os
from bisect import bisect_right
from io import BytesIO
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
        self._parquet_store = _ParquetImageStore()

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
        self.active_stage_ids = []
        self.stage_generators = {}
        if train:
            for stage_id, stage_info in payload["train_by_stage"].items():
                stage_id_int = int(stage_id)
                self.stage_indices[stage_id_int] = [
                    self.record_id_to_index[record_id]
                    for record_id in stage_info["sample_ids"]
                ]
                self.stage_generators[stage_id_int] = list(stage_info["generators"])
                if self.stage_indices[stage_id_int]:
                    self.active_stage_ids.append(stage_id_int)
            self.active_stage_ids.sort()

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
        image = self._load_image(record)
        if self.transform is not None:
            image = self.transform(image)
        target = record["class_id"] if record["class_id"] >= 0 else 0
        return image, target

    def _load_image(self, record):
        parquet_path = record.get("parquet_path")
        parquet_row_index = record.get("parquet_row_index")
        if parquet_path and parquet_row_index is not None:
            payload = self._parquet_store.read_image(
                parquet_path,
                int(parquet_row_index),
                record.get("parquet_image_column") or "image",
            )
            return _image_from_payload(payload)
        return Image.open(record["resolved_path"]).convert("RGB")

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    def make_eval_subset(self, indices):
        return _ProtocolEvalSubset(self, indices)


class _ParquetImageStore:
    def __init__(self):
        self._files = {}
        self._row_group_starts = {}
        self._last_key = None
        self._last_column = None

    def read_image(self, parquet_path, row_index, image_column):
        parquet_path = str(Path(parquet_path).expanduser())
        parquet = self._get_file(parquet_path)
        starts = self._get_row_group_starts(parquet_path, parquet)
        row_group_index = bisect_right(starts, row_index) - 1
        if row_group_index < 0:
            raise IndexError(f"Row {row_index} is out of range for {parquet_path}")

        row_group_start = starts[row_group_index]
        row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
        if row_index >= row_group_start + row_group_rows:
            raise IndexError(f"Row {row_index} is out of range for {parquet_path}")

        key = (parquet_path, row_group_index, image_column)
        if key != self._last_key:
            self._last_column = parquet.read_row_group(row_group_index, columns=[image_column]).column(0)
            self._last_key = key
        return self._last_column[row_index - row_group_start].as_py()

    def _get_file(self, parquet_path):
        parquet = self._files.get(parquet_path)
        if parquet is None:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(parquet_path)
            self._files[parquet_path] = parquet
        return parquet

    def _get_row_group_starts(self, parquet_path, parquet):
        starts = self._row_group_starts.get(parquet_path)
        if starts is not None:
            return starts
        starts = []
        offset = 0
        for row_group_index in range(parquet.metadata.num_row_groups):
            starts.append(offset)
            offset += parquet.metadata.row_group(row_group_index).num_rows
        self._row_group_starts[parquet_path] = starts
        return starts


def _image_from_payload(image_payload):
    image_bytes = None
    if isinstance(image_payload, dict):
        image_bytes = image_payload.get("bytes")
    elif isinstance(image_payload, (bytes, bytearray)):
        image_bytes = image_payload

    if image_bytes is None:
        raise ValueError("Image payload does not contain bytes.")

    return Image.open(BytesIO(image_bytes)).convert("RGB")
