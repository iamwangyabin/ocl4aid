from __future__ import annotations

from contextlib import suppress
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .image_quality import as_rgb_preserve_jpeg_metadata
from .safe_sample import make_bad_sample


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROTOCOL = _REPO_ROOT / "protocol_presets" / "caidbench" / "model_appearance_order_protocol.yaml"
_REQUIRED_INDEX_COLUMNS = {
    "task_id",
    "generator_name",
    "split",
    "label",
    "arrow_path",
    "batch_id",
    "row_in_batch",
}


class _ProtocolEvalSubset(Dataset):
    def __init__(self, base_dataset: "CAIDBenchmarkProtocol", indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        try:
            image, target = self.base_dataset[base_index]
        except Exception as exc:
            return make_bad_sample(base_index, exc)
        return image, target, self.base_dataset.binary_targets[base_index]


class _ArrowImageStore:
    def __init__(self, root: str | Path, image_column: str) -> None:
        self.root = Path(root).expanduser()
        self.image_column = image_column
        self._readers: dict[str, tuple[Any, Any]] = {}

    def close(self) -> None:
        for source, reader in self._readers.values():
            reader_close = getattr(reader, "close", None)
            if callable(reader_close):
                with suppress(Exception):
                    reader_close()
            source_close = getattr(source, "close", None)
            if callable(source_close):
                source_close()
        self._readers.clear()

    def read(self, arrow_path: str, batch_id: int, row_in_batch: int):
        reader = self._reader(arrow_path)
        batch = reader.get_batch(int(batch_id))
        if self.image_column not in batch.schema.names:
            raise ValueError(f"Missing image column {self.image_column!r} in {arrow_path}")
        return batch.column(self.image_column)[int(row_in_batch)].as_py()

    def _reader(self, arrow_path: str):
        full_path = self._resolve_arrow_path(arrow_path)
        key = str(full_path)
        cached = self._readers.get(key)
        if cached is not None:
            return cached[1]

        import pyarrow as pa
        import pyarrow.ipc as ipc

        source = pa.memory_map(key, "r")
        try:
            reader = ipc.open_file(source)
        except pa.ArrowInvalid as exc:
            source.close()
            raise ValueError(f"CAIDBenchmark requires Arrow IPC file format: {key}") from exc
        self._readers[key] = (source, reader)
        return reader

    def _resolve_arrow_path(self, arrow_path: str) -> Path:
        path = Path(str(arrow_path)).expanduser()
        if path.is_absolute():
            return path
        return self.root / path


class CAIDBenchmarkProtocol(Dataset):
    """CAIDBenchmark Arrow/index-backed online continual dataset.

    The protocol YAML controls online task order. The paired parquet index
    selects exact Arrow rows for each task and split.
    """

    def __init__(
        self,
        root,
        train=True,
        download=False,
        transform=None,
        protocol_path=None,
        index_path=None,
        image_column="image",
    ):
        super().__init__()
        del download

        if root is None:
            raise ValueError("--caidbench_data_dir is required for caidbench_protocol")
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"CAIDBenchmark data directory does not exist: {self.root}")

        self.train = bool(train)
        self.transform = transform
        self.image_column = str(image_column)
        self.protocol_path = _resolve_protocol_path(protocol_path)
        protocol = _load_protocol(self.protocol_path)
        self.protocol_tasks = _protocol_tasks(protocol)
        self.index_path = _resolve_index_path(index_path, protocol, self.protocol_path)

        index = _load_index(self.index_path)
        self.generator_order = []
        self._task_raw_ids: dict[int, list[int]] = {}
        for online_stage_id, task in enumerate(self.protocol_tasks):
            raw_ids = _task_raw_task_ids(task)
            name = str(task.get("name", task.get("id", f"task{online_stage_id}")))
            self._task_raw_ids[online_stage_id] = raw_ids
            self.generator_order.append(
                {
                    "stage_id": online_stage_id,
                    "generator_name": name,
                    "protocol_id": str(task.get("id", name)),
                    "raw_task_ids": raw_ids,
                }
            )

        split = "train" if self.train else "test"
        self.metadata = self._select_split(index, split)
        self._init_fast_columns()

        self.classes = [0, 1]
        self.label_space = {"real": 0, "fake": 1}

        self.stage_indices: dict[int, list[int]] = {}
        self.active_stage_ids: list[int] = []
        self.stage_generators: dict[int, list[str]] = {}
        self.internal_slices: dict[str, list[int]] = {}

        if self.train:
            for stage_id, group in self.metadata.groupby("_online_stage_id", sort=True):
                stage_id = int(stage_id)
                indices = [int(i) for i in group.index.tolist()]
                self.stage_indices[stage_id] = indices
                if indices:
                    self.active_stage_ids.append(stage_id)
                self.stage_generators[stage_id] = [self.generator_order[stage_id]["generator_name"]]
        else:
            for stage_id, group in self.metadata.groupby("_online_stage_id", sort=True):
                name = self.generator_order[int(stage_id)]["generator_name"]
                self.internal_slices[name] = [int(i) for i in group.index.tolist()]

        self._arrow_store: _ArrowImageStore | None = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self._load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __getstate__(self):
        state = self.__dict__.copy()
        store = state.get("_arrow_store")
        if store is not None:
            store.close()
        state["_arrow_store"] = None
        return state

    def close(self):
        if self._arrow_store is not None:
            self._arrow_store.close()
            self._arrow_store = None

    def make_eval_subset(self, indices):
        return _ProtocolEvalSubset(self, indices)

    def _select_split(self, index: pd.DataFrame, split: str) -> pd.DataFrame:
        frames = []
        split_index = index[index["split"] == split]
        grouped_by_task = {
            int(task_id): group
            for task_id, group in split_index.groupby("task_id", sort=False)
        }
        for online_stage_id in range(len(self.generator_order)):
            raw_ids = self._task_raw_ids[online_stage_id]
            raw_frames = [grouped_by_task[raw_id] for raw_id in raw_ids if raw_id in grouped_by_task]
            if not raw_frames:
                continue
            frame = pd.concat(raw_frames, axis=0, ignore_index=False).copy()
            if frame.empty:
                continue
            frame["_online_stage_id"] = online_stage_id
            frame["_online_task_name"] = self.generator_order[online_stage_id]["generator_name"]
            frame["_target"] = frame["label"].astype("int64")
            frames.append(frame)
        if not frames:
            raise ValueError(f"Protocol selected zero {split} samples from {self.index_path}")
        return pd.concat(frames, axis=0, ignore_index=True)

    def _init_fast_columns(self):
        self._arrow_paths = self.metadata["arrow_path"].astype(str).tolist()
        self._batch_ids = self.metadata["batch_id"].astype("int64").tolist()
        self._row_in_batch = self.metadata["row_in_batch"].astype("int64").tolist()
        self.targets = self.metadata["_target"].astype("int64").tolist()
        self.binary_targets = self.metadata["label"].astype("int64").tolist()

    def _store(self) -> _ArrowImageStore:
        if self._arrow_store is None:
            self._arrow_store = _ArrowImageStore(self.root, self.image_column)
        return self._arrow_store

    def _load_image(self, index: int):
        payload = self._store().read(
            self._arrow_paths[index],
            self._batch_ids[index],
            self._row_in_batch[index],
        )
        return _image_from_payload(payload)


def _load_protocol(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"CAIDBenchmark protocol must be a mapping: {path}")
    return payload


def _protocol_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("CAIDBenchmark protocol must define a non-empty tasks list")
    return [dict(task) for task in tasks]


def _task_raw_task_ids(task: dict[str, Any]) -> list[int]:
    include = dict((task.get("filter") or {}).get("include") or {})
    if "task_id" not in include and "task_id" in task:
        include["task_id"] = task["task_id"]
    if "task_id" not in include:
        raise ValueError(f"CAIDBenchmark task is missing filter.include.task_id: {task}")
    value = include["task_id"]
    if isinstance(value, (list, tuple, set)):
        raw_ids = [int(item) for item in value]
    else:
        raw_ids = [int(value)]
    if not raw_ids:
        raise ValueError(f"CAIDBenchmark task has empty task_id selector: {task}")
    return raw_ids


def _load_index(path: Path) -> pd.DataFrame:
    index = pd.read_parquet(path)
    missing = sorted(_REQUIRED_INDEX_COLUMNS - set(index.columns))
    if missing:
        raise ValueError(f"CAIDBenchmark index is missing required columns: {missing}")
    index = index.copy()
    index["task_id"] = index["task_id"].astype("int64")
    index["label"] = index["label"].astype("int64")
    index["split"] = index["split"].astype(str)
    index["arrow_path"] = index["arrow_path"].astype(str)
    index["batch_id"] = index["batch_id"].astype("int64")
    index["row_in_batch"] = index["row_in_batch"].astype("int64")
    return index


def _resolve_protocol_path(protocol_path) -> Path:
    if protocol_path is None:
        path = _DEFAULT_PROTOCOL
    else:
        path = Path(str(protocol_path)).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve() if not path.exists() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CAIDBenchmark protocol does not exist: {path}")
    return path


def _resolve_index_path(index_path, protocol: dict[str, Any], protocol_path: Path) -> Path:
    raw = index_path if index_path is not None else protocol.get("index_path", protocol.get("index"))
    if raw is None:
        raise ValueError("CAIDBenchmark protocol/index configuration must provide index_path")
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        resolved = path
    elif path.exists():
        resolved = path.resolve()
    elif (_REPO_ROOT / path).exists():
        resolved = (_REPO_ROOT / path).resolve()
    else:
        resolved = (protocol_path.parent / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"CAIDBenchmark index does not exist: {resolved}")
    return resolved


def _image_from_payload(image_payload):
    image_bytes = None
    if isinstance(image_payload, dict):
        image_bytes = image_payload.get("bytes")
    elif isinstance(image_payload, (bytes, bytearray, memoryview)):
        image_bytes = bytes(image_payload)
    elif isinstance(image_payload, str):
        with Image.open(image_payload) as image:
            return as_rgb_preserve_jpeg_metadata(image)

    if image_bytes is None:
        raise ValueError("Image payload does not contain bytes.")
    with Image.open(BytesIO(image_bytes)) as image:
        return as_rgb_preserve_jpeg_metadata(image)
