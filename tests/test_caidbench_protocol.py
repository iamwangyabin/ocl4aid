from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from datasets import CAIDBenchmarkProtocol, OnlineIterDataset
from utils.onlinesampler import ManifestStageSampler, ManifestStreamSampler


class _TinyStageDataset:
    classes = [0, 1]

    def __init__(self, size: int):
        self.targets = [idx % 2 for idx in range(size)]

    def __len__(self):
        return len(self.targets)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_arrow(path: Path, n: int = 4) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "image": [_png_bytes((i * 30, 20, 10)) for i in range(n)],
            "label": [i % 2 for i in range(n)],
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _write_protocol(path: Path) -> None:
    path.write_text(
        """
index_path: continual_index.parquet
tasks:
  - id: task_b_first
    name: Task B
    filter:
      include:
        task_id: 20
  - id: task_a_second
    name: Task A
    filter:
      include:
        task_id: 10
""".lstrip(),
        encoding="utf-8",
    )


def _write_index(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for raw_task_id, generator_name, arrow_dir in [
        (10, "Raw A", "Raw_A"),
        (20, "Raw B", "Raw_B"),
    ]:
        for split in ["train", "test"]:
            for row_in_batch, label in enumerate([0, 1, 0, 1]):
                rows.append(
                    {
                        "task_id": raw_task_id,
                        "generator_name": generator_name,
                        "raw_generator_name": generator_name,
                        "split": split,
                        "label": label,
                        "arrow_path": f"{arrow_dir}/{split}.arrow",
                        "batch_id": 0,
                        "row_in_batch": row_in_batch,
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), path)


class CAIDBenchmarkProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "caidbench"
        _write_arrow(self.root / "Raw_A" / "train.arrow")
        _write_arrow(self.root / "Raw_A" / "test.arrow")
        _write_arrow(self.root / "Raw_B" / "train.arrow")
        _write_arrow(self.root / "Raw_B" / "test.arrow")
        self.protocol_path = Path(self.tmp.name) / "model_appearance_order_protocol.yaml"
        self.index_path = Path(self.tmp.name) / "continual_index.parquet"
        _write_protocol(self.protocol_path)
        _write_index(self.index_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _dataset(self, *, train=True):
        return CAIDBenchmarkProtocol(
            root=self.root,
            train=train,
            transform=None,
            protocol_path=self.protocol_path,
            index_path=self.index_path,
        )

    def test_protocol_order_reorders_raw_task_ids(self):
        dataset = self._dataset(train=True)

        self.assertEqual([item["generator_name"] for item in dataset.generator_order], ["Task B", "Task A"])
        self.assertEqual(dataset.active_stage_ids, [0, 1])
        self.assertEqual(dataset.stage_generators, {0: ["Task B"], 1: ["Task A"]})

        first_stage_indices = dataset.stage_indices[0]
        second_stage_indices = dataset.stage_indices[1]
        self.assertEqual(len(first_stage_indices), 4)
        self.assertEqual(len(second_stage_indices), 4)
        self.assertEqual(set(dataset.metadata.iloc[first_stage_indices]["task_id"].tolist()), {20})
        self.assertEqual(set(dataset.metadata.iloc[second_stage_indices]["task_id"].tolist()), {10})

    def test_protocol_targets_are_binary(self):
        dataset = CAIDBenchmarkProtocol(
            root=self.root,
            train=True,
            transform=None,
            protocol_path=self.protocol_path,
            index_path=self.index_path,
        )

        self.assertEqual(dataset.classes, [0, 1])
        self.assertEqual(dataset.label_space, {"real": 0, "fake": 1})
        self.assertEqual([dataset.targets[i] for i in dataset.stage_indices[0]], [0, 1, 0, 1])
        self.assertEqual([dataset.targets[i] for i in dataset.stage_indices[1]], [0, 1, 0, 1])

    def test_lazy_arrow_loading_and_eval_subset(self):
        dataset = self._dataset(train=False)

        image, target = dataset[0]
        self.assertEqual(image.size, (8, 8))
        self.assertIn(target, {0, 1})
        self.assertEqual(set(dataset.internal_slices), {"Task B", "Task A"})

        subset = dataset.make_eval_subset(dataset.internal_slices["Task B"])
        image, target, binary_target = subset[1]
        self.assertEqual(image.size, (8, 8))
        self.assertEqual(target, 1)
        self.assertEqual(binary_target, 1)

    def test_manifest_stream_sampler_flattens_protocol_without_task_switch_api(self):
        dataset = self._dataset(train=True)
        wrapped = OnlineIterDataset(dataset)
        sampler = ManifestStreamSampler(wrapped, dataset.stage_indices, seed=3)

        stream_indices = list(iter(sampler))
        first_end = sampler.stage_end_offsets[0]
        second_end = sampler.stage_end_offsets[1]

        self.assertEqual(first_end, len(dataset.stage_indices[0]))
        self.assertEqual(second_end, len(dataset.stage_indices[0]) + len(dataset.stage_indices[1]))
        self.assertEqual(set(stream_indices[:first_end]), set(dataset.stage_indices[0]))
        self.assertEqual(set(stream_indices[first_end:second_end]), set(dataset.stage_indices[1]))

    def test_manifest_stage_sampler_switches_framework_tasks(self):
        dataset = self._dataset(train=True)
        wrapped = OnlineIterDataset(dataset)
        sampler = ManifestStageSampler(wrapped, dataset.stage_indices, seed=3)

        self.assertEqual(set(iter(sampler)), set(dataset.stage_indices[0]))
        sampler.set_task(1)
        self.assertEqual(set(iter(sampler)), set(dataset.stage_indices[1]))
        self.assertEqual(sampler.stage_end_offsets[0], len(dataset.stage_indices[0]))
        self.assertEqual(
            sampler.stage_end_offsets[1],
            len(dataset.stage_indices[0]) + len(dataset.stage_indices[1]),
        )

    def test_temporal_blurry_sampler_defaults_to_hard_boundaries(self):
        stage_indices = {
            0: list(range(0, 6)),
            1: list(range(6, 12)),
            2: list(range(12, 18)),
        }
        sampler = ManifestStageSampler(
            _TinyStageDataset(18),
            stage_indices,
            seed=11,
            stage_blurry_n=100,
            stage_blurry_m=100,
        )

        for stage_id, expected_indices in stage_indices.items():
            self.assertEqual(set(sampler.indices[stage_id]), set(expected_indices))

    def test_temporal_blurry_sampler_only_mixes_adjacent_stages(self):
        stage_indices = {
            0: list(range(0, 6)),
            1: list(range(6, 12)),
            2: list(range(12, 18)),
        }
        sampler = ManifestStageSampler(
            _TinyStageDataset(18),
            stage_indices,
            seed=11,
            stage_blurry_n=0,
            stage_blurry_m=50,
        )

        def origin(index):
            return index // 6

        all_indices = []
        for indices in sampler.indices.values():
            all_indices.extend(indices)
        self.assertEqual(set(all_indices), set(range(18)))
        self.assertEqual(len(all_indices), len(set(all_indices)))

        self.assertTrue({origin(i) for i in sampler.indices[0]}.issubset({0, 1}))
        self.assertTrue({origin(i) for i in sampler.indices[1]}.issubset({0, 1, 2}))
        self.assertTrue({origin(i) for i in sampler.indices[2]}.issubset({1, 2}))
        self.assertIn(1, {origin(i) for i in sampler.indices[0]})
        self.assertIn(0, {origin(i) for i in sampler.indices[1]})
        self.assertIn(2, {origin(i) for i in sampler.indices[1]})
        self.assertIn(1, {origin(i) for i in sampler.indices[2]})

    def test_temporal_blurry_sampler_can_freeze_base_stage(self):
        stage_indices = {
            0: list(range(0, 6)),
            1: list(range(6, 12)),
            2: list(range(12, 18)),
            3: list(range(18, 24)),
        }
        sampler = ManifestStageSampler(
            _TinyStageDataset(24),
            stage_indices,
            seed=11,
            stage_blurry_n=0,
            stage_blurry_m=50,
            stage_blurry_start_pos=1,
        )

        def origin(index):
            return index // 6

        self.assertEqual(set(sampler.indices[0]), set(stage_indices[0]))
        self.assertTrue({origin(i) for i in sampler.indices[1]}.issubset({1, 2}))
        self.assertTrue({origin(i) for i in sampler.indices[2]}.issubset({1, 2, 3}))
        self.assertTrue({origin(i) for i in sampler.indices[3]}.issubset({2, 3}))


if __name__ == "__main__":
    unittest.main()
