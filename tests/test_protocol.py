from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from datasets.OpenFakeProtocol import OpenFakeProtocol
from protocol_config import GENERATOR_ORDER, MAX_STAGE_ID
from protocol_manifest import build_protocol_from_records
from protocol_metrics import StageMetrics, compute_online_metrics
from utils.onlinesampler import ManifestStageSampler


def _make_record(
    record_id: str,
    path: str,
    source_dataset: str,
    split: str,
    binary_label: str,
    generator_name: str | None = None,
) -> dict[str, str]:
    payload = {
        "record_id": record_id,
        "path": path,
        "source_dataset": source_dataset,
        "split": split,
        "binary_label": binary_label,
    }
    if generator_name is not None:
        payload["generator_name"] = generator_name
    return payload


def _build_toy_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for index in range(15):
        records.append(
            _make_record(
                f"ab_train_real_{index}",
                f"/data/aigibench/train/real_{index}.jpg",
                "aigibench",
                "train",
                "real",
            )
        )
    for index in range(30):
        records.append(
            _make_record(
                f"of_train_real_{index}",
                f"/data/openfake/train/real_{index}.jpg",
                "openfake",
                "train",
                "real",
            )
        )

    for index in range(10):
        records.append(
            _make_record(
                f"ab_train_progan_fake_{index}",
                f"/data/aigibench/train/progan_fake_{index}.jpg",
                "aigibench",
                "train",
                "fake",
                "ProGAN",
            )
        )

    for index in range(10):
        records.append(
            _make_record(
                f"of_train_sd15_fake_{index}",
                f"/data/openfake/train/sd15_fake_{index}.jpg",
                "openfake",
                "train",
                "fake",
                "Stable Diffusion 1.5",
            )
        )

    for index in range(10):
        records.append(
            _make_record(
                f"of_train_sd21_fake_{index}",
                f"/data/openfake/train/sd21_fake_{index}.jpg",
                "openfake",
                "train",
                "fake",
                "Stable Diffusion 2.1",
            )
        )

    for index in range(12):
        records.append(
            _make_record(
                f"ab_test_real_{index}",
                f"/data/aigibench/test/real_{index}.jpg",
                "aigibench",
                "test",
                "real",
            )
        )
    for index in range(12):
        records.append(
            _make_record(
                f"of_test_real_{index}",
                f"/data/openfake/test/real_{index}.jpg",
                "openfake",
                "test",
                "real",
            )
        )

    for index in range(4):
        records.append(
            _make_record(
                f"ab_test_progan_fake_{index}",
                f"/data/aigibench/test/progan_fake_{index}.jpg",
                "aigibench",
                "test",
                "fake",
                "ProGAN",
            )
        )

    for index in range(3):
        records.append(
            _make_record(
                f"of_test_sd15_fake_{index}",
                f"/data/openfake/test/sd15_fake_{index}.jpg",
                "openfake",
                "test",
                "fake",
                "Stable Diffusion 1.5",
            )
        )

    for index in range(2):
        records.append(
            _make_record(
                f"of_test_sd21_fake_{index}",
                f"/data/openfake/test/sd21_fake_{index}.jpg",
                "openfake",
                "test",
                "fake",
                "Stable Diffusion 2.1",
            )
        )

    for index in range(5):
        records.append(
            _make_record(
                f"ab_test_r3gan_fake_{index}",
                f"/data/aigibench/test/r3gan_fake_{index}.jpg",
                "aigibench",
                "test",
                "fake",
                "R3GAN",
            )
        )
    return records


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = build_protocol_from_records(_build_toy_records(), seed=7)

    def test_generator_order_file_shape(self) -> None:
        self.assertEqual(len(GENERATOR_ORDER), 29)
        self.assertEqual(GENERATOR_ORDER[0]["generator_name"], "ProGAN")
        self.assertEqual(GENERATOR_ORDER[-1]["generator_name"], "Imagen 4")

    def test_stage_manifest_contains_all_stages(self) -> None:
        self.assertEqual(sorted(self.protocol.train_by_stage), list(range(MAX_STAGE_ID + 1)))

    def test_stage_zero_uses_only_progan_fakes(self) -> None:
        stage_zero = self.protocol.train_by_stage[0]
        self.assertEqual(stage_zero["generators"], ["ProGAN"])
        records_by_id = {record.record_id: record for record in self.protocol.records}
        fake_sources = {
            records_by_id[record_id].source_dataset
            for record_id in stage_zero["fake_ids"]
        }
        self.assertEqual(fake_sources, {"aigibench"})

    def test_fake_samples_follow_contiguous_702010_windows(self) -> None:
        records_by_id = {record.record_id: record for record in self.protocol.records}
        grouped: dict[str, list[int]] = {}
        for stage_id, stage_info in self.protocol.train_by_stage.items():
            for record_id in stage_info["fake_ids"]:
                record = records_by_id[record_id]
                grouped.setdefault(record.generator_name or "", []).append(stage_id)

        self.assertEqual(grouped["ProGAN"].count(0), 7)
        self.assertEqual(grouped["ProGAN"].count(1), 2)
        self.assertEqual(grouped["ProGAN"].count(2), 1)
        self.assertEqual(grouped["Stable Diffusion 1.5"].count(1), 7)
        self.assertEqual(grouped["Stable Diffusion 1.5"].count(2), 2)
        self.assertEqual(grouped["Stable Diffusion 1.5"].count(3), 1)

    def test_each_stage_has_real_fake_balance(self) -> None:
        for stage_info in self.protocol.train_by_stage.values():
            self.assertEqual(stage_info["fake_count"], stage_info["real_count"])

    def test_internal_test_slices_are_balanced(self) -> None:
        progan = self.protocol.internal_tests["ProGAN"]
        self.assertEqual(len(progan.fake_ids), len(progan.real_ids))
        self.assertEqual(progan.source_dataset, "aigibench")

        sd15 = self.protocol.internal_tests["Stable Diffusion 1.5"]
        self.assertEqual(len(sd15.fake_ids), len(sd15.real_ids))
        self.assertEqual(sd15.source_dataset, "openfake")

    def test_external_tests_exclude_progan(self) -> None:
        self.assertNotIn("ProGAN", self.protocol.external_tests)
        self.assertIn("R3GAN", self.protocol.external_tests)
        self.assertEqual(
            len(self.protocol.external_tests["R3GAN"].fake_ids),
            len(self.protocol.external_tests["R3GAN"].real_ids),
        )

    def test_metric_computation(self) -> None:
        stage_metrics = [
            StageMetrics(
                stage_id=0,
                internal_accuracy_by_generator={"ProGAN": 0.8},
                external_accuracy_by_subset={"R3GAN": 0.4},
                new_generators=["ProGAN"],
            ),
            StageMetrics(
                stage_id=1,
                internal_accuracy_by_generator={
                    "ProGAN": 0.6,
                    "Stable Diffusion 1.5": 0.9,
                },
                external_accuracy_by_subset={"R3GAN": 0.5},
                new_generators=["Stable Diffusion 1.5"],
            ),
            StageMetrics(
                stage_id=2,
                internal_accuracy_by_generator={
                    "ProGAN": 0.7,
                    "Stable Diffusion 1.5": 0.85,
                    "Stable Diffusion 2.1": 0.95,
                },
                external_accuracy_by_subset={"R3GAN": 0.55},
                new_generators=["Stable Diffusion 2.1"],
            ),
        ]

        metrics = compute_online_metrics(stage_metrics)
        self.assertAlmostEqual(metrics["avg_accuracy_by_stage"][0], 0.8)
        self.assertAlmostEqual(metrics["avg_accuracy_by_stage"][1], 0.75)
        self.assertAlmostEqual(metrics["forgetting_by_stage"][1], 0.1)
        self.assertAlmostEqual(metrics["plasticity_by_stage"][1], 0.9)
        self.assertAlmostEqual(metrics["external_accuracy_by_stage"][2], 0.55)

    def test_openfake_protocol_dataset_and_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            protocol = build_protocol_from_records(_build_toy_records(), seed=7)

            for record in protocol.records:
                record.path = record.path.lstrip("/")
                image_path = tmp_path / record.path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(123, 45, 67)).save(image_path)

            manifest_path = tmp_path / "stage_manifest.json"
            manifest_path.write_text(
                json.dumps(protocol.to_jsonable(), indent=2),
                encoding="utf-8",
            )

            train_dataset = OpenFakeProtocol(
                root=tmpdir,
                train=True,
                transform=None,
                protocol_manifest=str(manifest_path),
            )
            test_dataset = OpenFakeProtocol(
                root=tmpdir,
                train=False,
                transform=None,
                protocol_manifest=str(manifest_path),
            )

            self.assertEqual(len(train_dataset.stage_indices[0]), 7 + 7)
            self.assertIn("ProGAN", test_dataset.internal_slices)
            self.assertIn("R3GAN", test_dataset.external_slices)

            sampler = ManifestStageSampler(train_dataset, train_dataset.stage_indices)
            sampler.set_task(1)
            self.assertEqual(len(list(iter(sampler))), len(train_dataset.stage_indices[1]))


if __name__ == "__main__":
    unittest.main()
