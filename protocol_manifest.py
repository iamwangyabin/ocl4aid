"""Manifest construction utilities for the OpenFake protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from protocol_config import (
    BLURRY_WEIGHTS,
    DEFAULT_SEED,
    EXTERNAL_SOURCE_DATASET,
    GENERATOR_ORDER,
    GENERATOR_STAGE_MAP,
    INTERNAL_DATASET,
    MAX_STAGE_ID,
)


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    path: str
    source_dataset: str
    split: str
    binary_label: str
    generator_name: str | None = None
    subset_name: str | None = None
    release_date: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceRecord":
        required = ["record_id", "path", "source_dataset", "split", "binary_label"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return cls(
            record_id=str(payload["record_id"]),
            path=str(payload["path"]),
            source_dataset=str(payload["source_dataset"]),
            split=str(payload["split"]),
            binary_label=str(payload["binary_label"]),
            generator_name=payload.get("generator_name"),
            subset_name=payload.get("subset_name"),
            release_date=payload.get("release_date"),
        )


@dataclass
class ManifestRecord:
    record_id: str
    path: str
    source_dataset: str
    split: str
    binary_label: str
    generator_name: str | None
    subset_name: str | None
    stage_id: int | None
    class_id: int
    generator_id: int
    is_external: bool


@dataclass
class TestSlice:
    name: str
    fake_generator: str
    source_dataset: str
    fake_ids: list[str]
    real_ids: list[str]

    @property
    def sample_ids(self) -> list[str]:
        return self.fake_ids + self.real_ids

    @property
    def balanced_count(self) -> int:
        return len(self.fake_ids)


@dataclass
class ProtocolArtifacts:
    records: list[ManifestRecord]
    generator_order: list[dict[str, Any]]
    label_space: dict[str, int]
    train_by_stage: dict[int, dict[str, Any]]
    internal_tests: dict[str, TestSlice]
    external_tests: dict[str, TestSlice]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "generator_order": self.generator_order,
            "label_space": self.label_space,
            "records": [asdict(record) for record in self.records],
            "train_by_stage": self.train_by_stage,
            "internal_tests": {
                name: {
                    "name": test_slice.name,
                    "fake_generator": test_slice.fake_generator,
                    "source_dataset": test_slice.source_dataset,
                    "fake_ids": test_slice.fake_ids,
                    "real_ids": test_slice.real_ids,
                    "sample_ids": test_slice.sample_ids,
                    "balanced_count": test_slice.balanced_count,
                }
                for name, test_slice in self.internal_tests.items()
            },
            "external_tests": {
                name: {
                    "name": test_slice.name,
                    "fake_generator": test_slice.fake_generator,
                    "source_dataset": test_slice.source_dataset,
                    "fake_ids": test_slice.fake_ids,
                    "real_ids": test_slice.real_ids,
                    "sample_ids": test_slice.sample_ids,
                    "balanced_count": test_slice.balanced_count,
                }
                for name, test_slice in self.external_tests.items()
            },
        }

    def write_json(self, output_path: str | Path) -> None:
        Path(output_path).write_text(
            json.dumps(self.to_jsonable(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def load_records_jsonl(path: str | Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                records.append(SourceRecord.from_dict(json.loads(payload)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def build_protocol_from_records(
    source_records: Iterable[SourceRecord | dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
) -> ProtocolArtifacts:
    source = [
        record if isinstance(record, SourceRecord) else SourceRecord.from_dict(record)
        for record in source_records
    ]
    record_map = {record.record_id: record for record in source}
    _validate_unique_record_ids(source)

    shuffled_groups = _group_and_shuffle_training_fakes(source, seed=seed)
    stage_to_fake_ids = {stage_id: [] for stage_id in range(MAX_STAGE_ID + 1)}
    fake_stage_generators = {stage_id: set() for stage_id in range(MAX_STAGE_ID + 1)}

    for generator_name, fake_group in shuffled_groups.items():
        first_stage = GENERATOR_STAGE_MAP[generator_name]
        for assigned_stage, record_ids in _assign_blurry_windows(fake_group, first_stage).items():
            stage_to_fake_ids[assigned_stage].extend(record_ids)
            fake_stage_generators[assigned_stage].add(generator_name)

    stage_to_real_ids = _assign_real_slices(record_map, stage_to_fake_ids, seed=seed)

    records: list[ManifestRecord] = []
    for stage_id in range(MAX_STAGE_ID + 1):
        for record_id in stage_to_fake_ids[stage_id] + stage_to_real_ids[stage_id]:
            source_record = record_map[record_id]
            records.append(_to_manifest_record(source_record, stage_id=stage_id, is_external=False))

    internal_tests = _build_internal_test_slices(record_map, seed=seed)
    external_tests = _build_external_test_slices(record_map, seed=seed)

    used_test_ids = set()
    for test_slice in list(internal_tests.values()) + list(external_tests.values()):
        for record_id in test_slice.sample_ids:
            if record_id in used_test_ids:
                continue
            source_record = record_map[record_id]
            records.append(
                _to_manifest_record(
                    source_record,
                    stage_id=None,
                    is_external=test_slice.name in external_tests,
                )
            )
            used_test_ids.add(record_id)

    records_by_id: dict[str, ManifestRecord] = {}
    for record in records:
        existing = records_by_id.get(record.record_id)
        if existing is None:
            records_by_id[record.record_id] = record
            continue
        if record.is_external and not existing.is_external:
            existing.is_external = True

    train_by_stage = {
        stage_id: {
            "stage_id": stage_id,
            "stage_name": GENERATOR_ORDER[stage_id]["generator_name"],
            "sample_ids": stage_to_fake_ids[stage_id] + stage_to_real_ids[stage_id],
            "fake_ids": stage_to_fake_ids[stage_id],
            "real_ids": stage_to_real_ids[stage_id],
            "fake_count": len(stage_to_fake_ids[stage_id]),
            "real_count": len(stage_to_real_ids[stage_id]),
            "generators": sorted(fake_stage_generators[stage_id]),
        }
        for stage_id in range(MAX_STAGE_ID + 1)
    }

    label_space = {"real": 0}
    for entry in GENERATOR_ORDER:
        label_space[entry["generator_name"]] = entry["stage_id"] + 1

    return ProtocolArtifacts(
        records=sorted(records_by_id.values(), key=lambda item: item.record_id),
        generator_order=GENERATOR_ORDER,
        label_space=label_space,
        train_by_stage=train_by_stage,
        internal_tests=internal_tests,
        external_tests=external_tests,
    )


def _validate_unique_record_ids(records: Iterable[SourceRecord]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.record_id in seen:
            duplicates.add(record.record_id)
        seen.add(record.record_id)
    if duplicates:
        raise ValueError(f"Duplicate record_id values: {sorted(duplicates)}")


def _group_and_shuffle_training_fakes(
    records: Iterable[SourceRecord],
    *,
    seed: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        if record.split != "train" or record.binary_label != "fake":
            continue
        if record.generator_name not in GENERATOR_STAGE_MAP:
            continue
        if record.generator_name == "ProGAN" and record.source_dataset != EXTERNAL_SOURCE_DATASET:
            continue
        if record.generator_name != "ProGAN" and record.source_dataset != INTERNAL_DATASET:
            continue
        grouped.setdefault(record.generator_name, []).append(record.record_id)

    shuffled: dict[str, list[str]] = {}
    for generator_name, record_ids in grouped.items():
        ordered_ids = sorted(record_ids)
        rng = random.Random(seed + GENERATOR_STAGE_MAP[generator_name])
        rng.shuffle(ordered_ids)
        shuffled[generator_name] = ordered_ids
    return shuffled


def _assign_blurry_windows(record_ids: list[str], first_stage: int) -> dict[int, list[str]]:
    available = [
        (first_stage + offset, weight)
        for offset, weight in enumerate(BLURRY_WEIGHTS)
        if first_stage + offset <= MAX_STAGE_ID
    ]
    total_weight = sum(weight for _, weight in available)
    normalized = [(stage_id, weight / total_weight) for stage_id, weight in available]
    counts = _largest_remainder_counts(len(record_ids), [weight for _, weight in normalized])

    assigned: dict[int, list[str]] = {}
    cursor = 0
    for (stage_id, _), count in zip(normalized, counts):
        assigned[stage_id] = record_ids[cursor : cursor + count]
        cursor += count
    return assigned


def _largest_remainder_counts(total: int, weights: list[float]) -> list[int]:
    raw = [total * weight for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    ranked_indices = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in ranked_indices[:remainder]:
        counts[index] += 1
    return counts


def _assign_real_slices(
    record_map: dict[str, SourceRecord],
    stage_to_fake_ids: dict[int, list[str]],
    *,
    seed: int,
) -> dict[int, list[str]]:
    aigibench_real = sorted(
        record.record_id
        for record in record_map.values()
        if record.split == "train"
        and record.binary_label == "real"
        and record.source_dataset == EXTERNAL_SOURCE_DATASET
    )
    openfake_real = sorted(
        record.record_id
        for record in record_map.values()
        if record.split == "train"
        and record.binary_label == "real"
        and record.source_dataset == INTERNAL_DATASET
    )
    random.Random(seed).shuffle(aigibench_real)
    random.Random(seed + 1).shuffle(openfake_real)

    used: set[str] = set()
    stage_to_real_ids = {stage_id: [] for stage_id in range(MAX_STAGE_ID + 1)}
    for stage_id in range(MAX_STAGE_ID + 1):
        target = len(stage_to_fake_ids[stage_id])
        primary_pool = aigibench_real if stage_id == 0 else openfake_real
        fallback_pool = openfake_real if stage_id == 0 else aigibench_real
        selected = _take_available(primary_pool, used, target)
        if len(selected) < target:
            selected.extend(_take_available(fallback_pool, used, target - len(selected)))
        if len(selected) != target:
            raise ValueError(
                f"Not enough training real samples for stage {stage_id}: "
                f"need {target}, found {len(selected)}"
            )
        used.update(selected)
        stage_to_real_ids[stage_id] = selected
    return stage_to_real_ids


def _take_available(pool: list[str], used: set[str], count: int) -> list[str]:
    if count <= 0:
        return []
    selected: list[str] = []
    for record_id in pool:
        if record_id in used:
            continue
        selected.append(record_id)
        if len(selected) == count:
            break
    return selected


def _build_internal_test_slices(
    record_map: dict[str, SourceRecord],
    *,
    seed: int,
) -> dict[str, TestSlice]:
    openfake_real_pool = sorted(
        record.record_id
        for record in record_map.values()
        if record.source_dataset == INTERNAL_DATASET
        and record.split == "test"
        and record.binary_label == "real"
    )
    progan_real_pool = sorted(
        record.record_id
        for record in record_map.values()
        if record.source_dataset == EXTERNAL_SOURCE_DATASET
        and record.split == "test"
        and record.binary_label == "real"
    )

    internal: dict[str, TestSlice] = {}
    for generator_name in GENERATOR_STAGE_MAP:
        if generator_name == "ProGAN":
            fake_ids = sorted(
                record.record_id
                for record in record_map.values()
                if record.source_dataset == EXTERNAL_SOURCE_DATASET
                and record.split == "test"
                and record.binary_label == "fake"
                and record.generator_name == "ProGAN"
            )
            if not fake_ids:
                continue
            real_ids = _sample_real_ids(
                progan_real_pool,
                len(fake_ids),
                seed=seed + 1000,
                key="internal:ProGAN",
            )
            internal[generator_name] = TestSlice(
                name=generator_name,
                fake_generator=generator_name,
                source_dataset=EXTERNAL_SOURCE_DATASET,
                fake_ids=fake_ids,
                real_ids=real_ids,
            )
            continue

        fake_ids = sorted(
            record.record_id
            for record in record_map.values()
            if record.source_dataset == INTERNAL_DATASET
            and record.split == "test"
            and record.binary_label == "fake"
            and record.generator_name == generator_name
        )
        if not fake_ids:
            continue
        real_ids = _sample_real_ids(
            openfake_real_pool,
            len(fake_ids),
            seed=seed + 2000,
            key=f"internal:{generator_name}",
        )
        internal[generator_name] = TestSlice(
            name=generator_name,
            fake_generator=generator_name,
            source_dataset=INTERNAL_DATASET,
            fake_ids=fake_ids,
            real_ids=real_ids,
        )
    return internal


def _build_external_test_slices(
    record_map: dict[str, SourceRecord],
    *,
    seed: int,
) -> dict[str, TestSlice]:
    aigibench_real_pool = sorted(
        record.record_id
        for record in record_map.values()
        if record.source_dataset == EXTERNAL_SOURCE_DATASET
        and record.split == "test"
        and record.binary_label == "real"
    )
    fake_groups: dict[str, list[str]] = {}
    for record in record_map.values():
        if (
            record.source_dataset != EXTERNAL_SOURCE_DATASET
            or record.split != "test"
            or record.binary_label != "fake"
            or record.generator_name in (None, "ProGAN")
        ):
            continue
        fake_groups.setdefault(record.generator_name, []).append(record.record_id)

    external: dict[str, TestSlice] = {}
    for subset_name, fake_ids in sorted(fake_groups.items()):
        fake_ids = sorted(fake_ids)
        real_ids = _sample_real_ids(
            aigibench_real_pool,
            len(fake_ids),
            seed=seed + 3000,
            key=f"external:{subset_name}",
        )
        external[subset_name] = TestSlice(
            name=subset_name,
            fake_generator=subset_name,
            source_dataset=EXTERNAL_SOURCE_DATASET,
            fake_ids=fake_ids,
            real_ids=real_ids,
        )
    return external


def _sample_real_ids(pool: list[str], count: int, *, seed: int, key: str) -> list[str]:
    if count > len(pool):
        raise ValueError(
            f"Not enough real test samples for {key}: need {count}, available {len(pool)}"
        )
    rng = random.Random(f"{seed}:{key}")
    sampled = rng.sample(pool, count)
    sampled.sort()
    return sampled


def _to_manifest_record(
    source_record: SourceRecord,
    *,
    stage_id: int | None,
    is_external: bool,
) -> ManifestRecord:
    if source_record.binary_label == "real":
        class_id = 0
        generator_id = -1
    else:
        if source_record.generator_name not in GENERATOR_STAGE_MAP:
            if not is_external:
                raise ValueError(f"Unknown generator for fake sample: {source_record.generator_name}")
            generator_id = -1
            class_id = -1
        else:
            generator_id = GENERATOR_STAGE_MAP[source_record.generator_name]
            class_id = generator_id + 1

    return ManifestRecord(
        record_id=source_record.record_id,
        path=source_record.path,
        source_dataset=source_record.source_dataset,
        split=source_record.split,
        binary_label=source_record.binary_label,
        generator_name=source_record.generator_name,
        subset_name=source_record.subset_name,
        stage_id=stage_id,
        class_id=class_id,
        generator_id=generator_id,
        is_external=is_external,
    )
