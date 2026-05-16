"""Export fixed OpenFake v2 continual-learning protocol subsets.

This tool is intended for an already downloaded Hugging Face snapshot of
``ComplexDataLab/OpenFake``. It samples a deterministic capped training stream
from ``core/train`` and exports full ID/OOD/Wild evaluation splits by default:

- core/train -> training stream
- core/validation -> internal ID evaluation
- core/test -> external OOD evaluation
- reddit/test -> external Wild evaluation

It writes images to disk because the training dataset consumes normal image
paths through ``stage_manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from io import BytesIO
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protocol_manifest import build_protocol_from_records, load_records_jsonl


INTERNAL_SOURCE = "openfake"
OOD_SOURCE = "openfake_ood"
WILD_SOURCE = "openfake_wild"


@dataclass(frozen=True)
class SelectedRow:
    record_id: str
    parquet_path: str
    row_index: int
    path: str
    source_dataset: str
    split: str
    binary_label: str
    generator_name: str | None
    subset_name: str
    release_date: str | None
    score: int

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "path": self.path,
            "source_dataset": self.source_dataset,
            "split": self.split,
            "binary_label": self.binary_label,
            "subset_name": self.subset_name,
        }
        if self.generator_name is not None:
            payload["generator_name"] = self.generator_name
        if self.release_date is not None:
            payload["release_date"] = self.release_date
        return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)

    snapshot_root = Path(config["snapshot_root"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    metadata_csv = Path(config["model_metadata_csv"]).expanduser().resolve()
    seed = int(config.get("seed", 13))
    train_cap = int(config["train_fake_cap_per_model"])
    train_real_ratio = float(config.get("train_real_ratio", 1.0))

    model_rows = load_model_rows(metadata_csv)
    train_models = [
        row
        for row in model_rows
        if int(row.get("train_fake", 0)) > 0
        and not bool(config.get("exclude_video_models", False) and row.get("has_only_video") == "1")
    ]
    train_models.sort(key=model_sort_key)
    train_model_names = [row["model"] for row in train_models]
    train_model_set = set(train_model_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    generator_order = [
        {
            "stage_id": stage_id,
            "generator_name": row["model"],
            "source_dataset": INTERNAL_SOURCE,
            "release_date": row.get("adjusted_release_date") or row.get("dataset_release_date"),
            "date_precision": row.get("date_precision"),
        }
        for stage_id, row in enumerate(train_models)
    ]
    generator_order_path = output_dir / "generator_order.json"
    generator_order_path.write_text(
        json.dumps(generator_order, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    files_by_split = {
        "core/train": sorted((snapshot_root / "core").glob("train-*.parquet")),
        "core/validation": sorted((snapshot_root / "core").glob("validation-*.parquet")),
        "core/test": sorted((snapshot_root / "core").glob("test-*.parquet")),
        "reddit/test": sorted((snapshot_root / "reddit").glob("test-*.parquet")),
    }
    validate_files(files_by_split)

    print(
        "Files:",
        {split: len(files) for split, files in files_by_split.items()},
        flush=True,
    )
    print(
        f"Training models: {len(train_model_names)} | "
        f"cap/model: {train_cap} | seed: {seed}",
        flush=True,
    )

    selected = select_rows(
        files_by_split=files_by_split,
        model_rows={row["model"]: row for row in model_rows},
        train_model_set=train_model_set,
        seed=seed,
        train_cap=train_cap,
        train_real_ratio=train_real_ratio,
        include_full_validation=bool(config.get("include_full_validation", True)),
        include_full_ood_test=bool(config.get("include_full_ood_test", True)),
        include_full_wild_test=bool(config.get("include_full_wild_test", True)),
    )
    summary = summarize_selection(selected)
    print_summary(summary)

    metadata_path = output_dir / "metadata.jsonl"
    extract_images_and_write_metadata(
        selected=selected,
        output_dir=output_dir,
        metadata_path=metadata_path,
        batch_size=int(config.get("image_batch_size", 64)),
    )

    records = load_records_jsonl(metadata_path)
    protocol = build_protocol_from_records(
        records,
        seed=seed,
        openfake_only=True,
        generator_order=generator_order,
        include_external_tests=True,
        external_source_datasets=[OOD_SOURCE, WILD_SOURCE],
    )
    manifest_path = output_dir / "stage_manifest.json"
    protocol.write_json(manifest_path)

    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote generator order: {generator_order_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Train with: --protocol_manifest {manifest_path} --data_dir {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed OpenFake v2 protocol subset")
    parser.add_argument("--config", required=True, help="Preset JSON file")
    parser.add_argument("--snapshot-root", default=None, help="Local HF snapshot root")
    parser.add_argument("--model-metadata-csv", default=None, help="Model release/count CSV")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--train-fake-cap-per-model", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key in ["snapshot_root", "model_metadata_csv", "output_dir"]:
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    if args.train_fake_cap_per_model is not None:
        merged["train_fake_cap_per_model"] = args.train_fake_cap_per_model
    if args.seed is not None:
        merged["seed"] = args.seed

    required = ["snapshot_root", "model_metadata_csv", "output_dir", "train_fake_cap_per_model"]
    missing = [key for key in required if not merged.get(key)]
    if missing:
        raise ValueError(f"Missing required config values: {missing}")
    return merged


def load_model_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "model" not in rows[0]:
        raise ValueError(f"Invalid model metadata CSV: {path}")
    return rows


def model_sort_key(row: dict[str, str]) -> tuple[str, str]:
    return (normalize_date(row.get("adjusted_release_date") or row.get("dataset_release_date")), row["model"])


def normalize_date(value: str | None) -> str:
    if not value:
        return "9999-12-31"
    parts = value.strip().split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    return value.strip()


def validate_files(files_by_split: dict[str, list[Path]]) -> None:
    missing = [split for split, files in files_by_split.items() if not files]
    if missing:
        raise FileNotFoundError(f"No parquet files found for splits: {missing}")


def select_rows(
    *,
    files_by_split: dict[str, list[Path]],
    model_rows: dict[str, dict[str, str]],
    train_model_set: set[str],
    seed: int,
    train_cap: int,
    train_real_ratio: float,
    include_full_validation: bool,
    include_full_ood_test: bool,
    include_full_wild_test: bool,
) -> list[SelectedRow]:
    train_fake_heaps: dict[str, list[tuple[int, str, SelectedRow]]] = {
        model: [] for model in train_model_set
    }
    max_train_real = int(train_cap * len(train_model_set) * train_real_ratio) + len(train_model_set)
    train_real_heap: list[tuple[int, str, SelectedRow]] = []
    eval_rows: list[SelectedRow] = []

    for split, files in files_by_split.items():
        if split == "core/train":
            scan_train_files(
                files=files,
                model_rows=model_rows,
                train_model_set=train_model_set,
                seed=seed,
                train_cap=train_cap,
                max_train_real=max_train_real,
                fake_heaps=train_fake_heaps,
                real_heap=train_real_heap,
            )
            continue
        if split == "core/validation" and include_full_validation:
            eval_rows.extend(
                scan_eval_files(
                    files=files,
                    source_dataset=INTERNAL_SOURCE,
                    output_subset="core_validation",
                    manifest_split="test",
                    subset_name="core/validation",
                    model_rows=model_rows,
                    allowed_fake_models=train_model_set,
                    seed=seed,
                )
            )
        elif split == "core/test" and include_full_ood_test:
            eval_rows.extend(
                scan_eval_files(
                    files=files,
                    source_dataset=OOD_SOURCE,
                    output_subset="core_test",
                    manifest_split="test",
                    subset_name="core/test",
                    model_rows=model_rows,
                    allowed_fake_models=None,
                    seed=seed,
                )
            )
        elif split == "reddit/test" and include_full_wild_test:
            eval_rows.extend(
                scan_eval_files(
                    files=files,
                    source_dataset=WILD_SOURCE,
                    output_subset="reddit_test",
                    manifest_split="test",
                    subset_name="reddit/test",
                    model_rows=model_rows,
                    allowed_fake_models=None,
                    seed=seed,
                )
            )

    train_fake_rows: list[SelectedRow] = []
    for heap in train_fake_heaps.values():
        train_fake_rows.extend(item[2] for item in heap)
    target_real = int(round(len(train_fake_rows) * train_real_ratio))
    train_real_rows = sorted((item[2] for item in train_real_heap), key=lambda item: item.score)[:target_real]
    if len(train_real_rows) != target_real:
        raise RuntimeError(f"Need {target_real} training real rows, selected {len(train_real_rows)}")

    selected = train_fake_rows + train_real_rows + eval_rows
    selected.sort(key=lambda item: (item.subset_name, item.binary_label, item.generator_name or "", item.score))
    return selected


def scan_train_files(
    *,
    files: list[Path],
    model_rows: dict[str, dict[str, str]],
    train_model_set: set[str],
    seed: int,
    train_cap: int,
    max_train_real: int,
    fake_heaps: dict[str, list[tuple[int, str, SelectedRow]]],
    real_heap: list[tuple[int, str, SelectedRow]],
) -> None:
    import pyarrow.parquet as pq

    for file_index, path in enumerate(files, start=1):
        table = pq.read_table(path, columns=["label", "model", "release_date"])
        labels = table["label"].to_pylist()
        models = table["model"].to_pylist()
        release_dates = table["release_date"].to_pylist()
        for row_index, (label, model, release_date) in enumerate(zip(labels, models, release_dates)):
            if label == "fake":
                if model not in train_model_set:
                    continue
                row = make_selected_row(
                    path=path,
                    row_index=row_index,
                    source_dataset=INTERNAL_SOURCE,
                    manifest_split="train",
                    binary_label="fake",
                    model=model,
                    subset_name="core/train",
                    output_subset="train",
                    release_date=model_rows.get(model, {}).get("adjusted_release_date") or release_date,
                    seed=seed,
                )
                push_top_k(fake_heaps[model], train_cap, row)
            elif label == "real":
                row = make_selected_row(
                    path=path,
                    row_index=row_index,
                    source_dataset=INTERNAL_SOURCE,
                    manifest_split="train",
                    binary_label="real",
                    model=None,
                    subset_name="core/train",
                    output_subset="train",
                    release_date=release_date,
                    seed=seed,
                )
                push_top_k(real_heap, max_train_real, row)
        if file_index % 50 == 0:
            print(f"Scanned train metadata: {file_index}/{len(files)} files", flush=True)


def scan_eval_files(
    *,
    files: list[Path],
    source_dataset: str,
    output_subset: str,
    manifest_split: str,
    subset_name: str,
    model_rows: dict[str, dict[str, str]],
    allowed_fake_models: set[str] | None,
    seed: int,
) -> list[SelectedRow]:
    import pyarrow.parquet as pq

    selected: list[SelectedRow] = []
    for file_index, path in enumerate(files, start=1):
        table = pq.read_table(path, columns=["label", "model", "release_date"])
        labels = table["label"].to_pylist()
        models = table["model"].to_pylist()
        release_dates = table["release_date"].to_pylist()
        for row_index, (label, model, release_date) in enumerate(zip(labels, models, release_dates)):
            if label == "fake":
                if allowed_fake_models is not None and model not in allowed_fake_models:
                    continue
                generator_name = model
            elif label == "real":
                generator_name = None
            else:
                continue
            selected.append(
                make_selected_row(
                    path=path,
                    row_index=row_index,
                    source_dataset=source_dataset,
                    manifest_split=manifest_split,
                    binary_label=label,
                    model=generator_name,
                    subset_name=subset_name,
                    output_subset=output_subset,
                    release_date=model_rows.get(model, {}).get("adjusted_release_date") or release_date,
                    seed=seed,
                )
            )
        if file_index % 10 == 0:
            print(f"Scanned {subset_name} metadata: {file_index}/{len(files)} files", flush=True)
    return selected


def make_selected_row(
    *,
    path: Path,
    row_index: int,
    source_dataset: str,
    manifest_split: str,
    binary_label: str,
    model: str | None,
    subset_name: str,
    output_subset: str,
    release_date: str | None,
    seed: int,
) -> SelectedRow:
    parquet_key = f"{path.parent.name}/{path.name}"
    score = stable_score(seed, parquet_key, str(row_index), binary_label, model or "real")
    model_slug = slug(model or "real")
    subset_slug = slug(output_subset)
    record_id = f"{subset_slug}_{binary_label}_{model_slug}_{path.stem}_r{row_index:05d}"
    image_path = f"images/{subset_slug}/{record_id}.jpg"
    return SelectedRow(
        record_id=record_id,
        parquet_path=str(path),
        row_index=row_index,
        path=image_path,
        source_dataset=source_dataset,
        split=manifest_split,
        binary_label=binary_label,
        generator_name=model,
        subset_name=subset_name,
        release_date=release_date,
        score=score,
    )


def push_top_k(heap: list[tuple[int, str, SelectedRow]], limit: int, row: SelectedRow) -> None:
    item = (-row.score, row.record_id, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if row.score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def stable_score(seed: int, *parts: str) -> int:
    payload = "::".join([str(seed), *parts]).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest(), 16)


def slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "unknown"


def summarize_selection(selected: list[SelectedRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_records": len(selected),
        "by_subset": {},
    }
    for row in selected:
        subset = summary["by_subset"].setdefault(
            row.subset_name,
            {
                "total": 0,
                "fake": 0,
                "real": 0,
                "fake_by_model": {},
            },
        )
        subset["total"] += 1
        subset[row.binary_label] += 1
        if row.binary_label == "fake" and row.generator_name is not None:
            fake_by_model = subset["fake_by_model"]
            fake_by_model[row.generator_name] = fake_by_model.get(row.generator_name, 0) + 1
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("Selection summary:", flush=True)
    for subset_name, payload in summary["by_subset"].items():
        print(
            f"  {subset_name}: total={payload['total']} "
            f"fake={payload['fake']} real={payload['real']} "
            f"fake_models={len(payload['fake_by_model'])}",
            flush=True,
        )


def extract_images_and_write_metadata(
    *,
    selected: list[SelectedRow],
    output_dir: Path,
    metadata_path: Path,
    batch_size: int,
) -> None:
    import pyarrow.parquet as pq

    by_file: dict[Path, dict[int, SelectedRow]] = {}
    for row in selected:
        by_file.setdefault(Path(row.parquet_path), {})[row.row_index] = row

    written = 0
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        for file_index, (path, rows_by_index) in enumerate(sorted(by_file.items()), start=1):
            parquet = pq.ParquetFile(path)
            selected_indices = sorted(rows_by_index)
            offset = 0
            selected_cursor = 0
            for row_group_index in range(parquet.metadata.num_row_groups):
                row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
                row_group_start = offset
                row_group_end = offset + row_group_rows
                while (
                    selected_cursor < len(selected_indices)
                    and selected_indices[selected_cursor] < row_group_start
                ):
                    selected_cursor += 1
                if (
                    selected_cursor >= len(selected_indices)
                    or selected_indices[selected_cursor] >= row_group_end
                ):
                    offset = row_group_end
                    continue

                table = parquet.read_row_group(row_group_index, columns=["image"])
                images = table.column(0).to_pylist()
                while (
                    selected_cursor < len(selected_indices)
                    and selected_indices[selected_cursor] < row_group_end
                ):
                    row_index = selected_indices[selected_cursor]
                    row = rows_by_index[row_index]
                    image_payload = images[row_index - row_group_start]
                    output_path = output_dir / row.path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(image_payload, output_path)
                    metadata_handle.write(json.dumps(row.metadata(), ensure_ascii=True) + "\n")
                    written += 1
                    selected_cursor += 1
                offset = row_group_end
            if file_index % 25 == 0 or file_index == len(by_file):
                print(
                    f"Extracted images: files={file_index}/{len(by_file)} records={written}/{len(selected)}",
                    flush=True,
                )


def save_image(image_payload: Any, output_path: Path) -> None:
    from PIL import Image

    image_bytes = None
    if isinstance(image_payload, dict):
        image_bytes = image_payload.get("bytes")
    elif isinstance(image_payload, (bytes, bytearray)):
        image_bytes = image_payload

    if image_bytes is None:
        raise ValueError(f"Image payload does not contain bytes for {output_path}")

    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(output_path, format="JPEG", quality=95)


if __name__ == "__main__":
    main()
