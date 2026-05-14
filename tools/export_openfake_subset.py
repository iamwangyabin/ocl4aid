"""Export a small OpenFake subset into OCL4AID metadata format.

This script is intentionally small and streaming-friendly. It writes images to
disk and creates metadata.jsonl that can be consumed by protocol_cli.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_GENERATORS = [
    "Stable Diffusion 1.5",
    "Stable Diffusion 2.1",
]

OPENFAKE_MODEL_TO_PROTOCOL = {
    "sd-1.5": "Stable Diffusion 1.5",
    "sd-2.1": "Stable Diffusion 2.1",
    "sd-1.5-dreamshaper": "DreamShaper",
    "sd-1.5-epicdream": "EpicDream",
    "realistic-vision-v5.1": "Realism",
    "sdxl-touchofrealism": "Touch of Realism",
    "flux-mvc5000": "MVC5000",
    "mystic": "Mystic",
    "flux-amateursnapshotphotos": "Amateur Snapshot Photos",
    "sdxl": "Stable Diffusion XL (SDXL)",
    "dall-e-3": "DALL·E 3",
    "dalle-3": "DALL·E 3",
    "midjourney-6": "Midjourney v6",
    "sdxl-epic-realism": "Epic Realism",
    "sdxl-realvis-v5": "RealVisXL-v5",
    "sdxl-juggernaut": "Juggernaut",
    "imagen-3": "Imagen 3",
    "flux.1-schnell": "FLUX.1-Schnell",
    "flux.1-dev": "FLUX.1-dev",
    "grok-2-image-1212": "Grok 2",
    "flux-1.1-pro": "FLUX.1.1-Pro",
    "sd-3.5": "Stable Diffusion 3.5",
    "recraft-v3": "Recraft v3",
    "ideogram-3.0": "Ideogram 3.0",
    "midjourney-7": "Midjourney v7",
    "gpt-image-1": "GPT Image 1",
    "hidream-i1-full": "HiDream-I1 Full",
    "chroma": "Chroma",
    "imagen-4": "Imagen 4",
    "imagen-4.0": "Imagen 4",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a small OpenFake subset")
    parser.add_argument("--dataset-id", default="ComplexDataLab/OpenFake")
    parser.add_argument("--hf-config", default="core")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--output-dir", default="data/openfake_smoke")
    parser.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS)
    parser.add_argument("--fake-train-per-generator", type=int, default=8)
    parser.add_argument("--fake-test-per-generator", type=int, default=2)
    parser.add_argument("--real-train", type=int, default=32)
    parser.add_argument("--real-test", type=int, default=8)
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--model-field", default="model")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--type-field", default="type")
    args = parser.parse_args()

    load_dataset = _load_hf_load_dataset()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    generator_set = _requested_protocol_names(args.generators)
    fake_train_counts: dict[str, int] = defaultdict(int)
    fake_test_counts: dict[str, int] = defaultdict(int)
    real_train_count = 0
    real_test_count = 0
    written = 0

    metadata_path = output_dir / "metadata.jsonl"
    load_kwargs = {
        "split": args.hf_split,
        "streaming": True,
    }
    if args.hf_cache_dir is not None:
        load_kwargs["cache_dir"] = args.hf_cache_dir
    dataset = load_dataset(args.dataset_id, args.hf_config, **load_kwargs)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(dataset):
            binary_label = _binary_label(row, args.label_field, args.type_field)
            if binary_label is None:
                continue

            generator_name = None
            split = None
            if binary_label == "fake":
                raw_model_name = _clean_text(row.get(args.model_field))
                generator_name = _protocol_generator_name(raw_model_name)
                if generator_name not in generator_set:
                    continue
                if fake_train_counts[generator_name] < args.fake_train_per_generator:
                    split = "train"
                    fake_train_counts[generator_name] += 1
                elif fake_test_counts[generator_name] < args.fake_test_per_generator:
                    split = "test"
                    fake_test_counts[generator_name] += 1
                else:
                    continue
            else:
                if real_train_count < args.real_train:
                    split = "train"
                    real_train_count += 1
                elif real_test_count < args.real_test:
                    split = "test"
                    real_test_count += 1
                else:
                    continue

            record_id = _record_id(binary_label, generator_name, split, row_index)
            relative_path = Path("images") / split / f"{record_id}.jpg"
            image_path = output_dir / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            _save_image(row[args.image_field], image_path)

            payload: dict[str, Any] = {
                "record_id": record_id,
                "path": str(relative_path),
                "source_dataset": "openfake",
                "split": split,
                "binary_label": binary_label,
            }
            if generator_name is not None:
                payload["generator_name"] = generator_name
            if row.get("release_date") is not None:
                payload["release_date"] = str(row["release_date"])
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            written += 1

            if _done(
                generator_set,
                fake_train_counts,
                fake_test_counts,
                args.fake_train_per_generator,
                args.fake_test_per_generator,
                real_train_count,
                real_test_count,
                args.real_train,
                args.real_test,
            ):
                break

    print(f"Wrote {written} records to {metadata_path}")


def _binary_label(row: dict[str, Any], label_field: str, type_field: str) -> str | None:
    for field in (type_field, label_field):
        value = _clean_text(row.get(field))
        if value is None:
            continue
        normalized = value.lower().replace("_", " ").replace("-", " ")
        if normalized in {"real", "human", "natural", "authentic"}:
            return "real"
        if normalized in {"fake", "ai", "synthetic", "generated", "ai generated"}:
            return "fake"
        if normalized in {"real image"}:
            return "real"
        if normalized in {"fake image", "ai generated image"}:
            return "fake"
    return None


def _load_hf_load_dataset():
    """Import Hugging Face datasets without resolving this repo's datasets package."""

    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    existing_module = sys.modules.get("datasets")
    removed_local_module = None

    if existing_module is not None and _path_is_relative_to(
        getattr(existing_module, "__file__", None),
        repo_root / "datasets",
    ):
        removed_local_module = sys.modules.pop("datasets")

    try:
        sys.path = [
            path_entry
            for path_entry in sys.path
            if not _same_path(path_entry or os.getcwd(), repo_root)
        ]
        hf_datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        if removed_local_module is not None:
            sys.modules["datasets"] = removed_local_module
        raise SystemExit(
            "Install the Hugging Face datasets package first: pip install datasets"
        ) from exc
    finally:
        sys.path = original_path

    load_dataset = getattr(hf_datasets, "load_dataset", None)
    if load_dataset is None:
        if removed_local_module is not None:
            sys.modules["datasets"] = removed_local_module
        raise SystemExit(
            "Could not import Hugging Face datasets.load_dataset. "
            "Check that the PyPI package 'datasets' is installed."
        )
    return load_dataset


def _same_path(left: str | os.PathLike[str], right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except OSError:
        return False


def _path_is_relative_to(path: str | None, parent: Path) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _requested_protocol_names(generators: list[str]) -> set[str]:
    requested = set()
    known_protocol_names = set(OPENFAKE_MODEL_TO_PROTOCOL.values())
    for generator in generators:
        cleaned = _clean_text(generator)
        if cleaned is None:
            continue
        if cleaned in known_protocol_names:
            requested.add(cleaned)
            continue
        requested.add(_protocol_generator_name(cleaned))
    return requested


def _protocol_generator_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return OPENFAKE_MODEL_TO_PROTOCOL.get(model_name, model_name)


def _record_id(
    binary_label: str,
    generator_name: str | None,
    split: str,
    row_index: int,
) -> str:
    parts = ["openfake", split, binary_label]
    if generator_name is not None:
        parts.append(_slug(generator_name))
    parts.append(f"{row_index:08d}")
    return "_".join(parts)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _save_image(image: Any, path: Path) -> None:
    if hasattr(image, "convert"):
        image.convert("RGB").save(path, format="JPEG", quality=95)
        return
    raise TypeError(f"Unsupported image payload for {path}: {type(image)!r}")


def _done(
    generators: set[str],
    fake_train_counts: dict[str, int],
    fake_test_counts: dict[str, int],
    fake_train_target: int,
    fake_test_target: int,
    real_train_count: int,
    real_test_count: int,
    real_train_target: int,
    real_test_target: int,
) -> bool:
    fake_done = all(
        fake_train_counts[generator] >= fake_train_target
        and fake_test_counts[generator] >= fake_test_target
        for generator in generators
    )
    real_done = real_train_count >= real_train_target and real_test_count >= real_test_target
    return fake_done and real_done


if __name__ == "__main__":
    main()
