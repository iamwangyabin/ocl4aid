"""Hugging Face OpenFake preparation utilities."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from protocol_manifest import build_protocol_from_records, load_records_jsonl


DEFAULT_HF_DATASET_ID = "ComplexDataLab/OpenFake"
DEFAULT_HF_CONFIG = "core"
DEFAULT_HF_SPLIT = "train"
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


def prepare_openfake_protocol_from_hf(
    *,
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    hf_config: str = DEFAULT_HF_CONFIG,
    hf_split: str = DEFAULT_HF_SPLIT,
    output_dir: str | os.PathLike[str] | None = None,
    hf_cache_dir: str | os.PathLike[str] | None = None,
    generators: list[str] | None = None,
    fake_train_per_generator: int = 8,
    fake_test_per_generator: int = 2,
    real_train: int = 32,
    real_test: int = 8,
    image_field: str = "image",
    model_field: str = "model",
    label_field: str = "label",
    type_field: str = "type",
    seed: int = 13,
    streaming: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Create an OpenFake-only protocol cache from Hugging Face.

    The Hugging Face dataset itself is resolved through ``load_dataset``. The
    sampled images and manifest are stored under the Hugging Face datasets cache
    by default so callers do not need to manage a project-local data directory.
    """

    hf_datasets = load_hf_datasets_module()
    generator_names = generators or DEFAULT_GENERATORS
    resolved_output_dir = Path(output_dir) if output_dir else _default_cache_dir(
        hf_datasets,
        dataset_id=dataset_id,
        hf_config=hf_config,
        hf_split=hf_split,
        generators=generator_names,
        fake_train_per_generator=fake_train_per_generator,
        fake_test_per_generator=fake_test_per_generator,
        real_train=real_train,
        real_test=real_test,
        hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        seed=seed,
        streaming=streaming,
    )
    metadata_path = resolved_output_dir / "metadata.jsonl"
    manifest_path = resolved_output_dir / "stage_manifest.json"

    if not force and metadata_path.exists() and manifest_path.exists():
        return {
            "data_dir": resolved_output_dir,
            "metadata_path": metadata_path,
            "manifest_path": manifest_path,
        }

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _export_openfake_subset(
        load_dataset=hf_datasets.load_dataset,
        dataset_id=dataset_id,
        hf_config=hf_config,
        hf_split=hf_split,
        output_dir=resolved_output_dir,
        hf_cache_dir=hf_cache_dir,
        generators=generator_names,
        fake_train_per_generator=fake_train_per_generator,
        fake_test_per_generator=fake_test_per_generator,
        real_train=real_train,
        real_test=real_test,
        image_field=image_field,
        model_field=model_field,
        label_field=label_field,
        type_field=type_field,
        streaming=streaming,
    )
    protocol = build_protocol_from_records(
        load_records_jsonl(metadata_path),
        seed=seed,
        openfake_only=True,
    )
    protocol.write_json(manifest_path)
    return {
        "data_dir": resolved_output_dir,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
    }


def load_hf_datasets_module():
    """Import Hugging Face datasets without resolving this repo's package."""

    repo_root = Path(__file__).resolve().parent
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

    if getattr(hf_datasets, "load_dataset", None) is None:
        if removed_local_module is not None:
            sys.modules["datasets"] = removed_local_module
        raise SystemExit(
            "Could not import Hugging Face datasets.load_dataset. "
            "Check that the PyPI package 'datasets' is installed."
        )
    return hf_datasets


def _export_openfake_subset(
    *,
    load_dataset,
    dataset_id: str,
    hf_config: str,
    hf_split: str,
    output_dir: Path,
    hf_cache_dir: str | os.PathLike[str] | None,
    generators: list[str],
    fake_train_per_generator: int,
    fake_test_per_generator: int,
    real_train: int,
    real_test: int,
    image_field: str,
    model_field: str,
    label_field: str,
    type_field: str,
    streaming: bool,
) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    generator_set = requested_protocol_names(generators)
    fake_train_counts: dict[str, int] = defaultdict(int)
    fake_test_counts: dict[str, int] = defaultdict(int)
    real_train_count = 0
    real_test_count = 0
    written = 0

    metadata_path = output_dir / "metadata.jsonl"
    load_kwargs = {
        "split": hf_split,
        "streaming": streaming,
    }
    if hf_cache_dir is not None:
        load_kwargs["cache_dir"] = str(hf_cache_dir)
    dataset = load_dataset(dataset_id, hf_config, **load_kwargs)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(dataset):
            binary_label = binary_label_from_row(row, label_field, type_field)
            if binary_label is None:
                continue

            generator_name = None
            split = None
            if binary_label == "fake":
                raw_model_name = clean_text(row.get(model_field))
                generator_name = protocol_generator_name(raw_model_name)
                if generator_name not in generator_set:
                    continue
                if fake_train_counts[generator_name] < fake_train_per_generator:
                    split = "train"
                    fake_train_counts[generator_name] += 1
                elif fake_test_counts[generator_name] < fake_test_per_generator:
                    split = "test"
                    fake_test_counts[generator_name] += 1
                else:
                    continue
            else:
                if real_train_count < real_train:
                    split = "train"
                    real_train_count += 1
                elif real_test_count < real_test:
                    split = "test"
                    real_test_count += 1
                else:
                    continue

            record_id = record_id_for(binary_label, generator_name, split, row_index)
            relative_path = Path("images") / split / f"{record_id}.jpg"
            image_path = output_dir / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(row[image_field], image_path)

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

            if done(
                generator_set,
                fake_train_counts,
                fake_test_counts,
                fake_train_per_generator,
                fake_test_per_generator,
                real_train_count,
                real_test_count,
                real_train,
                real_test,
            ):
                break

    _validate_export_counts(
        generator_set,
        fake_train_counts,
        fake_test_counts,
        fake_train_per_generator,
        fake_test_per_generator,
        real_train_count,
        real_test_count,
        real_train,
        real_test,
    )
    print(f"Wrote {written} records to {metadata_path}")


def binary_label_from_row(row: dict[str, Any], label_field: str, type_field: str) -> str | None:
    for field in (type_field, label_field):
        value = clean_text(row.get(field))
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


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def requested_protocol_names(generators: list[str]) -> set[str]:
    requested = set()
    known_protocol_names = set(OPENFAKE_MODEL_TO_PROTOCOL.values())
    for generator in generators:
        cleaned = clean_text(generator)
        if cleaned is None:
            continue
        if cleaned in known_protocol_names:
            requested.add(cleaned)
            continue
        requested.add(protocol_generator_name(cleaned))
    return requested


def protocol_generator_name(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return OPENFAKE_MODEL_TO_PROTOCOL.get(model_name, model_name)


def record_id_for(
    binary_label: str,
    generator_name: str | None,
    split: str,
    row_index: int,
) -> str:
    parts = ["openfake", split, binary_label]
    if generator_name is not None:
        parts.append(slug(generator_name))
    parts.append(f"{row_index:08d}")
    return "_".join(parts)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_image(image: Any, path: Path) -> None:
    if hasattr(image, "convert"):
        image.convert("RGB").save(path, format="JPEG", quality=95)
        return
    raise TypeError(f"Unsupported image payload for {path}: {type(image)!r}")


def done(
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


def _default_cache_dir(hf_datasets, **params: Any) -> Path:
    cache_root = getattr(getattr(hf_datasets, "config", None), "HF_DATASETS_CACHE", None)
    if cache_root is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    digest = hashlib.sha1(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return Path(cache_root) / "ocl4aid" / "openfake_protocol" / digest


def _validate_export_counts(
    generators: set[str],
    fake_train_counts: dict[str, int],
    fake_test_counts: dict[str, int],
    fake_train_target: int,
    fake_test_target: int,
    real_train_count: int,
    real_test_count: int,
    real_train_target: int,
    real_test_target: int,
) -> None:
    missing = []
    for generator in sorted(generators):
        if fake_train_counts[generator] < fake_train_target:
            missing.append(
                f"{generator} train fake: {fake_train_counts[generator]}/{fake_train_target}"
            )
        if fake_test_counts[generator] < fake_test_target:
            missing.append(
                f"{generator} test fake: {fake_test_counts[generator]}/{fake_test_target}"
            )
    if real_train_count < real_train_target:
        missing.append(f"real train: {real_train_count}/{real_train_target}")
    if real_test_count < real_test_target:
        missing.append(f"real test: {real_test_count}/{real_test_target}")
    if missing:
        raise RuntimeError("OpenFake export did not find enough samples: " + "; ".join(missing))


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
