from __future__ import annotations

from io import BytesIO
from typing import Mapping, Sequence

from PIL import Image


_JPEG_LUMA_QTABLE = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)
_JPEG_CHROMA_QTABLE = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)


def as_rgb_preserve_jpeg_metadata(img: Image.Image) -> Image.Image:
    converted = img if img.mode == "RGB" else img.convert("RGB")
    out = converted.copy()
    image_format = getattr(img, "format", None)
    if image_format is not None:
        out.format = image_format
    quantization = getattr(img, "quantization", None)
    if quantization is not None:
        out.quantization = quantization
    return out


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    if str(getattr(img, "format", "")).upper() not in {"JPEG", "JPG"}:
        return None
    qtables = getattr(img, "quantization", None)
    if not isinstance(qtables, Mapping) or not qtables:
        return None

    observed: list[tuple[list[int], Sequence[int]]] = []
    if 0 in qtables:
        observed.append(([int(v) for v in qtables[0]], _JPEG_LUMA_QTABLE))
    if 1 in qtables:
        observed.append(([int(v) for v in qtables[1]], _JPEG_CHROMA_QTABLE))
    if not observed:
        first_table = next(iter(qtables.values()), None)
        if first_table is None:
            return None
        observed.append(([int(v) for v in first_table], _JPEG_LUMA_QTABLE))

    best_quality: int | None = None
    best_score = float("inf")
    for quality in range(1, 101):
        total = 0
        count = 0
        for table, base in observed:
            expected = _jpeg_qtable_for_quality(base, quality)
            n = min(len(table), len(expected))
            total += sum(abs(table[i] - expected[i]) for i in range(n))
            count += n
        if count == 0:
            continue
        score = total / count
        if score < best_score:
            best_score = score
            best_quality = quality
    return best_quality


class ConditionalJPEGCompress:
    def __init__(self, quality: int = 80, recompress_if_jpeg_quality_above: int | None = 80) -> None:
        self.quality = int(quality)
        if self.quality < 1 or self.quality > 100:
            raise ValueError(f"quality must be in [1, 100], got {quality!r}.")
        self.recompress_if_jpeg_quality_above = (
            None if recompress_if_jpeg_quality_above is None else int(recompress_if_jpeg_quality_above)
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        threshold = self.recompress_if_jpeg_quality_above
        if threshold is not None:
            quality = estimate_jpeg_quality(img)
            if quality is not None and quality <= threshold:
                return as_rgb_preserve_jpeg_metadata(img)
        return jpeg_roundtrip(as_rgb_preserve_jpeg_metadata(img), self.quality)


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        out = encoded.convert("RGB").copy()
        out.format = encoded.format
        if hasattr(encoded, "quantization"):
            out.quantization = encoded.quantization
    buffer.close()
    return out


def _jpeg_qtable_for_quality(base: Sequence[int], quality: int) -> list[int]:
    quality = max(1, min(int(quality), 100))
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    return [max(1, min((int(v) * scale + 50) // 100, 255)) for v in base]
