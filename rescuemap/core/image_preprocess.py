from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class EnhancementSummary:
    image_count: int
    low_percentile: float
    high_percentile: float
    output_directory: Path


def enhance_low_light_dataset(
    image_directory: Path,
    output_directory: Path,
    *,
    gamma: float = 0.86,
) -> EnhancementSummary:
    """Apply one dataset-wide luminance transform without modifying originals.

    A shared percentile range avoids the frame-to-frame brightness jumps caused
    by independently auto-contrasting every image.  Chroma is retained by
    scaling RGB with the luminance ratio, which is friendlier to feature
    matching than processing the three channels independently.
    """
    paths = sorted(
        path for path in image_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError("照片目录中没有可增强的图片。")

    samples: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as opened:
            small = ImageOps.exif_transpose(opened).convert("RGB")
            small.thumbnail((480, 480), Image.Resampling.BILINEAR)
            rgb = np.asarray(small, dtype=np.float32)
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        samples.append(luminance.ravel()[::8])
    combined = np.concatenate(samples)
    low, high = np.percentile(combined, (1.0, 99.5))
    if high - low < 12.0:
        low, high = 0.0, 255.0

    output_directory.mkdir(parents=True, exist_ok=True)
    for path in paths:
        relative = path.relative_to(image_directory)
        target = output_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(image, dtype=np.float32)
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        stretched = np.clip((luminance - low) / max(high - low, 1.0), 0.0, 1.0)
        enhanced_luminance = np.power(stretched, gamma) * 255.0
        ratio = enhanced_luminance / np.maximum(luminance, 4.0)
        enhanced = np.clip(rgb * ratio[:, :, None], 0.0, 255.0).astype(np.uint8)
        options = {"quality": 95, "subsampling": 0} if target.suffix.lower() in {".jpg", ".jpeg"} else {}
        Image.fromarray(enhanced).save(target, **options)
    return EnhancementSummary(len(paths), float(low), float(high), output_directory)
