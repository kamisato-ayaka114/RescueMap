from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    gaussian_filter,
    label,
)

from rescuemap.core.background_completion import _restore_texture_after_feather


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class ClaritySummary:
    image_count: int
    selected_images: int
    selected_pixels: int
    strong_texture_pixels: int = 0
    weak_texture_pixels: int = 0


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _blur_mask(rgb: np.ndarray, max_dimension: int = 1100) -> np.ndarray:
    """Find areas whose detail is weak although their neighbourhood is sharp."""
    height, width = rgb.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        reduced = np.asarray(Image.fromarray(rgb).resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        ))
    else:
        reduced = rgb
    gray = reduced.astype(float) @ np.array([0.299, 0.587, 0.114])
    high_frequency = np.abs(gray - gaussian_filter(gray, sigma=1.15, mode="nearest"))
    detail = gaussian_filter(high_frequency, sigma=2.4, mode="nearest")
    context = gaussian_filter(detail, sigma=16.0, mode="nearest")
    floor = max(0.8, float(np.percentile(context, 58)) * 0.65)
    candidate = (context >= floor) & (detail < context * 0.48)
    candidate = binary_opening(candidate, iterations=1)
    candidate = binary_closing(candidate, iterations=3)

    labels, count = label(candidate)
    retained = np.zeros_like(candidate)
    minimum = max(20, int(candidate.size * 0.00018))
    maximum = max(minimum + 1, int(candidate.size * 0.18))
    for component_id in range(1, count + 1):
        component = labels == component_id
        area = int(component.sum())
        if minimum <= area <= maximum:
            retained |= binary_dilation(component, iterations=2)
    if retained.shape != (height, width):
        retained = np.asarray(
            Image.fromarray((retained * 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.NEAREST
            )
        ) > 127
    return retained


def detect_blurry_regions(
    image_directory: Path,
    mask_directory: Path,
    report_path: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ClaritySummary:
    paths = _image_paths(image_directory)
    mask_directory.mkdir(parents=True, exist_ok=True)
    records = []
    selected_images = selected_pixels = 0
    for index, image_path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        relative = image_path.relative_to(image_directory)
        with Image.open(image_path) as opened:
            rgb = np.asarray(opened.convert("RGB")).copy()
        blurry = _blur_mask(rgb)
        target = mask_directory / (str(relative) + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(blurry, 0, 255).astype(np.uint8), "L").save(target)
        labels, component_count = label(blurry)
        pixels = int(blurry.sum())
        selected_images += int(pixels > 0)
        selected_pixels += pixels
        records.append({
            "image": str(relative).replace("\\", "/"),
            "blur_regions": int(component_count),
            "selected_pixels": pixels,
        })
        if progress_callback is not None:
            progress_callback(index, len(paths))
    report_path.write_text(json.dumps({
        "method": "relative_local_detail_deficit",
        "images": records,
        "summary": {
            "image_count": len(paths),
            "selected_images": selected_images,
            "selected_pixels": selected_pixels,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return ClaritySummary(len(paths), selected_images, selected_pixels)


def _weak_texture_clarity(rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    ring = binary_dilation(region, iterations=18) & ~binary_dilation(region, iterations=4)
    if not ring.any():
        return rgb[region].copy()
    samples = rgb[ring].astype(float)
    quantised = np.clip(samples // 16, 0, 15).astype(np.int16)
    codes = quantised[:, 0] * 256 + quantised[:, 1] * 16 + quantised[:, 2]
    dominant_code = int(np.bincount(codes, minlength=4096).argmax())
    dominant = samples[codes == dominant_code].mean(axis=0)
    rgb_float = rgb.astype(float)
    low = gaussian_filter(rgb_float, sigma=(1.15, 1.15, 0.0), mode="nearest")
    residual = np.clip(rgb_float - low, -18.0, 18.0)
    # Stabilise a weak-texture patch toward its surrounding majority while
    # retaining and mildly strengthening any real detail already visible.
    values = 0.80 * rgb_float[region] + 0.20 * dominant + 0.30 * residual[region]
    return np.clip(values, 0, 255).astype(np.uint8)


def clarify_selected_regions(
    image_directory: Path,
    mask_directory: Path,
    backup_directory: Path,
    report_path: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ClaritySummary:
    if not backup_directory.exists():
        shutil.copytree(image_directory, backup_directory)
    paths = _image_paths(image_directory)
    records = []
    selected_images = selected_pixels = strong_pixels = weak_pixels = 0
    for index, image_path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        relative = image_path.relative_to(image_directory)
        mask_path = mask_directory / (str(relative) + ".png")
        with Image.open(image_path) as opened:
            rgb = np.asarray(opened.convert("RGB")).copy()
        if mask_path.is_file():
            with Image.open(mask_path) as opened:
                selected = np.asarray(opened.convert("L")) <= 127
        else:
            selected = np.zeros(rgb.shape[:2], dtype=bool)
        if selected.shape != rgb.shape[:2]:
            selected = np.zeros(rgb.shape[:2], dtype=bool)
        result = rgb.copy()
        labels, component_count = label(selected)
        image_strong = image_weak = 0
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            recovered, strong = _restore_texture_after_feather(result, component)
            if strong:
                result[component] = recovered
                image_strong += int(component.sum())
            else:
                result[component] = _weak_texture_clarity(result, component)
                image_weak += int(component.sum())
        options = {"quality": 95, "subsampling": 0} \
            if image_path.suffix.lower() in {".jpg", ".jpeg"} else {}
        Image.fromarray(result).save(image_path, **options)
        pixels = int(selected.sum())
        selected_images += int(pixels > 0)
        selected_pixels += pixels
        strong_pixels += image_strong
        weak_pixels += image_weak
        records.append({
            "image": str(relative).replace("\\", "/"),
            "selected_pixels": pixels,
            "strong_texture_pixels": image_strong,
            "weak_texture_pixels": image_weak,
        })
        if progress_callback is not None:
            progress_callback(index, len(paths))
    report_path.write_text(json.dumps({
        "method": "human_reviewed_adaptive_clarity",
        "strong_texture": "directional_outer_high_frequency_residual",
        "weak_texture": "dominant_surrounding_colour_plus_bounded_unsharp_residual",
        "images": records,
        "summary": {
            "image_count": len(paths),
            "selected_images": selected_images,
            "selected_pixels": selected_pixels,
            "strong_texture_pixels": strong_pixels,
            "weak_texture_pixels": weak_pixels,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return ClaritySummary(
        len(paths), selected_images, selected_pixels, strong_pixels, weak_pixels
    )
