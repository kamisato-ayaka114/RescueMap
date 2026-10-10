from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label, map_coordinates

from rescuemap.core.dynamic_mask import MAX_IMAGE_ID, _blob_array, _decode_database_text


@dataclass(frozen=True, slots=True)
class CompletionSummary:
    dynamic_pixels: int
    completed_pixels: int
    unresolved_pixels: int
    completed_images: int
    strong_texture_pixels: int
    weak_texture_pixels: int
    report_path: Path


@dataclass(slots=True)
class _PairGeometry:
    other_id: int
    target_points: np.ndarray
    other_points: np.ndarray


def _normalised_homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    def normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centre = points.mean(axis=0)
        scale = np.sqrt(2.0) / max(float(np.mean(np.linalg.norm(points - centre, axis=1))), 1e-9)
        transform = np.array([[scale, 0, -scale * centre[0]],
                              [0, scale, -scale * centre[1]], [0, 0, 1.0]])
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (transform @ homogeneous.T).T[:, :2], transform

    if len(source) < 4:
        raise ValueError("估计投影至少需要4对匹配点。")
    src, source_transform = normalise(np.asarray(source, dtype=float))
    dst, target_transform = normalise(np.asarray(target, dtype=float))
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                     [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    # Four correspondences produce an 8x9 design matrix; the null-space vector
    # is only present when the full right-singular basis is retained.
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=True)
    normalised = vh[-1].reshape(3, 3)
    matrix = np.linalg.inv(target_transform) @ normalised @ source_transform
    return matrix / matrix[2, 2]


def _project_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (matrix @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _robust_homography(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) < 6:
        raise ValueError("局部有效匹配少于6对。")
    rng = np.random.default_rng(20261010)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(min(240, max(80, len(source) * 3))):
        indices = rng.choice(len(source), 4, replace=False)
        try:
            matrix = _normalised_homography(source[indices], target[indices])
            projected = _project_homography(source, matrix)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        error = np.linalg.norm(projected - target, axis=1)
        inliers = np.isfinite(error) & (error < 5.0)
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 6:
        raise ValueError("局部投影没有足够内点。")
    return _normalised_homography(source[best], target[best]), float(best.mean())


def _load_pair_geometry(database_path: Path, neighbor_span: int = 3):
    with sqlite3.connect(database_path) as connection:
        connection.text_factory = bytes
        image_rows = connection.execute("SELECT image_id, name FROM images ORDER BY name").fetchall()
        keypoint_rows = connection.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall()
        pair_rows = connection.execute(
            "SELECT pair_id, rows, cols, data FROM two_view_geometries"
        ).fetchall()
    names = {int(image_id): _decode_database_text(name) for image_id, name in image_rows}
    order = {int(image_id): index for index, (image_id, _name) in enumerate(image_rows)}
    keypoints = {int(image_id): _blob_array(data, np.float32, int(rows), int(cols))[:, :2]
                 for image_id, rows, cols, data in keypoint_rows}
    pairs: dict[int, list[_PairGeometry]] = {image_id: [] for image_id in names}
    for pair_id, rows, cols, data in pair_rows:
        first = int(pair_id) // MAX_IMAGE_ID
        second = int(pair_id) % MAX_IMAGE_ID
        if first not in names or second not in names or abs(order[first] - order[second]) > neighbor_span:
            continue
        matches = _blob_array(data, np.uint32, int(rows), int(cols))
        if len(matches) < 6:
            continue
        first_indices, second_indices = matches[:, 0].astype(int), matches[:, 1].astype(int)
        valid = (first_indices < len(keypoints[first])) & (second_indices < len(keypoints[second]))
        first_points = keypoints[first][first_indices[valid]]
        second_points = keypoints[second][second_indices[valid]]
        if len(first_points) >= 6:
            pairs[first].append(_PairGeometry(second, first_points, second_points))
            pairs[second].append(_PairGeometry(first, second_points, first_points))
    return names, pairs


def _sample_rgb(rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.column_stack([
        map_coordinates(rgb[:, :, channel].astype(float), [points[:, 1], points[:, 0]],
                        order=1, mode="nearest") for channel in range(3)
    ])


def _normalised_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_gray = np.asarray(first, dtype=float) @ np.array([0.299, 0.587, 0.114])
    second_gray = np.asarray(second, dtype=float) @ np.array([0.299, 0.587, 0.114])
    first_gray -= first_gray.mean()
    second_gray -= second_gray.mean()
    denominator = float(np.linalg.norm(first_gray) * np.linalg.norm(second_gray))
    if denominator < 1e-9:
        return 0.0
    return float(np.clip(np.dot(first_gray, second_gray) / denominator, -1.0, 1.0))


def complete_dynamic_backgrounds(
    database_path: Path,
    image_directory: Path,
    mask_directory: Path,
    output_directory: Path,
    *,
    preview_directory: Path,
    report_path: Path,
) -> CompletionSummary:
    """Fill dynamic regions only when two geometrically aligned donor views agree."""
    names, pairs = _load_pair_geometry(database_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    preview_directory.mkdir(parents=True, exist_ok=True)
    total_dynamic = total_completed = completed_images = 0
    total_strong = total_weak = 0
    image_reports = []
    for image_id, relative_name in names.items():
        source_path = image_directory / Path(relative_name)
        mask_path = mask_directory / (relative_name + ".png")
        rgb = np.asarray(Image.open(source_path).convert("RGB")).copy()
        static_mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        dynamic = ~static_mask
        completed = np.zeros_like(dynamic)
        completed_strong = np.zeros_like(dynamic)
        completed_weak = np.zeros_like(dynamic)
        result = rgb.copy()
        gray = rgb.astype(float) @ np.array([0.299, 0.587, 0.114])
        gradient = np.hypot(*np.gradient(gray))
        labels, component_count = label(dynamic)
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            ys, xs = np.where(component)
            if len(xs) < 16:
                continue
            pad = max(80, int(max(xs.max()-xs.min(), ys.max()-ys.min()) * 1.5))
            x0, x1 = xs.min()-pad, xs.max()+pad
            y0, y1 = ys.min()-pad, ys.max()+pad
            target_pixels = np.column_stack((xs, ys)).astype(float)
            ring = binary_dilation(component, iterations=5) & ~dynamic
            ring_y, ring_x = np.where(ring)
            if len(ring_x) > 300:
                chosen = np.linspace(0, len(ring_x)-1, 300).astype(int)
                ring_x, ring_y = ring_x[chosen], ring_y[chosen]
            ring_points = np.column_stack((ring_x, ring_y)).astype(float)
            texture_strength = float(
                np.clip(np.median(gradient[ring_y, ring_x]) / 24.0, 0.0, 1.0)
            ) if len(ring_x) else 0.0
            strong_texture = texture_strength >= 0.35
            candidates = []
            for geometry in pairs.get(image_id, []):
                local = ((geometry.target_points[:, 0] >= x0) & (geometry.target_points[:, 0] <= x1)
                         & (geometry.target_points[:, 1] >= y0) & (geometry.target_points[:, 1] <= y1))
                if local.sum() < 8:
                    continue
                try:
                    matrix, inlier_ratio = _robust_homography(
                        geometry.target_points[local], geometry.other_points[local]
                    )
                except ValueError:
                    continue
                donor_name = names[geometry.other_id]
                donor_rgb = np.asarray(Image.open(image_directory / Path(donor_name)).convert("RGB"))
                donor_static = np.asarray(Image.open(mask_directory / (donor_name + ".png")).convert("L")) > 127
                projected_ring = _project_homography(ring_points, matrix)
                ring_valid = ((projected_ring[:, 0] >= 1) & (projected_ring[:, 0] < donor_rgb.shape[1]-1)
                              & (projected_ring[:, 1] >= 1) & (projected_ring[:, 1] < donor_rgb.shape[0]-1))
                ring_ix = np.clip(np.rint(projected_ring[:, 0]).astype(int), 0, donor_rgb.shape[1]-1)
                ring_iy = np.clip(np.rint(projected_ring[:, 1]).astype(int), 0, donor_rgb.shape[0]-1)
                ring_valid &= donor_static[ring_iy, ring_ix]
                if ring_valid.sum() < 12:
                    continue
                donor_ring = _sample_rgb(donor_rgb, projected_ring[ring_valid])
                target_ring = rgb[ring_y[ring_valid], ring_x[ring_valid]].astype(float)
                offset = np.median(target_ring - donor_ring, axis=0)
                boundary_error = float(np.median(np.abs(target_ring - (donor_ring + offset))))
                if boundary_error > 32.0:
                    continue
                texture_correlation = _normalised_correlation(target_ring, donor_ring + offset)
                if strong_texture and texture_correlation < 0.45:
                    continue
                if not strong_texture and inlier_ratio < 0.55:
                    continue
                projected = _project_homography(target_pixels, matrix)
                valid = ((projected[:, 0] >= 1) & (projected[:, 0] < donor_rgb.shape[1]-1)
                         & (projected[:, 1] >= 1) & (projected[:, 1] < donor_rgb.shape[0]-1))
                ix = np.clip(np.rint(projected[:, 0]).astype(int), 0, donor_rgb.shape[1]-1)
                iy = np.clip(np.rint(projected[:, 1]).astype(int), 0, donor_rgb.shape[0]-1)
                valid &= donor_static[iy, ix]
                colours = np.full((len(target_pixels), 3), np.nan)
                colours[valid] = np.clip(_sample_rgb(donor_rgb, projected[valid]) + offset, 0, 255)
                candidates.append((colours, inlier_ratio, boundary_error, texture_correlation))
            if len(candidates) < 2:
                continue
            stack = np.stack([candidate[0] for candidate in candidates])
            support = np.sum(np.isfinite(stack[:, :, 0]), axis=0)
            median = np.full((len(target_pixels), 3), np.nan)
            dispersion = np.full(len(target_pixels), np.inf)
            has_candidate = support > 0
            median[has_candidate] = np.nanmedian(stack[:, has_candidate, :], axis=0)
            dispersion[has_candidate] = np.nanmedian(
                np.abs(stack[:, has_candidate, :] - median[None, has_candidate, :]),
                axis=(0, 2),
            )
            if strong_texture:
                # Strong-texture regions extend only texture that remains
                # correlated on the observed boundary in multiple views.
                accepted = ((support >= 3) & (dispersion < 18.0)) | (
                    (support == 2) & (dispersion < 10.0)
                )
            else:
                # Weak texture cannot provide a meaningful NCC decision.  It
                # therefore relies on projection inliers and tighter colour
                # agreement between independently projected donor views.
                accepted = ((support >= 3) & (dispersion < 13.0)) | (
                    (support == 2) & (dispersion < 7.0)
                )
            result[ys[accepted], xs[accepted]] = median[accepted].astype(np.uint8)
            completed[ys[accepted], xs[accepted]] = True
            if strong_texture:
                completed_strong[ys[accepted], xs[accepted]] = True
            else:
                completed_weak[ys[accepted], xs[accepted]] = True
        unresolved = dynamic & ~completed
        result[unresolved] = 0
        updated_mask = np.where(unresolved, 0, 255).astype(np.uint8)
        target = output_directory / Path(relative_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        options = {"quality": 95, "subsampling": 0} if target.suffix.lower() in {".jpg", ".jpeg"} else {}
        Image.fromarray(result).save(target, **options)
        Image.fromarray(updated_mask).save(mask_path)
        preview = rgb.copy()
        preview[completed_strong] = (
            0.35 * preview[completed_strong] + 0.65 * np.array([30, 220, 80])
        ).astype(np.uint8)
        preview[completed_weak] = (
            0.35 * preview[completed_weak] + 0.65 * np.array([40, 180, 245])
        ).astype(np.uint8)
        preview[unresolved] = (0.25 * preview[unresolved] + 0.75 * np.array([240, 50, 50])).astype(np.uint8)
        preview_target = preview_directory / Path(relative_name)
        preview_target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(preview_target, quality=92)
        dynamic_count, completed_count = int(dynamic.sum()), int(completed.sum())
        total_dynamic += dynamic_count
        total_completed += completed_count
        strong_count = int(completed_strong.sum())
        weak_count = int(completed_weak.sum())
        total_strong += strong_count
        total_weak += weak_count
        completed_images += int(completed_count > 0)
        image_reports.append({"image": relative_name, "dynamic_pixels": dynamic_count,
                              "completed_pixels": completed_count,
                              "strong_texture_completed_pixels": strong_count,
                              "weak_texture_completed_pixels": weak_count,
                              "unresolved_pixels": dynamic_count-completed_count,
                              "completion_ratio": completed_count/max(dynamic_count, 1)})
    report_path.write_text(json.dumps({"method": "local_multiview_background_completion",
        "summary": {"dynamic_pixels": total_dynamic, "completed_pixels": total_completed,
                    "strong_texture_completed_pixels": total_strong,
                    "weak_texture_completed_pixels": total_weak,
                    "unresolved_pixels": total_dynamic-total_completed,
                    "completion_ratio": total_completed/max(total_dynamic, 1)},
        "images": image_reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    return CompletionSummary(total_dynamic, total_completed, total_dynamic-total_completed,
                             completed_images, total_strong, total_weak, report_path)
