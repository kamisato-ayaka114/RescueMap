from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
)

from rescuemap.core.dynamic_mask import MAX_IMAGE_ID, _blob_array, _decode_database_text


@dataclass(frozen=True, slots=True)
class CompletionSummary:
    dynamic_pixels: int
    completed_pixels: int
    unresolved_pixels: int
    completed_images: int
    strong_texture_pixels: int
    weak_texture_pixels: int
    local_texture_pixels: int
    small_hole_pixels: int
    near_side_pixels: int
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
    # Local RANSAC used to run as many as 240 SVD fits for every dynamic
    # component and every neighbouring frame.  A deterministic, spatially
    # distributed subset and 64 trials retain the useful outlier rejection
    # while making a 70-frame review practical on an integrated GPU laptop.
    if len(source) > 120:
        keep = np.linspace(0, len(source) - 1, 120).astype(int)
        source = source[keep]
        target = target[keep]
    rng = np.random.default_rng(20261010)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(min(64, max(32, len(source)))):
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


class _FrameCache:
    """Small LRU cache for neighbouring RGB frames and masks.

    A frame was previously decoded again for every candidate component.  The
    sequential matcher only uses nearby images, so a bounded cache removes the
    repeated disk/JPEG cost without retaining the complete dataset in memory.
    """

    def __init__(self, image_directory: Path, mask_directory: Path) -> None:
        self.image_directory = image_directory
        self.mask_directory = mask_directory

    @lru_cache(maxsize=10)
    def rgb(self, relative_name: str) -> np.ndarray:
        with Image.open(self.image_directory / Path(relative_name)) as opened:
            return np.asarray(opened.convert("RGB")).copy()

    @lru_cache(maxsize=10)
    def static(self, relative_name: str) -> np.ndarray:
        with Image.open(self.mask_directory / (relative_name + ".png")) as opened:
            return np.asarray(opened.convert("L")) > 127


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


def _directional_boundary_fill(
    rgb: np.ndarray,
    hole: np.ndarray,
    *,
    use_median: bool = False,
) -> np.ndarray:
    """Extend visible boundary colours into a hole from surrounding directions.

    Each directional pass transports the first visible pixel along its ray.
    Blending several opposing rays prevents the one-sided smearing produced by
    a single scan direction.  Medium holes use a texture-orientation weighted
    mean; tiny holes use the per-channel median as a robust local majority.
    """
    rgb = np.asarray(rgb, dtype=float)
    hole = np.asarray(hole, dtype=bool)
    if not hole.any():
        return np.empty((0, 3), dtype=np.uint8)

    ys, xs = np.where(hole)
    margin = max(8, min(48, int(np.sqrt(len(xs))) + 4))
    y0, y1 = max(0, ys.min() - margin), min(hole.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(hole.shape[1], xs.max() + margin + 1)
    local_hole = hole[y0:y1, x0:x1]
    local_rgb = rgb[y0:y1, x0:x1]
    height, width = local_hole.shape

    ring = binary_dilation(local_hole, iterations=min(5, margin)) & ~local_hole
    gray = local_rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    if ring.any():
        structure = np.array(
            [
                [np.mean(gx[ring] ** 2), np.mean(gx[ring] * gy[ring])],
                [np.mean(gx[ring] * gy[ring]), np.mean(gy[ring] ** 2)],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(structure)
        texture_direction = eigenvectors[:, int(np.argmin(eigenvalues))]
    else:
        texture_direction = np.array([1.0, 0.0])

    known = ~local_hole
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    candidates: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def add_candidate(source_y, source_x, valid, distance, direction) -> None:
        candidate = local_rgb[
            np.clip(source_y, 0, height - 1),
            np.clip(source_x, 0, width - 1),
        ]
        unit = np.asarray(direction, dtype=float)
        unit /= max(float(np.linalg.norm(unit)), 1e-9)
        alignment = 0.35 + 0.65 * abs(float(np.dot(unit, texture_direction)))
        weight = np.zeros((height, width), dtype=float)
        accepted = local_hole & valid
        weight[accepted] = alignment / (1.0 + distance[accepted])
        candidates.append(candidate)
        weights.append(weight)

    left_x = np.maximum.accumulate(np.where(known, columns, -1), axis=1)
    add_candidate(rows, left_x, left_x >= 0, columns - left_x, (1, 0))

    reverse_x = np.maximum.accumulate(
        np.where(known[:, ::-1], columns, -1), axis=1
    )
    right_x = width - 1 - reverse_x[:, ::-1]
    right_valid = reverse_x[:, ::-1] >= 0
    add_candidate(rows, right_x, right_valid, right_x - columns, (-1, 0))

    top_y = np.maximum.accumulate(np.where(known, rows, -1), axis=0)
    add_candidate(top_y, columns, top_y >= 0, rows - top_y, (0, 1))

    reverse_y = np.maximum.accumulate(
        np.where(known[::-1, :], rows, -1), axis=0
    )
    bottom_y = height - 1 - reverse_y[::-1, :]
    bottom_valid = reverse_y[::-1, :] >= 0
    add_candidate(bottom_y, columns, bottom_valid, bottom_y - rows, (0, -1))

    nearest_distance, nearest = distance_transform_edt(
        local_hole, return_indices=True
    )
    add_candidate(
        nearest[0], nearest[1], np.ones_like(local_hole), nearest_distance, (1, 1)
    )

    stack = np.stack(candidates)
    weight_stack = np.stack(weights)
    if use_median:
        masked = np.where(weight_stack[:, :, :, None] > 0, stack, np.nan)
        filled = np.nanmedian(masked, axis=0)
    else:
        denominator = weight_stack.sum(axis=0)
        filled = np.divide(
            (stack * weight_stack[:, :, :, None]).sum(axis=0),
            denominator[:, :, None],
            out=np.zeros_like(local_rgb),
            where=denominator[:, :, None] > 0,
        )
    invalid = ~np.isfinite(filled).all(axis=2)
    if not use_median:
        invalid |= weight_stack.sum(axis=0) <= 0
    if invalid.any():
        _, nearest = distance_transform_edt(local_hole, return_indices=True)
        nearest_values = local_rgb[nearest[0], nearest[1]]
        filled[invalid] = nearest_values[invalid]
    local_y, local_x = np.where(local_hole)
    return np.clip(filled[local_y, local_x], 0, 255).astype(np.uint8)


def _texture_adaptive_boundary_fill(
    rgb: np.ndarray, hole: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Fill a selected region from its complete two-dimensional boundary.

    The low-frequency colour field is obtained by a discrete harmonic-style
    diffusion with every visible boundary pixel held fixed.  For a textured
    boundary, only the high-frequency residual is transported along the local
    structure direction; this avoids the large vertical colour bars produced
    by extending raw RGB columns.  A weak-texture boundary instead contributes
    its dominant quantised colour, which is robust to isolated lamps and cars.
    """
    rgb_float = np.asarray(rgb, dtype=float)
    hole = np.asarray(hole, dtype=bool)
    ys, xs = np.where(hole)
    if not len(xs):
        return np.empty((0, 3), dtype=np.uint8), False

    margin = max(12, min(64, int(np.sqrt(len(xs))) // 2 + 8))
    y0, y1 = max(0, ys.min() - margin), min(hole.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(hole.shape[1], xs.max() + margin + 1)
    local_hole = hole[y0:y1, x0:x1]
    local_rgb = rgb_float[y0:y1, x0:x1]
    # References are intentionally sampled farther outside the selected box.
    # The near seam may already be blurred or black after the first stitching
    # round, while the 8-12 pixel outer ring usually still contains reliable
    # background colour and texture direction.
    ring_width = min(12, margin)
    ring = binary_dilation(local_hole, iterations=ring_width) & ~local_hole
    near_ring = binary_dilation(local_hole, iterations=min(4, margin)) & ~local_hole

    gray = local_rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    texture_strength = (
        float(np.percentile(np.hypot(gx[near_ring], gy[near_ring]), 65))
        if near_ring.any() else 0.0
    )
    strong_texture = texture_strength >= 8.0

    # Nearest-boundary initialisation followed by fixed-boundary diffusion is
    # a fast approximation to a Laplace solution with full 2-D constraints.
    _, nearest = distance_transform_edt(local_hole, return_indices=True)
    work = local_rgb.copy()
    work[local_hole] = local_rgb[nearest[0][local_hole], nearest[1][local_hole]]
    iterations = min(48, max(12, int(np.sqrt(int(local_hole.sum()))) // 5))
    for _ in range(iterations):
        smoothed = gaussian_filter(work, sigma=(1.05, 1.05, 0.0), mode="nearest")
        work[local_hole] = smoothed[local_hole]

    def extended_residual(expansion: int) -> np.ndarray:
        reference_hole = binary_dilation(local_hole, iterations=expansion)
        low_frequency = gaussian_filter(
            local_rgb, sigma=(2.0, 2.0, 0.0), mode="nearest"
        )
        shifted = np.clip(local_rgb - low_frequency + 128.0, 0, 255)
        propagated = _directional_boundary_fill(
            shifted.astype(np.uint8), reference_hole
        ).astype(float) - 128.0
        field = np.zeros_like(local_rgb)
        field[reference_hole] = propagated
        return field[local_hole]

    if strong_texture:
        residual_values = extended_residual(3)
        ly, lx = np.where(local_hole)
        texture_weight = float(np.clip(texture_strength / 42.0, 0.25, 0.72))
        work[ly, lx] += texture_weight * residual_values
    elif ring.any():
        # "Most common surrounding pixel" is estimated in 16-level colour
        # bins so sensor noise does not make every RGB triplet unique.
        samples = local_rgb[near_ring if near_ring.any() else ring]
        quantised = np.clip(samples // 16, 0, 15).astype(np.int16)
        codes = quantised[:, 0] * 256 + quantised[:, 1] * 16 + quantised[:, 2]
        dominant = int(np.bincount(codes, minlength=4096).argmax())
        dominant_colour = samples[codes == dominant].mean(axis=0)
        work[local_hole] = 0.78 * work[local_hole] + 0.22 * dominant_colour

        # After the weak area has received its stable colour field, borrow a
        # low-amplitude, directionally coherent residual from farther outside.
        # This supplies COLMAP with real neighbouring texture without turning
        # lamp noise or a single edge into a fabricated high-contrast stripe.
        outer = (
            binary_dilation(local_hole, iterations=min(16, margin))
            & ~binary_dilation(local_hole, iterations=min(4, margin))
        )
        if outer.any():
            outer_gradient = np.hypot(gx[outer], gy[outer])
            outer_strength = float(np.percentile(outer_gradient, 70))
            structure = np.array([
                [np.mean(gx[outer] ** 2), np.mean(gx[outer] * gy[outer])],
                [np.mean(gx[outer] * gy[outer]), np.mean(gy[outer] ** 2)],
            ])
            eigenvalues = np.linalg.eigvalsh(structure)
            coherence = float(
                (eigenvalues[-1] - eigenvalues[0])
                / max(eigenvalues[-1] + eigenvalues[0], 1e-9)
            )
            if outer_strength >= 7.0 and coherence >= 0.28:
                work[local_hole] += 0.18 * extended_residual(5)

    ly, lx = np.where(local_hole)
    return np.clip(work[ly, lx], 0, 255).astype(np.uint8), strong_texture


def _restore_texture_after_feather(
    rgb: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Re-inject reliable outer texture after Gaussian seam feathering.

    Feathering is deliberately a low-pass operation: it removes a rectangular
    seam, but repeated refinement can also erase road markings and surface
    texture.  This stage keeps the completed low-frequency colour field and
    transports only the zero-mean high-frequency residual from a farther,
    strongly textured outer ring.  Therefore an object colour or silhouette is
    not copied back wholesale into the repaired region.
    """
    rgb_float = np.asarray(rgb, dtype=float)
    region = np.asarray(region, dtype=bool)
    ys, xs = np.where(region)
    if not len(xs):
        return np.empty((0, 3), dtype=np.uint8), False

    margin = max(24, min(96, int(np.sqrt(len(xs))) // 2 + 18))
    y0, y1 = max(0, ys.min() - margin), min(region.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(region.shape[1], xs.max() + margin + 1)
    local_region = region[y0:y1, x0:x1]
    local_rgb = rgb_float[y0:y1, x0:x1]

    inner_guard = binary_dilation(local_region, iterations=min(5, margin))
    outer_limit = binary_dilation(local_region, iterations=min(22, margin))
    outer_ring = outer_limit & ~inner_guard
    if int(outer_ring.sum()) < 24:
        return np.clip(local_rgb[local_region], 0, 255).astype(np.uint8), False

    gray = local_rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    gradient = np.hypot(gx, gy)
    outer_strength = float(np.percentile(gradient[outer_ring], 72))
    inner_strength = (
        float(np.percentile(gradient[local_region], 72))
        if int(local_region.sum()) >= 8 else 0.0
    )
    # Flat surroundings cannot provide meaningful texture.  If the repaired
    # area is already at least as sharp as its surroundings, adding residuals
    # would merely amplify noise.
    if outer_strength < 7.0 or inner_strength >= outer_strength * 0.92:
        return np.clip(local_rgb[local_region], 0, 255).astype(np.uint8), False

    source_threshold = max(6.0, float(np.percentile(gradient[outer_ring], 62)))
    strong_sources = outer_ring & (gradient >= source_threshold)
    if int(strong_sources.sum()) < 16:
        return np.clip(local_rgb[local_region], 0, 255).astype(np.uint8), False

    low_frequency = gaussian_filter(local_rgb, sigma=(2.2, 2.2, 0.0), mode="nearest")
    residual = local_rgb - low_frequency
    # Extend the closest reliable strong-texture residual through the context
    # band first.  The existing directional boundary transport then combines
    # opposing directions instead of copying one raw RGB column across a box.
    _, nearest_strong = distance_transform_edt(~strong_sources, return_indices=True)
    reliable_residual = residual[nearest_strong[0], nearest_strong[1]]
    shifted = np.clip(reliable_residual + 128.0, 0, 255).astype(np.uint8)
    transported = (
        _directional_boundary_fill(shifted, local_region).astype(float) - 128.0
    )
    # Remove any DC component: colour and illumination must still come from
    # the completed image, while only texture is restored here.
    transported -= np.median(transported, axis=0, keepdims=True)
    transported = np.clip(transported, -32.0, 32.0)

    deficit = float(
        np.clip((outer_strength - inner_strength) / max(outer_strength, 1e-9), 0.0, 1.0)
    )
    gain = float(np.clip(0.26 + 0.40 * deficit, 0.28, 0.62))
    values = local_rgb[local_region] + gain * transported
    return np.clip(values, 0, 255).astype(np.uint8), True


def _near_side_ground_fill(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill from the lower/near-side boundary for forward-looking images."""
    rgb = np.asarray(rgb, dtype=float)
    hole = np.asarray(hole, dtype=bool)
    ys, xs = np.where(hole)
    if not len(xs):
        return np.empty((0, 3), dtype=np.uint8)
    ring = binary_dilation(hole, iterations=8) & ~hole
    ring_y, ring_x = np.where(ring)
    if not len(ring_x):
        return _directional_boundary_fill(rgb, hole)
    near_cut = np.quantile(ring_y, 0.62)
    near_ring = ring & (np.indices(ring.shape)[0] >= near_cut)
    samples = rgb[near_ring]
    if len(samples) < 12:
        samples = rgb[ring]
    low, high = np.percentile(samples, (10, 90), axis=0)
    trimmed = samples[np.all((samples >= low) & (samples <= high), axis=1)]
    robust_mean = (trimmed if len(trimmed) >= 8 else samples).mean(axis=0)

    height, width = hole.shape
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]
    known = ~hole
    reverse_y = np.maximum.accumulate(
        np.where(known[::-1, :], rows, -1), axis=0
    )
    source_y = height - 1 - reverse_y[::-1, :]
    valid = reverse_y[::-1, :] >= 0
    local = np.broadcast_to(robust_mean, rgb.shape).copy()
    local[valid] = rgb[source_y[valid], np.broadcast_to(columns, hole.shape)[valid]]
    # The lower boundary carries the near-side texture; mixing its ray colour
    # with a trimmed local mean reduces car-light streaks and exposure jumps.
    local = 0.68 * local + 0.32 * robust_mean
    values = local[ys, xs]
    missing = ~valid[ys, xs]
    if missing.any():
        values[missing] = _directional_boundary_fill(rgb, hole)[missing]
    return np.clip(values, 0, 255).astype(np.uint8)


def complete_dynamic_backgrounds(
    database_path: Path,
    image_directory: Path,
    mask_directory: Path,
    output_directory: Path,
    *,
    preview_directory: Path,
    report_path: Path,
    mode: str = "multiview",
    progress_callback: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CompletionSummary:
    """Restore dynamic regions by multiview stitching or manual refinement."""
    if mode not in {"multiview", "refine"}:
        raise ValueError(f"未知的背景恢复模式：{mode}")
    names, pairs = _load_pair_geometry(database_path)
    cache = _FrameCache(image_directory, mask_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    preview_directory.mkdir(parents=True, exist_ok=True)
    total_dynamic = total_completed = completed_images = 0
    total_strong = total_weak = 0
    total_local_texture = total_small_hole = total_near_side = 0
    image_reports = []
    total_images = len(names)
    for image_number, (image_id, relative_name) in enumerate(names.items(), start=1):
        if should_cancel is not None and should_cancel():
            break
        mask_path = mask_directory / (relative_name + ".png")
        rgb = cache.rgb(relative_name).copy()
        static_mask = cache.static(relative_name)
        # Manual refinement honours the selected rectangle exactly.  Automatic
        # dynamic masks receive a tiny closing operation but are no longer
        # enlarged, because expansion was erasing good boundary texture.
        dynamic = (
            ~static_mask
            if mode == "refine"
            else binary_closing(~static_mask, iterations=1)
        )
        completed = np.zeros_like(dynamic)
        completed_strong = np.zeros_like(dynamic)
        completed_weak = np.zeros_like(dynamic)
        completed_local_texture = np.zeros_like(dynamic)
        completed_small_hole = np.zeros_like(dynamic)
        completed_near_side = np.zeros_like(dynamic)
        result = rgb.copy()
        gray = rgb.astype(float) @ np.array([0.299, 0.587, 0.114])
        gradient = np.hypot(*np.gradient(gray))
        labels, component_count = label(dynamic)
        if mode == "refine":
            for component_id in range(1, component_count + 1):
                component = labels == component_id
                ys, xs = np.where(component)
                if not len(xs):
                    continue
                values, strong_texture = _texture_adaptive_boundary_fill(result, component)
                result[ys, xs] = values
                completed[ys, xs] = True
                if strong_texture:
                    completed_local_texture[ys, xs] = True
                else:
                    completed_small_hole[ys, xs] = True

            # Feather three pixels to both sides of the chosen boundary.  The
            # slight outward expansion removes a visible box edge and samples
            # reliable static background just outside the manual selection,
            # while the region interior keeps its reconstructed texture.
            seam = (
                binary_dilation(completed, iterations=3)
                & ~binary_erosion(completed, iterations=3)
            )
            if seam.any():
                smoothed = gaussian_filter(result.astype(float), sigma=(1.0, 1.0, 0.0))
                result[seam] = np.clip(
                    0.62 * result[seam] + 0.38 * smoothed[seam], 0, 255
                ).astype(np.uint8)

            # Every completion pass is followed by one restoration pass.
            # Gaussian feathering removes the seam; the second stage restores
            # directional detail only where the selected area is measurably
            # blurrier than reliable strong texture farther outside it.
            for component_id in range(1, component_count + 1):
                component = labels == component_id
                restored, texture_recovered = _restore_texture_after_feather(
                    result, component
                )
                if texture_recovered:
                    result[component] = restored
                    completed_local_texture[component] = True

        if mode == "multiview":
            # First pass intentionally retains the earlier conservative
            # behaviour: only geometrically verified neighbouring views may
            # replace a moving object.  Local synthesis is deferred until the
            # user explicitly selects a region in the second review.
            pass
        for component_id in (range(1, component_count + 1) if mode == "multiview" else ()):
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
                donor_rgb = cache.rgb(donor_name)
                donor_static = cache.static(donor_name)
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
        if mode == "multiview":
            result[unresolved] = 0
        updated_mask = np.where(unresolved, 0, 255).astype(np.uint8)
        target = output_directory / Path(relative_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        options = {"quality": 95, "subsampling": 0} if target.suffix.lower() in {".jpg", ".jpeg"} else {}
        Image.fromarray(result).save(target, **options)
        if mode == "multiview":
            Image.fromarray(updated_mask).save(mask_path)
        preview = rgb.copy()
        preview[completed_strong] = (
            0.35 * preview[completed_strong] + 0.65 * np.array([30, 220, 80])
        ).astype(np.uint8)
        preview[completed_weak] = (
            0.35 * preview[completed_weak] + 0.65 * np.array([40, 180, 245])
        ).astype(np.uint8)
        preview[completed_local_texture] = (
            0.35 * preview[completed_local_texture] + 0.65 * np.array([245, 170, 35])
        ).astype(np.uint8)
        preview[completed_small_hole] = (
            0.35 * preview[completed_small_hole] + 0.65 * np.array([180, 90, 230])
        ).astype(np.uint8)
        preview[completed_near_side] = (
            0.35 * preview[completed_near_side] + 0.65 * np.array([40, 220, 200])
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
        local_texture_count = int(completed_local_texture.sum())
        small_hole_count = int(completed_small_hole.sum())
        total_local_texture += local_texture_count
        total_small_hole += small_hole_count
        near_side_count = int(completed_near_side.sum())
        total_near_side += near_side_count
        completed_images += int(completed_count > 0)
        image_reports.append({"image": relative_name, "dynamic_pixels": dynamic_count,
                              "completed_pixels": completed_count,
                              "strong_texture_completed_pixels": strong_count,
                              "weak_texture_completed_pixels": weak_count,
                              "local_texture_completed_pixels": local_texture_count,
                              "small_hole_completed_pixels": small_hole_count,
                              "near_side_completed_pixels": near_side_count,
                              "unresolved_pixels": dynamic_count-completed_count,
                              "completion_ratio": completed_count/max(dynamic_count, 1)})
        if progress_callback is not None:
            progress_callback(image_number, total_images)
    report_path.write_text(json.dumps({
        "method": (
            "conservative_multiview_stitching"
            if mode == "multiview"
            else "manual_texture_adaptive_boundary_refinement"
        ),
        "refinement_policy": {
            "strong_texture": "harmonic_colour_field_plus_directional_high_frequency_residual",
            "weak_texture": "dominant_colour_then_coherent_outer_texture_residual",
            "seam": "three_pixel_two_sided_gaussian_feather",
            "post_feather": "outer_strong_texture_directional_residual_recovery",
        } if mode == "refine" else None,
        "summary": {"dynamic_pixels": total_dynamic, "completed_pixels": total_completed,
                    "strong_texture_completed_pixels": total_strong,
                    "weak_texture_completed_pixels": total_weak,
                    "local_texture_completed_pixels": total_local_texture,
                    "small_hole_completed_pixels": total_small_hole,
                    "near_side_completed_pixels": total_near_side,
                    "unresolved_pixels": total_dynamic-total_completed,
                    "completion_ratio": total_completed/max(total_dynamic, 1)},
        "images": image_reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    return CompletionSummary(total_dynamic, total_completed, total_dynamic-total_completed,
                             completed_images, total_strong, total_weak,
                             total_local_texture, total_small_hole,
                             total_near_side, report_path)
