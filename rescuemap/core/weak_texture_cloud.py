from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import pyvista as pv
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

from rescuemap.core.candidate_validation import (
    locate_dense_colmap_context,
    project_points,
    read_colmap_binary_model,
)


@dataclass(frozen=True, slots=True)
class CloudCompletionSummary:
    original_points: int
    predicted_points: int
    combined_path: Path
    predicted_path: Path
    report_path: Path


def _rgb_values(data: pv.DataSet) -> np.ndarray:
    for name in ("RGB", "rgb", "Colors", "colors"):
        values = data.point_data.get(name)
        if values is not None and values.ndim == 2 and values.shape[1] >= 3:
            return np.clip(values[:, :3], 0, 255).astype(np.uint8)
    return np.full((data.n_points, 3), 190, dtype=np.uint8)


def _save_cloud(path: Path, points: np.ndarray, rgb: np.ndarray, source: np.ndarray) -> None:
    cloud = pv.PolyData(np.asarray(points, dtype=np.float32))
    cloud.point_data["RGB"] = np.asarray(rgb, dtype=np.uint8)
    cloud.point_data["PointSource"] = np.asarray(source, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".ply":
        cloud.save(path, binary=True, texture="RGB")
    else:
        cloud.save(path, binary=True)


def complete_weak_texture_cloud(
    fused_path: Path,
    *,
    output_directory: Path | None = None,
    max_views: int = 12,
    grid_step: int = 10,
    max_reference_points: int = 280_000,
    max_predicted_points: int = 240_000,
    progress_callback: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CloudCompletionSummary:
    """Create a separately labelled low-frequency point prediction layer.

    Only projection-grid holes surrounded by existing points and exhibiting a
    locally continuous depth field are accepted.  The method is conservative:
    it does not extrapolate silhouettes or fill unbounded image background.
    """
    fused_path = Path(fused_path).resolve()
    context = locate_dense_colmap_context(fused_path)
    if context is None:
        raise FileNotFoundError(
            "点云旁缺少 dense/sparse 相机模型或 dense/images 校正图片，无法投影补全。"
        )
    model_directory, image_directory = context
    cameras, registered_images = read_colmap_binary_model(model_directory)
    if not registered_images:
        raise ValueError("COLMAP模型中没有已注册图片。")
    source_cloud = pv.read(fused_path)
    original_points = np.asarray(source_cloud.points, dtype=float)
    if len(original_points) < 100:
        raise ValueError("点云数量过少，无法估计连续表面。")
    original_rgb = _rgb_values(source_cloud)
    output_directory = output_directory or fused_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261011)
    if len(original_points) > max_reference_points:
        reference_indices = np.sort(
            rng.choice(len(original_points), max_reference_points, replace=False)
        )
    else:
        reference_indices = np.arange(len(original_points))
    reference = original_points[reference_indices]
    spacing_sample = reference
    if len(spacing_sample) > 20_000:
        spacing_sample = spacing_sample[
            rng.choice(len(spacing_sample), 20_000, replace=False)
        ]
    spacing_tree = cKDTree(spacing_sample)
    distances, _ = spacing_tree.query(spacing_sample, k=2)
    spacing = max(float(np.median(distances[:, 1])), 1e-8)

    view_indices = np.linspace(
        0, len(registered_images) - 1, min(max_views, len(registered_images))
    ).round().astype(int)
    views = [registered_images[index] for index in np.unique(view_indices)]
    predicted_points: list[np.ndarray] = []
    predicted_colours: list[np.ndarray] = []
    predicted_confidence: list[np.ndarray] = []
    view_reports = []
    for view_number, image in enumerate(views, start=1):
        if should_cancel is not None and should_cancel():
            break
        camera = cameras.get(image.camera_id)
        image_path = image_directory / Path(image.name)
        if camera is None or not image_path.is_file():
            continue
        uv, depth = project_points(reference, camera, image)
        valid = (
            np.isfinite(uv).all(axis=1) & (depth > 0)
            & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
        )
        if int(valid.sum()) < 100:
            continue
        grid_width = int(np.ceil(camera.width / grid_step))
        grid_height = int(np.ceil(camera.height / grid_step))
        gx = np.clip((uv[valid, 0] / grid_step).astype(int), 0, grid_width - 1)
        gy = np.clip((uv[valid, 1] / grid_step).astype(int), 0, grid_height - 1)
        flat_index = gy * grid_width + gx
        depth_grid = np.full(grid_height * grid_width, np.inf, dtype=float)
        np.minimum.at(depth_grid, flat_index, depth[valid])
        depth_grid = depth_grid.reshape(grid_height, grid_width)
        occupied = np.isfinite(depth_grid)
        support = uniform_filter(occupied.astype(float), size=9, mode="constant")
        enclosed = binary_closing(occupied, iterations=3)
        nearby = binary_dilation(occupied, iterations=4)
        holes = ~occupied & nearby & (enclosed | (support >= 0.12))

        weighted_depth = gaussian_filter(
            np.where(occupied, depth_grid, 0.0), sigma=2.0, mode="nearest"
        )
        weight = gaussian_filter(occupied.astype(float), sigma=2.0, mode="nearest")
        estimated_depth = np.divide(
            weighted_depth, weight,
            out=np.zeros_like(weighted_depth), where=weight > 1e-6,
        )
        second = gaussian_filter(
            np.where(occupied, depth_grid ** 2, 0.0), sigma=2.0, mode="nearest"
        )
        variance = np.maximum(
            np.divide(second, weight, out=np.zeros_like(second), where=weight > 1e-6)
            - estimated_depth ** 2,
            0.0,
        )
        relative_spread = np.sqrt(variance) / np.maximum(estimated_depth, 1e-6)
        holes &= (weight >= 0.055) & (relative_spread <= 0.10)
        hy, hx = np.where(holes)
        if len(hx) > 24_000:
            keep = rng.choice(len(hx), 24_000, replace=False)
            hy, hx = hy[keep], hx[keep]
        if not len(hx):
            continue
        z = estimated_depth[hy, hx]
        pixel_x = np.minimum((hx + 0.5) * grid_step, camera.width - 1)
        pixel_y = np.minimum((hy + 0.5) * grid_step, camera.height - 1)
        fx, fy, cx, cy = camera.pinhole_parameters()
        camera_points = np.column_stack((
            (pixel_x - cx) * z / fx,
            (pixel_y - cy) * z / fy,
            z,
        ))
        world = (image.rotation.T @ (camera_points - image.translation).T).T

        with Image.open(image_path) as opened:
            colour_image = np.asarray(opened.convert("RGB")).astype(float)
        if colour_image.shape[:2] != (camera.height, camera.width):
            colour_image = np.asarray(Image.fromarray(colour_image.astype(np.uint8)).resize(
                (camera.width, camera.height), Image.Resampling.BILINEAR
            )).astype(float)
        low_frequency_colour = gaussian_filter(
            colour_image, sigma=(3.0, 3.0, 0.0), mode="nearest"
        )
        colours = low_frequency_colour[
            np.clip(pixel_y.astype(int), 0, camera.height - 1),
            np.clip(pixel_x.astype(int), 0, camera.width - 1),
        ]
        confidence = np.clip(weight[hy, hx] * (1.0 - relative_spread[hy, hx]), 0, 1)
        predicted_points.append(world)
        predicted_colours.append(np.clip(colours, 0, 255).astype(np.uint8))
        predicted_confidence.append(confidence)
        view_reports.append({"image": image.name, "predicted_cells": int(len(world))})
        if progress_callback is not None:
            progress_callback(view_number, len(views))

    if should_cancel is not None and should_cancel():
        raise RuntimeError("用户已取消弱纹理点云预测，原始点云未修改。")

    if not predicted_points:
        raise ValueError("没有找到同时满足包围约束和深度连续性的投影空洞。")
    points = np.vstack(predicted_points)
    colours = np.vstack(predicted_colours)
    confidence = np.concatenate(predicted_confidence)

    reference_tree = cKDTree(reference)
    nearest_distance, _ = reference_tree.query(points, k=1)
    keep = (nearest_distance >= spacing * 0.75) & (nearest_distance <= spacing * 12.0)
    points, colours, confidence = points[keep], colours[keep], confidence[keep]
    if not len(points):
        raise ValueError("预测点均与原点云过近或过远，已为避免伪点而放弃补全。")
    voxel = max(spacing * 1.25, 1e-8)
    keys = np.floor(points / voxel).astype(np.int64)
    order = np.argsort(-confidence)
    _, unique_position = np.unique(keys[order], axis=0, return_index=True)
    chosen = order[unique_position]
    if len(chosen) > max_predicted_points:
        chosen = chosen[np.argsort(-confidence[chosen])[:max_predicted_points]]
    points, colours = points[chosen], colours[chosen]

    # VTP is used for the combined cloud because the common PLY writer drops
    # arbitrary scalar fields such as PointSource.  The predicted-only export
    # remains a colour PLY for interoperability with Blender and CloudCompare.
    combined_path = output_directory / "fused-weak-texture-completed.vtp"
    predicted_path = output_directory / "weak-texture-predicted-only.ply"
    report_path = output_directory / "weak-texture-completion-report.json"
    _save_cloud(
        combined_path,
        np.vstack((original_points, points)),
        np.vstack((original_rgb, colours)),
        np.concatenate((np.zeros(len(original_points), dtype=np.uint8),
                        np.ones(len(points), dtype=np.uint8))),
    )
    _save_cloud(
        predicted_path, points, colours, np.ones(len(points), dtype=np.uint8)
    )
    report_path.write_text(json.dumps({
        "method": "multiview_projection_enclosed_depth_interpolation",
        "grid_step_pixels": grid_step,
        "estimated_point_spacing": spacing,
        "original_points": int(len(original_points)),
        "predicted_points": int(len(points)),
        "colour": "gaussian_low_frequency_static_image_projection",
        "point_source": {"0": "original", "1": "predicted"},
        "views": view_reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return CloudCompletionSummary(
        len(original_points), len(points), combined_path, predicted_path, report_path
    )
