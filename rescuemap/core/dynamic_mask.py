from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve


MAX_IMAGE_ID = 2_147_483_647


@dataclass(frozen=True, slots=True)
class DynamicMaskSummary:
    image_count: int
    masked_image_count: int
    masked_pixel_count: int
    total_pixel_count: int
    mean_static_visibility: float = 1.0
    report_path: Path | None = None

    @property
    def masked_ratio(self) -> float:
        if self.total_pixel_count <= 0:
            return 0.0
        return self.masked_pixel_count / self.total_pixel_count


def _decode_pair_id(pair_id: int) -> tuple[int, int]:
    return pair_id // MAX_IMAGE_ID, pair_id % MAX_IMAGE_ID


def _decode_database_text(value: str | bytes) -> str:
    if isinstance(value, str):
        return value
    for encoding in ("utf-8", "gb18030"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            pass
    return value.decode("utf-8", errors="surrogateescape")


def _blob_array(data: bytes | None, dtype: np.dtype, rows: int, cols: int) -> np.ndarray:
    if data is None or rows <= 0 or cols <= 0:
        return np.empty((0, max(cols, 0)), dtype=dtype)
    values = np.frombuffer(data, dtype=dtype)
    expected = rows * cols
    if values.size < expected:
        return np.empty((0, cols), dtype=dtype)
    return values[:expected].reshape(rows, cols)


def _clustered_dynamic_cells(
    rejected: np.ndarray,
    observed: np.ndarray,
    *,
    minimum_rejections: int = 4,
    minimum_ratio: float = 0.60,
) -> np.ndarray:
    """Return conservative spatial clusters of repeatedly rejected matches."""
    ratio = rejected / np.maximum(observed, 1.0)
    seeds = (rejected >= minimum_rejections) & (ratio >= minimum_ratio)
    seed_neighbors = convolve(
        seeds.astype(np.int8), np.ones((3, 3), dtype=np.int8), mode="constant"
    )
    clustered = seeds & (seed_neighbors >= 2)
    return clustered


_STRENGTH_PARAMETERS = {
    # minimum rejected observations, rejection ratio, point-mask radius/cell
    "conservative": (6, 0.72, 0.42),
    "standard": (4, 0.60, 0.55),
    "aggressive": (3, 0.50, 0.72),
}


def _point_level_dynamic_mask(
    width: int,
    height: int,
    cell_size: int,
    dynamic_cells: np.ndarray,
    evidence: list[tuple[np.ndarray, bool]],
    radius_scale: float,
) -> np.ndarray:
    """Rasterize compact circles around inconsistent features in valid clusters.

    This deliberately avoids blacking the complete grid cell.  The old block
    mask removed visible road/bridge pixels surrounding a moving vehicle.
    """
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    radius = max(4, int(round(cell_size * radius_scale)))
    for point, is_rejected in evidence:
        if not is_rejected:
            continue
        x = min(width - 1, max(0, int(round(float(point[0])))))
        y = min(height - 1, max(0, int(round(float(point[1])))))
        grid_x = min(dynamic_cells.shape[1] - 1, x // cell_size)
        grid_y = min(dynamic_cells.shape[0] - 1, y // cell_size)
        if not dynamic_cells[grid_y, grid_x]:
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def generate_geometry_masks(
    database_path: Path,
    image_directory: Path,
    mask_directory: Path,
    static_image_directory: Path,
    *,
    neighbor_span: int = 2,
    strength: str = "conservative",
    preview_directory: Path | None = None,
    report_path: Path | None = None,
) -> DynamicMaskSummary:
    """Create COLMAP masks from spatially clustered epipolar-inconsistent matches.

    The preview database contains both raw descriptor matches and geometrically
    verified inliers. Matches absent from the verified set are treated as
    inconsistent evidence. Only repeated, spatially clustered evidence from
    nearby frames is masked, which deliberately favours retaining uncertain data.
    """
    database_path = database_path.resolve()
    image_directory = image_directory.resolve()
    mask_directory.mkdir(parents=True, exist_ok=True)
    static_image_directory.mkdir(parents=True, exist_ok=True)
    if strength not in _STRENGTH_PARAMETERS:
        raise ValueError(f"未知动态抑制强度：{strength}")
    minimum_rejections, minimum_ratio, radius_scale = _STRENGTH_PARAMETERS[strength]
    if preview_directory is not None:
        preview_directory.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        # Windows COLMAP may store paths using the active ANSI code page rather
        # than UTF-8. Reading bytes lets us support both UTF-8 and Chinese GBK.
        connection.text_factory = bytes
        image_rows = connection.execute(
            "SELECT image_id, name FROM images ORDER BY name"
        ).fetchall()
        keypoint_rows = connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"
        ).fetchall()
        match_rows = connection.execute(
            "SELECT m.pair_id, m.rows, m.cols, m.data, "
            "g.rows, g.cols, g.data "
            "FROM matches AS m LEFT JOIN two_view_geometries AS g "
            "ON m.pair_id = g.pair_id"
        ).fetchall()

    if not image_rows:
        raise RuntimeError("快速重建数据库中没有图像记录，无法生成动态区域掩膜。")

    names = {
        int(image_id): _decode_database_text(name) for image_id, name in image_rows
    }
    order = {image_id: index for index, (image_id, _name) in enumerate(image_rows)}
    keypoints = {
        int(image_id): _blob_array(data, np.float32, int(rows), int(cols))[:, :2]
        for image_id, rows, cols, data in keypoint_rows
    }
    evidence: dict[int, list[tuple[np.ndarray, bool]]] = {
        image_id: [] for image_id in names
    }

    for pair_id, raw_rows, raw_cols, raw_data, geo_rows, geo_cols, geo_data in match_rows:
        first_id, second_id = _decode_pair_id(int(pair_id))
        if first_id not in names or second_id not in names:
            continue
        if abs(order[first_id] - order[second_id]) > neighbor_span:
            continue
        raw = _blob_array(raw_data, np.uint32, int(raw_rows), int(raw_cols))
        if raw.shape[1] < 2 or raw.size == 0:
            continue
        verified = _blob_array(
            geo_data, np.uint32, int(geo_rows or 0), int(geo_cols or 0)
        )
        verified_pairs = {
            (int(row[0]), int(row[1])) for row in verified if row.size >= 2
        }
        first_points = keypoints.get(first_id, np.empty((0, 2)))
        second_points = keypoints.get(second_id, np.empty((0, 2)))
        for first_index, second_index in raw[:, :2]:
            first_index = int(first_index)
            second_index = int(second_index)
            if first_index >= len(first_points) or second_index >= len(second_points):
                continue
            rejected = (first_index, second_index) not in verified_pairs
            evidence[first_id].append((first_points[first_index], rejected))
            evidence[second_id].append((second_points[second_index], rejected))

    masked_images = 0
    masked_pixels = 0
    total_pixels = 0
    image_reports: list[dict[str, object]] = []
    static_visibility_values: list[float] = []
    for image_id, relative_name in names.items():
        source = image_directory / Path(relative_name)
        if not source.is_file():
            raise FileNotFoundError(f"快速重建使用的照片不存在：{source}")
        with Image.open(source) as opened:
            rgb = opened.convert("RGB")
            width, height = rgb.size
            pixel_data = np.asarray(rgb).copy()

        cell_size = max(24, min(width, height) // 40)
        grid_width = max(1, (width + cell_size - 1) // cell_size)
        grid_height = max(1, (height + cell_size - 1) // cell_size)
        observed = np.zeros((grid_height, grid_width), dtype=np.int32)
        rejected = np.zeros_like(observed)
        for point, is_rejected in evidence[image_id]:
            x = min(grid_width - 1, max(0, int(float(point[0]) // cell_size)))
            y = min(grid_height - 1, max(0, int(float(point[1]) // cell_size)))
            observed[y, x] += 1
            if is_rejected:
                rejected[y, x] += 1

        dynamic_cells = _clustered_dynamic_cells(
            rejected,
            observed,
            minimum_rejections=minimum_rejections,
            minimum_ratio=minimum_ratio,
        )
        dynamic_pixels = _point_level_dynamic_mask(
            width,
            height,
            cell_size,
            dynamic_cells,
            evidence[image_id],
            radius_scale,
        )

        static_mask = np.where(dynamic_pixels, 0, 255).astype(np.uint8)
        mask_target = mask_directory / (relative_name + ".png")
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(static_mask, mode="L").save(mask_target)

        static_target = static_image_directory / Path(relative_name)
        static_target.parent.mkdir(parents=True, exist_ok=True)
        dynamic_count = int(dynamic_pixels.sum())
        if dynamic_count:
            if preview_directory is not None:
                overlay = pixel_data.copy()
                overlay[dynamic_pixels] = (
                    overlay[dynamic_pixels].astype(np.float32) * 0.25
                    + np.array([255.0, 40.0, 40.0]) * 0.75
                ).astype(np.uint8)
                preview_target = preview_directory / Path(relative_name)
                preview_target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay).save(preview_target, quality=92)
            pixel_data[dynamic_pixels] = 0
            save_options: dict[str, object] = {}
            if static_target.suffix.lower() in {".jpg", ".jpeg"}:
                save_options = {"quality": 95, "subsampling": 0}
            Image.fromarray(pixel_data, mode="RGB").save(static_target, **save_options)
            masked_images += 1
        else:
            shutil.copy2(source, static_target)
        masked_pixels += dynamic_count
        total_pixels += width * height
        observation_count = len(evidence[image_id])
        rejection_count = sum(1 for _point, flag in evidence[image_id] if flag)
        static_visibility = (
            1.0 - rejection_count / observation_count
            if observation_count > 0
            else 1.0
        )
        static_visibility = float(np.clip(static_visibility, 0.0, 1.0))
        static_visibility_values.append(static_visibility)
        image_reports.append(
            {
                "image": relative_name,
                "geometric_static_visibility": static_visibility,
                "masked_pixels": dynamic_count,
                "masked_ratio": dynamic_count / max(width * height, 1),
                "dynamic_feature_clusters": int(dynamic_cells.sum()),
            }
        )

    mean_static_visibility = float(np.mean(static_visibility_values))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "method": "cross_frame_geometry_consistency",
                    "strength": strength,
                    "meaning": (
                        "geometric_static_visibility is a feature-observation score, "
                        "not a semantic probability"
                    ),
                    "summary": {
                        "image_count": len(names),
                        "masked_image_count": masked_images,
                        "masked_pixel_count": masked_pixels,
                        "total_pixel_count": total_pixels,
                        "masked_ratio": masked_pixels / max(total_pixels, 1),
                        "mean_static_visibility": mean_static_visibility,
                    },
                    "images": image_reports,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DynamicMaskSummary(
        image_count=len(names),
        masked_image_count=masked_images,
        masked_pixel_count=masked_pixels,
        total_pixel_count=total_pixels,
        mean_static_visibility=mean_static_visibility,
        report_path=report_path,
    )
