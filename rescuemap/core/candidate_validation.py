from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt

from rescuemap.core.terrain import (
    FILL_AUTO,
    FILL_PLANE,
    FILL_QUADRATIC,
    TerrainGrid,
)


_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(slots=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def pinhole_parameters(self) -> tuple[float, float, float, float]:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
            focal, cx, cy = self.params[:3]
            return float(focal), float(focal), float(cx), float(cy)
        if self.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE",
                          "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
            fx, fy, cx, cy = self.params[:4]
            return float(fx), float(fy), float(cx), float(cy)
        raise ValueError(f"暂不支持相机模型：{self.model}")


@dataclass(slots=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(slots=True)
class CandidateScore:
    strategy: str
    geometry_score: float
    model_agreement_score: float
    texture_strength: float
    photometric_score: float | None
    effective_views: int
    final_confidence: float


@dataclass(slots=True)
class CandidateValidationResult:
    region_id: int
    scores: list[CandidateScore]
    recommended_strategy: str
    image_evidence_available: bool
    context_message: str


def _read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("COLMAP模型文件不完整。")
    return data


def _read_c_string(stream) -> str:
    parts = bytearray()
    while True:
        byte = _read_exact(stream, 1)
        if byte == b"\x00":
            break
        parts.extend(byte)
    raw = bytes(parts)
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(qvec, dtype=float)
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,
             2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz,
             2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw,
             1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=float,
    )


def read_colmap_binary_model(
    model_directory: Path,
) -> tuple[dict[int, ColmapCamera], list[ColmapImage]]:
    """Read the camera poses needed for reverse projection.

    COLMAP's undistorted ``dense/sparse`` model normally uses PINHOLE cameras,
    but the parser retains the common model identifiers for imported projects.
    """
    model_directory = Path(model_directory)
    cameras_path = model_directory / "cameras.bin"
    images_path = model_directory / "images.bin"
    if not cameras_path.is_file() or not images_path.is_file():
        raise FileNotFoundError("找不到dense/sparse中的cameras.bin或images.bin。")

    cameras: dict[int, ColmapCamera] = {}
    with cameras_path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        for _ in range(count):
            camera_id, model_id = struct.unpack("<ii", _read_exact(stream, 8))
            width, height = struct.unpack("<QQ", _read_exact(stream, 16))
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"未知COLMAP相机模型编号：{model_id}")
            model_name, parameter_count = _CAMERA_MODELS[model_id]
            params = np.asarray(
                struct.unpack(
                    "<" + "d" * parameter_count,
                    _read_exact(stream, 8 * parameter_count),
                ),
                dtype=float,
            )
            cameras[camera_id] = ColmapCamera(
                camera_id, model_name, int(width), int(height), params
            )

    images: list[ColmapImage] = []
    with images_path.open("rb") as stream:
        count = struct.unpack("<Q", _read_exact(stream, 8))[0]
        for _ in range(count):
            image_id = struct.unpack("<i", _read_exact(stream, 4))[0]
            qvec = np.asarray(struct.unpack("<4d", _read_exact(stream, 32)))
            tvec = np.asarray(struct.unpack("<3d", _read_exact(stream, 24)))
            camera_id = struct.unpack("<i", _read_exact(stream, 4))[0]
            name = _read_c_string(stream)
            point_count = struct.unpack("<Q", _read_exact(stream, 8))[0]
            stream.seek(int(point_count) * 24, 1)
            images.append(
                ColmapImage(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name,
                    rotation=_quaternion_to_rotation(qvec),
                    translation=tvec,
                )
            )
    return cameras, images


def project_points(
    points: np.ndarray,
    camera: ColmapCamera,
    image: ColmapImage,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    camera_points = (image.rotation @ points.T).T + image.translation
    depth = camera_points[:, 2]
    fx, fy, cx, cy = camera.pinhole_parameters()
    uv = np.full((len(points), 2), np.nan, dtype=float)
    positive = depth > 1e-9
    uv[positive, 0] = fx * camera_points[positive, 0] / depth[positive] + cx
    uv[positive, 1] = fy * camera_points[positive, 1] / depth[positive] + cy
    return uv, depth


def locate_dense_colmap_context(
    model_path: Path,
) -> tuple[Path, Path] | None:
    """Locate undistorted images and poses beside a loaded fused point cloud."""
    model_path = Path(model_path).resolve()
    dense = model_path.parent
    model_directory = dense / "sparse"
    image_directory = dense / "images"
    if (model_directory / "cameras.bin").is_file() and image_directory.is_dir():
        return model_directory, image_directory
    return None


def _candidate_heights(
    grid: TerrainGrid, region_id: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    mask = grid.region_mask(region_id)
    if not np.any(mask):
        raise ValueError("预测区域编号无效。")
    boundary = grid._surrounding_observations(mask)
    automatic = np.asarray(grid.automatic_height[mask], dtype=float).copy()
    plane = grid._fit_surrounding_plane(region_id, boundary, mask)
    quadratic = grid._fit_surrounding_quadratic(region_id, boundary, mask)
    # Explicit rectangles can fall outside the original harmonic solution.
    automatic[~np.isfinite(automatic)] = plane[~np.isfinite(automatic)]
    return mask, {
        FILL_AUTO: automatic,
        FILL_PLANE: np.asarray(plane, dtype=float),
        FILL_QUADRATIC: np.asarray(quadratic, dtype=float),
    }, boundary


def _geometry_scores(
    grid: TerrainGrid,
    mask: np.ndarray,
    candidates: dict[str, np.ndarray],
    boundary: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    bx, by = np.where(boundary)
    target_index_map = np.full(mask.shape, -1, dtype=np.int64)
    tx, ty = np.where(mask)
    target_index_map[tx, ty] = np.arange(len(tx))
    candidate_stack = np.vstack([candidates[key] for key in candidates])
    agreement_scale = max(float(np.nanstd(candidate_stack)), grid.cell_size * 0.05, 1e-9)
    geometry: dict[str, float] = {}
    agreement: dict[str, float] = {}
    median_surface = np.nanmedian(candidate_stack, axis=0)
    observed_span = max(
        float(np.nanpercentile(grid.observed_height[boundary], 90)
              - np.nanpercentile(grid.observed_height[boundary], 10)),
        grid.cell_size * 0.25,
        1e-9,
    )
    for strategy, heights in candidates.items():
        nearest_x = nearest[0, bx, by]
        nearest_y = nearest[1, bx, by]
        indices = target_index_map[nearest_x, nearest_y]
        valid = indices >= 0
        residual = heights[indices[valid]] - grid.observed_height[bx[valid], by[valid]]
        boundary_rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else np.inf
        geometry[strategy] = float(np.exp(-boundary_rmse / observed_span))
        disagreement = float(np.sqrt(np.mean((heights - median_surface) ** 2)))
        agreement[strategy] = float(np.exp(-disagreement / agreement_scale))
    return geometry, agreement


def _sample_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, limit, dtype=int)


def _image_measurements(
    points_raw: np.ndarray,
    cameras: dict[int, ColmapCamera],
    registered_images: list[ColmapImage],
    image_directory: Path,
    *,
    max_images: int = 12,
) -> tuple[float, float | None, int]:
    """Return texture strength, robust photometric consistency and view count."""
    image_choices = registered_images
    if len(image_choices) > max_images:
        ids = _sample_indices(len(image_choices), max_images)
        image_choices = [image_choices[int(index)] for index in ids]

    samples_by_point: list[list[float]] = [[] for _ in range(len(points_raw))]
    texture_by_point: list[list[float]] = [[] for _ in range(len(points_raw))]
    effective_views = 0
    for registered in image_choices:
        path = image_directory / Path(registered.name)
        camera = cameras.get(registered.camera_id)
        if camera is None or not path.is_file():
            continue
        with Image.open(path) as pil_image:
            gray_image = pil_image.convert("L")
            gray_image.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
            gray = np.asarray(gray_image, dtype=np.float32) / 255.0
        uv, depth = project_points(points_raw, camera, registered)
        scale_x = gray.shape[1] / max(float(camera.width), 1.0)
        scale_y = gray.shape[0] / max(float(camera.height), 1.0)
        uv[:, 0] *= scale_x
        uv[:, 1] *= scale_y
        finite_uv = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        x = np.zeros(len(uv), dtype=np.int64)
        y = np.zeros(len(uv), dtype=np.int64)
        x[finite_uv] = np.rint(uv[finite_uv, 0]).astype(np.int64)
        y[finite_uv] = np.rint(uv[finite_uv, 1]).astype(np.int64)
        valid = (
            (depth > 0)
            & finite_uv
            & (x >= 2) & (x < gray.shape[1] - 2)
            & (y >= 2) & (y < gray.shape[0] - 2)
        )
        if int(valid.sum()) < max(3, len(points_raw) // 20):
            continue
        image_mean = float(np.mean(gray))
        image_std = max(float(np.std(gray)), 0.03)
        accepted_in_view = 0
        for point_index in np.where(valid)[0]:
            px, py = int(x[point_index]), int(y[point_index])
            # The two-pass reconstruction writes dynamic masks into the static
            # images as black pixels.  A mostly black neighbourhood therefore
            # means "no trustworthy observation", not weak terrain texture.
            patch = gray[py - 1:py + 2, px - 1:px + 2]
            if float(np.mean(patch < 0.01)) > 0.40:
                continue
            normalized_intensity = (float(gray[py, px]) - image_mean) / image_std
            gx = float(gray[py, px + 1] - gray[py, px - 1]) * 0.5
            gy = float(gray[py + 1, px] - gray[py - 1, px]) * 0.5
            texture = min(np.hypot(gx, gy) / 0.12, 1.0)
            samples_by_point[point_index].append(normalized_intensity)
            texture_by_point[point_index].append(float(texture))
            accepted_in_view += 1
        if accepted_in_view >= max(3, len(points_raw) // 20):
            effective_views += 1

    texture_values: list[float] = []
    dispersions: list[float] = []
    for intensities, textures in zip(samples_by_point, texture_by_point):
        if len(intensities) < 2:
            continue
        values = np.asarray(intensities, dtype=float)
        median = float(np.median(values))
        dispersions.append(float(np.median(np.abs(values - median))))
        texture_values.append(float(np.median(textures)))
    if not texture_values:
        return 0.0, None, effective_views
    texture_strength = float(np.clip(np.median(texture_values), 0.0, 1.0))
    dispersion = float(np.median(dispersions))
    photometric_score = float(np.exp(-dispersion / 0.45))
    return texture_strength, photometric_score, effective_views


def evaluate_region_candidates(
    grid: TerrainGrid,
    region_id: int,
    *,
    model_path: Path,
    calibration_matrix: np.ndarray,
    max_points: int = 96,
) -> CandidateValidationResult:
    """Evaluate three surfaces using geometry, model agreement and images."""
    mask, candidates, boundary = _candidate_heights(grid, region_id)
    geometry, agreement = _geometry_scores(grid, mask, candidates, boundary)
    context = locate_dense_colmap_context(model_path)
    cameras: dict[int, ColmapCamera] = {}
    registered_images: list[ColmapImage] = []
    image_directory: Path | None = None
    context_message = "未找到dense/sparse与dense/images，仅进行几何候选评价。"
    if context is not None:
        try:
            cameras, registered_images = read_colmap_binary_model(context[0])
            image_directory = context[1]
            context_message = f"已读取{len(registered_images)}个COLMAP有效视角。"
        except (OSError, ValueError) as exc:
            context_message = f"COLMAP相机读取失败，仅进行几何评价：{exc}"

    tx, ty = np.where(mask)
    point_ids = _sample_indices(len(tx), max_points)
    inverse_calibration = np.linalg.inv(np.asarray(calibration_matrix, dtype=float))
    scores: list[CandidateScore] = []
    any_image_evidence = False
    for strategy, heights in candidates.items():
        calibrated_points = np.column_stack(
            (grid.x_centers[tx], grid.y_centers[ty], heights)
        )[point_ids]
        homogeneous = np.column_stack((calibrated_points, np.ones(len(calibrated_points))))
        raw_points_h = (inverse_calibration @ homogeneous.T).T
        raw_points = raw_points_h[:, :3] / raw_points_h[:, 3:4]
        texture = 0.0
        photometric: float | None = None
        effective_views = 0
        if image_directory is not None and cameras and registered_images:
            texture, photometric, effective_views = _image_measurements(
                raw_points, cameras, registered_images, image_directory
            )
        image_quality = texture * min(effective_views / 3.0, 1.0)
        if photometric is not None and effective_views >= 2:
            any_image_evidence = True
            photo_weight = min(0.50, 0.50 * image_quality)
        else:
            photo_weight = 0.0
        model_weight = 0.25
        geometry_weight = 1.0 - model_weight - photo_weight
        final = (
            geometry_weight * geometry[strategy]
            + model_weight * agreement[strategy]
            + photo_weight * (photometric if photometric is not None else 0.0)
        )
        scores.append(
            CandidateScore(
                strategy=strategy,
                geometry_score=geometry[strategy],
                model_agreement_score=agreement[strategy],
                texture_strength=texture,
                photometric_score=photometric,
                effective_views=effective_views,
                final_confidence=float(np.clip(final, 0.0, 1.0)),
            )
        )
    recommended = max(scores, key=lambda item: item.final_confidence).strategy
    return CandidateValidationResult(
        region_id=region_id,
        scores=scores,
        recommended_strategy=recommended,
        image_evidence_available=any_image_evidence,
        context_message=context_message,
    )
