from __future__ import annotations

import numpy as np


def _rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-12:
        if dot > 0:
            return np.eye(3)
        seed = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, seed)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (cross_norm**2))


def ground_alignment_transform(points: np.ndarray) -> np.ndarray:
    """Map the plane defined by three picked points to Z=0."""
    points = np.asarray(points, dtype=float)
    if points.shape != (3, 3):
        raise ValueError("需要三个三维点来确定地面。")
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(normal)
    if length < 1e-10:
        raise ValueError("三个点几乎在同一直线上，请重新选择。")
    normal /= length
    rotation = _rotation_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ points[0])
    return transform


def uniform_scale_transform(scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("尺度系数必须为正数。")
    transform = np.eye(4)
    transform[0, 0] = transform[1, 1] = transform[2, 2] = scale
    return transform


def flip_vertical_transform() -> np.ndarray:
    """Rotate 180 degrees around X while keeping a right-handed frame."""
    transform = np.eye(4)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return transform
