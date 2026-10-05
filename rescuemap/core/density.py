from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class DensityResult:
    mean_neighbor_distance: np.ndarray
    score: np.ndarray
    classes: np.ndarray
    colors: np.ndarray
    k_neighbors: int
    dense_distance: float
    sparse_distance: float

    @property
    def sparse_count(self) -> int:
        return int(np.count_nonzero(self.classes == 0))

    @property
    def medium_count(self) -> int:
        return int(np.count_nonzero(self.classes == 1))

    @property
    def dense_count(self) -> int:
        return int(np.count_nonzero(self.classes == 2))


def analyze_point_density(
    points: np.ndarray,
    *,
    k_neighbors: int = 16,
    green_median_multiplier: float = 1.5,
    red_median_multiplier: float = 3.0,
    chunk_size: int = 100_000,
) -> DensityResult:
    """Classify relative point density using local nearest-neighbor spacing.

    A smaller mean distance to the K nearest neighboring points means denser local
    sampling. Thresholds are relative to the median local spacing, so ordinary
    terrain does not have to belong to the densest fixed percentile to be green.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云坐标必须是 N×3 数组。")
    if len(points) <= k_neighbors:
        raise ValueError(f"点数必须大于近邻数量 {k_neighbors}。")
    if not 0 < green_median_multiplier < red_median_multiplier:
        raise ValueError("绿色与红色密度倍数设置无效。")

    tree = cKDTree(points)
    mean_distance = np.empty(len(points), dtype=np.float32)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        distances, _ = tree.query(
            points[start:end],
            k=k_neighbors + 1,
            workers=-1,
        )
        # Column zero is the point itself and always has distance zero.
        mean_distance[start:end] = distances[:, 1:].mean(axis=1)

    median_distance = float(np.median(mean_distance))
    dense_distance = median_distance * green_median_multiplier
    sparse_distance = median_distance * red_median_multiplier
    span = max(sparse_distance - dense_distance, np.finfo(np.float32).eps)

    score = 1.0 - np.clip(
        (mean_distance - dense_distance) / span,
        0.0,
        1.0,
    )
    score = score.astype(np.float32)

    classes = np.ones(len(points), dtype=np.uint8)
    classes[mean_distance >= sparse_distance] = 0
    classes[mean_distance <= dense_distance] = 2

    # Red = sparse, yellow = medium, green = dense.
    palette = np.array(
        [
            [220, 55, 55],
            [242, 190, 45],
            [52, 180, 90],
        ],
        dtype=np.uint8,
    )
    colors = palette[classes]

    return DensityResult(
        mean_neighbor_distance=mean_distance,
        score=score,
        classes=classes,
        colors=colors,
        k_neighbors=k_neighbors,
        dense_distance=dense_distance,
        sparse_distance=sparse_distance,
    )
