from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.stats import binned_statistic_2d


PREDICTED = "PREDICTED"
ACCEPTED = "ACCEPTED"
UNCERTAIN = "UNCERTAIN"
RESHOOT = "RESHOOT"

FILL_AUTO = "AUTO"
FILL_FLAT = "FLAT"
FILL_PLANE = "PLANE"
FILL_QUADRATIC = "QUADRATIC"
FILL_CONVEX = "CONVEX"
FILL_CONCAVE = "CONCAVE"

FILL_STRATEGY_NAMES = {
    FILL_AUTO: "自动二维补全",
    FILL_FLAT: "填平",
    FILL_PLANE: "倾斜平面",
    FILL_QUADRATIC: "缓变曲面（遮挡恢复）",
    FILL_CONVEX: "上凸",
    FILL_CONCAVE: "下凹",
}

FILL_STRATEGY_CODES = {
    FILL_AUTO: 1,
    FILL_FLAT: 2,
    FILL_PLANE: 3,
    FILL_QUADRATIC: 4,
    FILL_CONVEX: 5,
    FILL_CONCAVE: 6,
}

DECISION_COLORS = {
    PREDICTED: np.array([242, 190, 45], dtype=np.uint8),
    ACCEPTED: np.array([70, 150, 235], dtype=np.uint8),
    UNCERTAIN: np.array([150, 150, 150], dtype=np.uint8),
    RESHOOT: np.array([220, 55, 55], dtype=np.uint8),
}


@dataclass(slots=True)
class TerrainGrid:
    x_centers: np.ndarray
    y_centers: np.ndarray
    observed_height: np.ndarray
    predicted_height: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    distance_to_observation: np.ndarray
    region_labels: np.ndarray
    cell_size: float
    min_points: int
    algorithm: str
    decisions: dict[int, str] = field(default_factory=dict)
    fill_strategies: dict[int, str] = field(default_factory=dict)
    fill_parameters: dict[int, float] = field(default_factory=dict)
    guide_points: dict[int, list[list[float]]] = field(default_factory=dict)
    manual_fill_bounds: dict[int, list[list[float]]] = field(default_factory=dict)
    automatic_height: np.ndarray = field(init=False, repr=False)
    manual_fill_labels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.automatic_height = np.asarray(self.predicted_height, dtype=float).copy()
        self.manual_fill_labels = np.zeros_like(self.region_labels, dtype=np.int32)

    @property
    def region_count(self) -> int:
        return int(self.region_labels.max(initial=0))

    def set_decision(self, region_id: int, decision: str) -> None:
        if region_id < 1 or region_id > self.region_count:
            raise ValueError("预测区域编号无效。")
        if decision not in DECISION_COLORS:
            raise ValueError("预测区域状态无效。")
        self.decisions[region_id] = decision

    def decision_for(self, region_id: int) -> str:
        return self.decisions.get(region_id, PREDICTED)

    def fill_strategy_for(self, region_id: int) -> str:
        return self.fill_strategies.get(region_id, FILL_AUTO)

    def set_guide_points(self, region_id: int, points: np.ndarray) -> None:
        if region_id < 1 or region_id > self.region_count:
            raise ValueError("预测区域编号无效。")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            raise ValueError("至少需要3个三维参考点。")
        if not np.all(np.isfinite(points)):
            raise ValueError("参考点包含无效坐标。")
        self.guide_points[region_id] = points.tolist()

    def guide_points_for(self, region_id: int) -> np.ndarray:
        points = self.guide_points.get(region_id, [])
        return np.asarray(points, dtype=float).reshape((-1, 3))

    def effective_region_labels(self) -> np.ndarray:
        """Return automatic labels with explicit user rectangles overlaid."""
        return np.where(
            self.manual_fill_labels > 0,
            self.manual_fill_labels,
            self.region_labels,
        )

    def region_mask(self, region_id: int) -> np.ndarray:
        labels = self.effective_region_labels()
        return (labels == region_id) & self.prediction_mask()

    def prediction_mask(self) -> np.ndarray:
        return self.predicted | (self.manual_fill_labels > 0)

    def observation_mask(self) -> np.ndarray:
        return self.observed & (self.manual_fill_labels == 0)

    def add_manual_fill_rectangle(
        self,
        region_id: int,
        bounds_xy: tuple[float, float, float, float],
    ) -> int:
        """Assign a user-drawn XY rectangle to an existing prediction region.

        The rectangle is independent from the guide points.  It deliberately
        overrides lower/upper observations at the same XY position because the
        user has explicitly selected which 2.5-D surface should be continued.
        """
        if region_id < 1 or region_id > self.region_count:
            raise ValueError("预测区域编号无效。")
        xmin, xmax = sorted((float(bounds_xy[0]), float(bounds_xy[1])))
        ymin, ymax = sorted((float(bounds_xy[2]), float(bounds_xy[3])))
        if not np.all(np.isfinite((xmin, xmax, ymin, ymax))):
            raise ValueError("人工补充范围包含无效坐标。")
        x_selected = (self.x_centers >= xmin) & (self.x_centers <= xmax)
        y_selected = (self.y_centers >= ymin) & (self.y_centers <= ymax)
        rectangle = x_selected[:, None] & y_selected[None, :]
        if not np.any(rectangle):
            raise ValueError("框选范围内没有地形栅格，请扩大框选范围。")
        self.manual_fill_labels[rectangle] = region_id
        self.manual_fill_bounds.setdefault(region_id, []).append(
            [xmin, xmax, ymin, ymax]
        )
        return int(rectangle.sum())

    def apply_fill_strategy(
        self,
        region_id: int,
        strategy: str,
        amplitude: float = 0.0,
    ) -> None:
        if region_id < 1 or region_id > self.region_count:
            raise ValueError("预测区域编号无效。")
        if strategy not in FILL_STRATEGY_NAMES:
            raise ValueError("不支持的人工填充策略。")
        if amplitude < 0:
            raise ValueError("起伏强度不能小于零。")
        mask = self.region_mask(region_id)
        manual_mask = self.manual_fill_labels == region_id
        if strategy == FILL_AUTO:
            automatic_mask = mask & ~manual_mask
            self.predicted_height[automatic_mask] = self.automatic_height[automatic_mask]
            if np.any(manual_mask):
                boundary_mask = self._surrounding_observations(mask)
                plane = self._fit_surrounding_plane(region_id, boundary_mask, mask)
                self.predicted_height[manual_mask] = plane[manual_mask[mask]]
        else:
            boundary_mask = self._surrounding_observations(mask)
            plane = self._fit_surrounding_plane(region_id, boundary_mask, mask)
            if strategy == FILL_FLAT:
                guide_points = self.guide_points_for(region_id)
                reference_height = (
                    guide_points[:, 2]
                    if len(guide_points) >= 3
                    else self.observed_height[boundary_mask]
                )
                self.predicted_height[mask] = float(np.median(reference_height))
            elif strategy == FILL_PLANE:
                self.predicted_height[mask] = plane
            elif strategy == FILL_QUADRATIC:
                self.predicted_height[mask] = self._fit_surrounding_quadratic(
                    region_id, boundary_mask, mask
                )
            else:
                distance_inside = distance_transform_edt(mask)
                region_distance = distance_inside[mask]
                if float(region_distance.max(initial=0.0)) <= 1.0:
                    shape = np.ones_like(region_distance)
                else:
                    shape = np.clip(
                        (region_distance - 1.0)
                        / (float(region_distance.max()) - 1.0),
                        0.0,
                        1.0,
                    )
                    shape = np.sin(shape * np.pi / 2.0) ** 2
                direction = 1.0 if strategy == FILL_CONVEX else -1.0
                self.predicted_height[mask] = plane + direction * amplitude * shape
        self.fill_strategies[region_id] = strategy
        self.fill_parameters[region_id] = float(amplitude)

    def _surrounding_observations(self, region_mask: np.ndarray) -> np.ndarray:
        expanded = region_mask.copy()
        for _ in range(12):
            expanded = binary_dilation(expanded, structure=np.ones((3, 3), dtype=bool))
            boundary = expanded & self.observation_mask()
            if int(boundary.sum()) >= 6:
                return boundary
        boundary = self.observation_mask().copy()
        if int(boundary.sum()) < 3:
            raise ValueError("区域周围真实点不足，无法应用人工填充策略。")
        return boundary

    def _fit_surrounding_plane(
        self,
        region_id: int,
        boundary_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        tx, ty = np.where(target_mask)
        center_x = float(np.mean(self.x_centers[tx]))
        center_y = float(np.mean(self.y_centers[ty]))
        guide_points = self.guide_points_for(region_id)
        if len(guide_points) >= 3:
            reference_points = guide_points
        else:
            bx, by = np.where(boundary_mask)
            reference_points = np.column_stack(
                (
                    self.x_centers[bx],
                    self.y_centers[by],
                    self.observed_height[boundary_mask],
                )
            )
        coefficients = _robust_plane_coefficients(
            reference_points, center_x=center_x, center_y=center_y
        )
        return (
            coefficients[0] * (self.x_centers[tx] - center_x)
            + coefficients[1] * (self.y_centers[ty] - center_y)
            + coefficients[2]
        )

    def _fit_surrounding_quadratic(
        self,
        region_id: int,
        boundary_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        tx, ty = np.where(target_mask)
        center_x = float(np.mean(self.x_centers[tx]))
        center_y = float(np.mean(self.y_centers[ty]))
        guide_points = self.guide_points_for(region_id)
        if len(guide_points) >= 6:
            reference_points = guide_points
        else:
            bx, by = np.where(boundary_mask)
            reference_points = np.column_stack(
                (
                    self.x_centers[bx],
                    self.y_centers[by],
                    self.observed_height[boundary_mask],
                )
            )
        if len(reference_points) < 6:
            return self._fit_surrounding_plane(region_id, boundary_mask, target_mask)
        coefficients = _robust_quadratic_coefficients(
            reference_points, center_x=center_x, center_y=center_y
        )
        dx = self.x_centers[tx] - center_x
        dy = self.y_centers[ty] - center_y
        return (
            coefficients[0] * dx**2
            + coefficients[1] * dy**2
            + coefficients[2] * dx * dy
            + coefficients[3] * dx
            + coefficients[4] * dy
            + coefficients[5]
        )

    def prediction_polydata(self) -> pv.PolyData:
        predicted_mask = self.prediction_mask()
        labels = self.effective_region_labels()
        ix, iy = np.where(predicted_mask)
        if len(ix) == 0:
            return pv.PolyData()
        points = np.column_stack(
            (self.x_centers[ix], self.y_centers[iy], self.predicted_height[ix, iy])
        )
        regions = labels[ix, iy].astype(np.int32)
        colors = np.vstack(
            [DECISION_COLORS[self.decision_for(int(region))] for region in regions]
        )
        max_distance = max(
            float(self.distance_to_observation[predicted_mask].max(initial=1.0)),
            1.0,
        )
        uncertainty = np.clip(
            self.distance_to_observation[ix, iy] / max_distance, 0.0, 1.0
        ).astype(np.float32)
        # Explicitly drawn cells are geometric hypotheses confirmed by a user,
        # not new sensor observations.  Keep them visibly low-confidence for
        # later safety-aware path planning.
        uncertainty[self.manual_fill_labels[ix, iy] > 0] = 1.0
        cloud = pv.PolyData(points)
        cloud.point_data["RegionId"] = regions
        cloud.point_data["PredictionUncertainty"] = uncertainty
        cloud.point_data["DecisionRGB"] = colors
        cloud.point_data["FillStrategy"] = np.asarray(
            [FILL_STRATEGY_CODES[self.fill_strategy_for(int(region))]
             for region in regions],
            dtype=np.uint8,
        )
        return cloud

    def region_surface_polydata(self, region_id: int) -> pv.PolyData:
        """Create an adjacency-preserving surface preview for one prediction region."""
        mask = self.region_mask(region_id)
        if not np.any(mask):
            return pv.PolyData()
        index_map = np.full(mask.shape, -1, dtype=np.int64)
        ix, iy = np.where(mask)
        points = np.column_stack(
            (
                self.x_centers[ix],
                self.y_centers[iy],
                self.predicted_height[ix, iy],
            )
        )
        index_map[ix, iy] = np.arange(len(points), dtype=np.int64)
        triangles: list[tuple[int, int, int]] = []
        for x_index in range(mask.shape[0] - 1):
            for y_index in range(mask.shape[1] - 1):
                corners = (
                    int(index_map[x_index, y_index]),
                    int(index_map[x_index + 1, y_index]),
                    int(index_map[x_index + 1, y_index + 1]),
                    int(index_map[x_index, y_index + 1]),
                )
                for triangle in (
                    (corners[0], corners[1], corners[2]),
                    (corners[0], corners[2], corners[3]),
                ):
                    if min(triangle) >= 0:
                        triangles.append(triangle)
        if not triangles:
            return pv.PolyData(points)
        triangle_array = np.asarray(triangles, dtype=np.int64)
        faces = np.column_stack(
            (np.full(len(triangle_array), 3, dtype=np.int64), triangle_array)
        ).ravel()
        return pv.PolyData(points, faces)

    def region_summary(self, region_id: int) -> dict[str, float | int | str]:
        mask = self.region_mask(region_id)
        if not np.any(mask):
            raise ValueError("预测区域编号无效。")
        return {
            "region_id": region_id,
            "cells": int(mask.sum()),
            "area": float(mask.sum() * self.cell_size**2),
            "max_gap": float(self.distance_to_observation[mask].max() * self.cell_size),
            "decision": self.decision_for(region_id),
            "fill_strategy": self.fill_strategy_for(region_id),
            "fill_parameter": self.fill_parameters.get(region_id, 0.0),
            "guide_points": len(self.guide_points.get(region_id, [])),
        }


def _robust_plane_coefficients(
    points: np.ndarray,
    *,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Fit z=a(x-x0)+b(y-y0)+c using the majority trend of guide points."""
    points = np.asarray(points, dtype=float)
    design = np.column_stack(
        (
            points[:, 0] - center_x,
            points[:, 1] - center_y,
            np.ones(len(points)),
        )
    )
    if len(points) <= 3:
        coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
        return coefficients

    # Candidate planes from triples make the fit resistant to accidentally
    # selecting a few points from a lower/upper surface. Limit the candidate
    # pool so interactive use stays quick even when many guides are selected.
    sample_indices = np.linspace(
        0, len(points) - 1, min(len(points), 24), dtype=int
    )
    best_coefficients = None
    best_score = (np.inf, np.inf)
    for triple in combinations(sample_indices.tolist(), 3):
        candidate_design = design[np.asarray(triple)]
        if abs(float(np.linalg.det(candidate_design))) <= 1e-12:
            continue
        candidate = np.linalg.solve(candidate_design, points[np.asarray(triple), 2])
        residual = np.abs(points[:, 2] - design @ candidate)
        score = (float(np.median(residual)), float(np.mean(residual)))
        if score < best_score:
            best_score = score
            best_coefficients = candidate
    if best_coefficients is None:
        best_coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)

    residual = np.abs(points[:, 2] - design @ best_coefficients)
    median_residual = float(np.median(residual))
    height_span = max(float(np.ptp(points[:, 2])), 1e-6)
    threshold = max(3.0 * median_residual, 0.01 * height_span, 1e-6)
    inliers = residual <= threshold
    if int(inliers.sum()) >= 3:
        best_coefficients, *_ = np.linalg.lstsq(
            design[inliers], points[inliers, 2], rcond=None
        )
    return best_coefficients


def _robust_quadratic_coefficients(
    points: np.ndarray,
    *,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Fit a slowly varying quadratic surface and reject wrong-layer points."""
    points = np.asarray(points, dtype=float)
    dx = points[:, 0] - center_x
    dy = points[:, 1] - center_y
    design = np.column_stack((dx**2, dy**2, dx * dy, dx, dy, np.ones(len(points))))
    coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    if len(points) <= 6:
        return coefficients

    # Two robust refits are enough for an interactive terrain tool. MAD keeps
    # lower roads, vehicle remnants and vegetation from bending the bridge deck.
    for _ in range(2):
        residual = points[:, 2] - design @ coefficients
        median = float(np.median(residual))
        absolute_deviation = np.abs(residual - median)
        mad = float(np.median(absolute_deviation))
        height_span = max(float(np.ptp(points[:, 2])), 1e-6)
        threshold = max(3.5 * 1.4826 * mad, 0.01 * height_span, 1e-6)
        inliers = absolute_deviation <= threshold
        if int(inliers.sum()) < 6:
            break
        coefficients, *_ = np.linalg.lstsq(
            design[inliers], points[inliers, 2], rcond=None
        )
    return coefficients


def build_terrain_grid(
    points: np.ndarray,
    *,
    bounds_xy: tuple[float, float, float, float],
    cell_size: float,
    min_points: int = 3,
    max_prediction_distance_cells: float = 4.0,
    algorithm: str = "linear",
) -> TerrainGrid:
    """Build a 2.5D grid and predict short gaps.

    ``harmonic`` is an isotropic two-dimensional completion constrained by all
    observed cells and fills the complete selected rectangle. ``linear`` is a
    triangular-surface baseline. ``kalman`` is retained as the legacy row/column
    directional experiment.
    """
    points = np.asarray(points, dtype=float)
    if cell_size <= 0:
        raise ValueError("栅格尺寸必须大于零。")
    xmin, xmax, ymin, ymax = bounds_xy
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("地形范围无效。")
    inside = (
        (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
    )
    selected = points[inside]
    if len(selected) < 20:
        raise ValueError("所选范围内点数太少，请扩大范围。")
    nx = max(2, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(2, int(np.ceil((ymax - ymin) / cell_size)))
    if nx * ny > 2_000_000:
        raise ValueError("栅格数量过多，请增大栅格尺寸。")
    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5
    count, _, _, _ = binned_statistic_2d(
        selected[:, 0], selected[:, 1], selected[:, 2],
        statistic="count", bins=[x_edges, y_edges],
    )
    median_height, _, _, _ = binned_statistic_2d(
        selected[:, 0], selected[:, 1], selected[:, 2],
        statistic="median", bins=[x_edges, y_edges],
    )
    observed = (count >= min_points) & np.isfinite(median_height)
    if observed.sum() < 3:
        raise ValueError("有效地形格子不足，无法进行曲面预测。")
    gx, gy = np.meshgrid(x_centers, y_centers, indexing="ij")
    sample_xy = np.column_stack((gx[observed], gy[observed]))
    if algorithm == "harmonic":
        predicted_height = _harmonic_completion(median_height, observed)
    elif algorithm == "linear":
        predicted_height = griddata(
            sample_xy, median_height[observed], (gx, gy), method="linear"
        )
    elif algorithm == "kalman":
        predicted_height = _multidirectional_kalman(
            median_height, observed, cell_size
        )
    else:
        raise ValueError(f"不支持的地形预测算法：{algorithm}")
    distance_cells = distance_transform_edt(~observed)
    if algorithm == "harmonic":
        # Complete-fill mode: every unobserved cell in the user-selected ROI
        # is predicted. Distance remains available as an uncertainty signal.
        predicted = ~observed & np.isfinite(predicted_height)
    else:
        predicted = (
            ~observed & np.isfinite(predicted_height)
            & (distance_cells <= max_prediction_distance_cells)
        )
    region_labels, region_count = label(predicted, structure=np.ones((3, 3), dtype=int))
    decisions = {region_id: PREDICTED for region_id in range(1, region_count + 1)}
    return TerrainGrid(
        x_centers=x_centers,
        y_centers=y_centers,
        observed_height=median_height,
        predicted_height=predicted_height,
        observed=observed,
        predicted=predicted,
        distance_to_observation=distance_cells,
        region_labels=region_labels,
        cell_size=cell_size,
        min_points=min_points,
        algorithm=algorithm,
        decisions=decisions,
    )


def _harmonic_completion(
    heights: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Fill all unknown grid cells from the complete 2-D observed boundary.

    The discrete Laplace equation uses four cardinal and four diagonal
    neighbours. It is isotropic on the XY grid: a flat observed neighbourhood
    stays flat, and a planar slope extends through a hole without the drift of
    a one-dimensional directional extrapolation.
    """
    unknown = ~observed
    unknown_count = int(unknown.sum())
    result = np.asarray(heights, dtype=float).copy()
    if unknown_count == 0:
        return result
    if unknown_count > 300_000:
        raise ValueError(
            "需要补全的空白格超过30万个，请增大栅格边长后重试。"
        )

    unknown_index = np.full(observed.shape, -1, dtype=np.int64)
    unknown_index[unknown] = np.arange(unknown_count, dtype=np.int64)
    coordinates = np.column_stack(np.where(unknown))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    right_hand_side = np.zeros(unknown_count, dtype=float)
    diagonal_weight = 1.0 / np.sqrt(2.0)
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, diagonal_weight), (-1, 1, diagonal_weight),
        (1, -1, diagonal_weight), (1, 1, diagonal_weight),
    )
    nx, ny = observed.shape
    for equation, (x_index, y_index) in enumerate(coordinates):
        total_weight = 0.0
        for dx, dy, weight in neighbours:
            neighbour_x = int(x_index + dx)
            neighbour_y = int(y_index + dy)
            if not (0 <= neighbour_x < nx and 0 <= neighbour_y < ny):
                continue
            total_weight += weight
            if observed[neighbour_x, neighbour_y]:
                right_hand_side[equation] += (
                    weight * heights[neighbour_x, neighbour_y]
                )
            else:
                rows.append(equation)
                columns.append(int(unknown_index[neighbour_x, neighbour_y]))
                values.append(-weight)
        rows.append(equation)
        columns.append(equation)
        values.append(total_weight)

    matrix = csr_matrix(
        (values, (rows, columns)), shape=(unknown_count, unknown_count)
    )
    _, nearest_indices = distance_transform_edt(
        unknown, return_distances=True, return_indices=True
    )
    initial = heights[
        nearest_indices[0][unknown], nearest_indices[1][unknown]
    ]
    solution, solver_status = cg(
        matrix,
        right_hand_side,
        x0=initial,
        rtol=1e-7,
        atol=0.0,
        maxiter=3_000,
    )
    if solver_status != 0 or not np.all(np.isfinite(solution)):
        raise ValueError(
            "二维地形补全未能稳定收敛，请增大栅格边长后重试。"
        )
    result[unknown] = solution
    result[observed] = heights[observed]
    return result


def _kalman_direction(values: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-slope Kalman pass for one elevation profile."""
    output = np.full(len(values), np.nan, dtype=float)
    variance = np.full(len(values), np.inf, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return output, variance
    finite_values = values[finite]
    spread = max(float(np.median(np.abs(np.diff(finite_values)))), 1e-6)
    measurement_variance = max((spread * 0.25) ** 2, 1e-10)
    acceleration_variance = max((spread / max(step, 1e-9)) ** 2, 1e-10)
    f = np.array([[1.0, step], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q = acceleration_variance * np.array(
        [[step**4 / 4.0, step**3 / 2.0], [step**3 / 2.0, step**2]]
    )
    identity = np.eye(2)
    state = None
    covariance = None
    for index, measurement in enumerate(values):
        if state is None:
            if not np.isfinite(measurement):
                continue
            state = np.array([measurement, 0.0])
            covariance = np.diag([measurement_variance, acceleration_variance])
        else:
            state = f @ state
            covariance = f @ covariance @ f.T + q
            if np.isfinite(measurement):
                innovation = float(measurement - (h @ state)[0])
                innovation_covariance = float((h @ covariance @ h.T)[0, 0] + measurement_variance)
                gain = covariance @ h.T / innovation_covariance
                state = state + gain[:, 0] * innovation
                covariance = (identity - gain @ h) @ covariance
        output[index] = state[0]
        variance[index] = max(float(covariance[0, 0]), 1e-12)
    return output, variance


def _bidirectional_profile(values: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    forward, forward_var = _kalman_direction(values, step)
    backward, backward_var = _kalman_direction(values[::-1], step)
    backward = backward[::-1]
    backward_var = backward_var[::-1]
    valid = np.isfinite(forward) & np.isfinite(backward)
    fused = np.full(len(values), np.nan, dtype=float)
    fused_var = np.full(len(values), np.inf, dtype=float)
    wf = np.zeros(len(values), dtype=float)
    wb = np.zeros(len(values), dtype=float)
    wf[valid] = 1.0 / forward_var[valid]
    wb[valid] = 1.0 / backward_var[valid]
    total = wf + wb
    fused[valid] = (wf[valid] * forward[valid] + wb[valid] * backward[valid]) / total[valid]
    fused_var[valid] = 1.0 / total[valid]
    return fused, fused_var


def _multidirectional_kalman(
    heights: np.ndarray,
    observed: np.ndarray,
    step: float,
) -> np.ndarray:
    values = np.where(observed, heights, np.nan)
    estimates: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    row_estimate = np.full_like(values, np.nan, dtype=float)
    row_variance = np.full_like(values, np.inf, dtype=float)
    for row in range(values.shape[0]):
        row_estimate[row], row_variance[row] = _bidirectional_profile(values[row], step)
    estimates.append(row_estimate)
    variances.append(row_variance)
    column_estimate = np.full_like(values, np.nan, dtype=float)
    column_variance = np.full_like(values, np.inf, dtype=float)
    for column in range(values.shape[1]):
        result, result_var = _bidirectional_profile(values[:, column], step)
        column_estimate[:, column] = result
        column_variance[:, column] = result_var
    estimates.append(column_estimate)
    variances.append(column_variance)

    numerator = np.zeros_like(values, dtype=float)
    denominator = np.zeros_like(values, dtype=float)
    for estimate, variance in zip(estimates, variances):
        valid = np.isfinite(estimate) & np.isfinite(variance)
        weight = np.zeros_like(values, dtype=float)
        weight[valid] = 1.0 / np.maximum(variance[valid], 1e-12)
        numerator[valid] += estimate[valid] * weight[valid]
        denominator[valid] += weight[valid]
    result = np.full_like(values, np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    result[observed] = heights[observed]
    return result
