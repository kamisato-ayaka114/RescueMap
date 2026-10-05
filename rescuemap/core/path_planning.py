from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


@dataclass(slots=True)
class RouteResult:
    vertex_indices: np.ndarray
    points: np.ndarray
    traversable_vertices: np.ndarray
    traversable_points: np.ndarray
    total_cost: float
    length_3d: float
    horizontal_length: float
    ascent: float
    descent: float
    max_slope_degrees: float
    mean_safety_risk: float


def plan_safe_path(
    mesh: pv.PolyData,
    start_point: np.ndarray,
    end_point: np.ndarray,
    *,
    max_slope_degrees: float = 35.0,
    slope_weight: float = 3.0,
    safety_weight: float = 2.0,
) -> RouteResult:
    """Plan a terrain route on mesh edges with Dijkstra's algorithm.

    Edge cost combines three-dimensional length, slope, reconstruction
    confidence, predicted-surface provenance and an optional ``MaterialRisk``
    point field.  The latter is reserved for a later ground-material model.
    """
    if not 0.0 < max_slope_degrees < 90.0:
        raise ValueError("最大可通行坡度必须在0°到90°之间。")
    if slope_weight < 0.0 or safety_weight < 0.0:
        raise ValueError("坡度权重和安全权重不能为负数。")
    surface = mesh.triangulate()
    if surface.n_points < 2 or surface.n_cells < 1:
        raise ValueError("地形网格没有足够的顶点或三角面。")
    points = np.asarray(surface.points, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape((-1, 4))[:, 1:]
    confidence = _point_scalar(surface, "Confidence", default=1.0)
    source = _point_scalar(surface, "VertexSource", default=0.0)
    material_risk = np.clip(
        _point_scalar(surface, "MaterialRisk", default=0.0), 0.0, 1.0
    )
    adjacency: list[list[tuple[int, float, float]]] = [
        [] for _ in range(surface.n_points)
    ]
    traversable = np.zeros(surface.n_points, dtype=bool)
    seen_edges: set[tuple[int, int]] = set()
    for triangle in faces:
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            edge = (min(first, second), max(first, second))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            delta = points[second] - points[first]
            horizontal = float(np.hypot(delta[0], delta[1]))
            if horizontal <= 1e-12:
                continue
            slope_degrees = float(
                np.degrees(np.arctan2(abs(float(delta[2])), horizontal))
            )
            if slope_degrees > max_slope_degrees:
                continue
            length_3d = float(np.linalg.norm(delta))
            slope_risk = (slope_degrees / max_slope_degrees) ** 2
            confidence_risk = 1.0 - float(
                np.clip(min(confidence[first], confidence[second]), 0.0, 1.0)
            )
            prediction_risk = float(max(source[first], source[second]) > 0.0)
            ground_risk = float(
                np.clip((material_risk[first] + material_risk[second]) * 0.5, 0.0, 1.0)
            )
            safety_risk = float(
                np.clip(
                    0.55 * slope_risk
                    + 0.20 * confidence_risk
                    + 0.15 * prediction_risk
                    + 0.10 * ground_risk,
                    0.0,
                    1.0,
                )
            )
            base_cost = length_3d * (
                1.0 + slope_weight * slope_risk + safety_weight * safety_risk
            )
            # Uphill travel is normally more costly than descending the same
            # edge, so the graph is directed even though the mesh is not.
            grade = abs(float(delta[2])) / horizontal
            first_to_second = base_cost * (
                1.0 + (0.75 if delta[2] > 0 else 0.30) * grade
            )
            second_to_first = base_cost * (
                1.0 + (0.75 if delta[2] < 0 else 0.30) * grade
            )
            adjacency[first].append((second, first_to_second, safety_risk))
            adjacency[second].append((first, second_to_first, safety_risk))
            traversable[first] = True
            traversable[second] = True

    traversable_indices = np.flatnonzero(traversable).astype(np.int64)
    if len(traversable_indices) < 2:
        raise ValueError("当前坡度限制下没有足够的可通行节点。")
    traversable_tree = cKDTree(points[traversable_indices])
    start_nearest = int(
        traversable_tree.query(np.asarray(start_point, dtype=float))[1]
    )
    end_nearest = int(
        traversable_tree.query(np.asarray(end_point, dtype=float))[1]
    )
    start_vertex = int(traversable_indices[start_nearest])
    end_vertex = int(traversable_indices[end_nearest])
    if start_vertex == end_vertex:
        raise ValueError("起点和终点落在同一个可通行节点上，请重新选择。")

    path, total_cost, edge_risks = _dijkstra(
        adjacency, start_vertex, end_vertex
    )
    route_points = points[path]
    deltas = np.diff(route_points, axis=0)
    horizontal_steps = np.hypot(deltas[:, 0], deltas[:, 1])
    slopes = np.degrees(
        np.arctan2(np.abs(deltas[:, 2]), np.maximum(horizontal_steps, 1e-12))
    )
    return RouteResult(
        vertex_indices=np.asarray(path, dtype=np.int64),
        points=route_points,
        traversable_vertices=traversable_indices,
        traversable_points=points[traversable],
        total_cost=float(total_cost),
        length_3d=float(np.linalg.norm(deltas, axis=1).sum()),
        horizontal_length=float(horizontal_steps.sum()),
        ascent=float(np.clip(deltas[:, 2], 0.0, None).sum()),
        descent=float(np.clip(-deltas[:, 2], 0.0, None).sum()),
        max_slope_degrees=float(slopes.max(initial=0.0)),
        mean_safety_risk=float(np.mean(edge_risks)) if edge_risks else 0.0,
    )


def _point_scalar(mesh: pv.PolyData, name: str, *, default: float) -> np.ndarray:
    if name not in mesh.point_data:
        return np.full(mesh.n_points, default, dtype=float)
    values = np.asarray(mesh.point_data[name], dtype=float).reshape((-1,))
    if len(values) != mesh.n_points:
        return np.full(mesh.n_points, default, dtype=float)
    return values


def _dijkstra(
    adjacency: list[list[tuple[int, float, float]]],
    start: int,
    goal: int,
) -> tuple[list[int], float, list[float]]:
    distances = np.full(len(adjacency), np.inf, dtype=float)
    previous = np.full(len(adjacency), -1, dtype=np.int64)
    previous_risk = np.zeros(len(adjacency), dtype=float)
    distances[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        if current == goal:
            break
        for neighbour, edge_cost, edge_risk in adjacency[current]:
            candidate = distance + edge_cost
            if candidate >= distances[neighbour]:
                continue
            distances[neighbour] = candidate
            previous[neighbour] = current
            previous_risk[neighbour] = edge_risk
            heapq.heappush(queue, (candidate, neighbour))
    if not np.isfinite(distances[goal]):
        raise ValueError(
            "起点与终点之间没有可通行路径；可尝试提高最大坡度，"
            "或检查地形网格中是否仍有断裂。"
        )
    path = [goal]
    risks: list[float] = []
    while path[-1] != start:
        risks.append(float(previous_risk[path[-1]]))
        parent = int(previous[path[-1]])
        if parent < 0:
            raise ValueError("路径回溯失败。")
        path.append(parent)
    path.reverse()
    risks.reverse()
    return path, float(distances[goal]), risks
