from __future__ import annotations

import numpy as np
import pyvista as pv

from rescuemap.core.terrain import (
    ACCEPTED,
    FILL_STRATEGY_CODES,
    TerrainGrid,
)


OBSERVED = np.uint8(0)
ACCEPTED_PREDICTION = np.uint8(1)

OBSERVED_RGB = np.array([55, 185, 95], dtype=np.uint8)
PREDICTED_RGB = np.array([242, 190, 45], dtype=np.uint8)


def build_terrain_mesh(
    grid: TerrainGrid,
    *,
    max_slope_degrees: float = 75.0,
) -> pv.PolyData:
    """Triangulate observed cells and predictions explicitly accepted by a user.

    Grid cells marked as uncertain, reshoot, or not yet reviewed remain holes.
    Slope filtering protects observed and observed/predicted transition faces
    from accidental cross-layer connections.  A face made only of explicitly
    accepted prediction vertices is retained so a completed surface is not cut
    into holes; its actual traversability is checked later by path planning.
    The returned mesh retains point- and face-level provenance arrays.
    """
    if not 0.0 < max_slope_degrees < 90.0:
        raise ValueError("最大坡度必须在0°到90°之间。")

    accepted_regions = [
        region_id
        for region_id in range(1, grid.region_count + 1)
        if grid.decision_for(region_id) == ACCEPTED
    ]
    prediction_mask = grid.prediction_mask()
    effective_labels = grid.effective_region_labels()
    accepted_prediction = prediction_mask & np.isin(
        effective_labels, accepted_regions
    )
    observed = grid.observation_mask()
    valid = observed | accepted_prediction
    if int(valid.sum()) < 3:
        raise ValueError("有效地形格子不足，至少需要3个真实或已接受的预测格子。")

    heights = np.where(observed, grid.observed_height, grid.predicted_height)
    index_map = np.full(valid.shape, -1, dtype=np.int64)
    gx, gy = np.meshgrid(grid.x_centers, grid.y_centers, indexing="ij")
    ix, iy = np.where(valid)
    points = np.column_stack((gx[ix, iy], gy[ix, iy], heights[ix, iy]))
    index_map[ix, iy] = np.arange(len(points), dtype=np.int64)

    vertex_source = np.where(
        accepted_prediction[ix, iy], ACCEPTED_PREDICTION, OBSERVED
    ).astype(np.uint8)
    source_rgb = np.where(
        vertex_source[:, None] == ACCEPTED_PREDICTION,
        PREDICTED_RGB,
        OBSERVED_RGB,
    ).astype(np.uint8)
    confidence = np.ones(len(points), dtype=np.float32)
    if np.any(accepted_prediction):
        max_distance = max(
            float(grid.distance_to_observation[prediction_mask].max(initial=1.0)),
            1.0,
        )
        predicted_vertex = vertex_source == ACCEPTED_PREDICTION
        confidence[predicted_vertex] = np.clip(
            1.0
            - grid.distance_to_observation[ix[predicted_vertex], iy[predicted_vertex]]
            / max_distance,
            0.05,
            0.99,
        ).astype(np.float32)
        manual_prediction = grid.manual_fill_labels[ix, iy] > 0
        confidence[manual_prediction] = np.minimum(
            confidence[manual_prediction], 0.35
        )

    max_slope = np.tan(np.deg2rad(max_slope_degrees))
    triangle_indices: list[tuple[int, int, int]] = []
    face_sources: list[int] = []
    face_slopes: list[float] = []
    slope_rejected_faces = 0
    accepted_prediction_faces = 0
    nx, ny = valid.shape
    for x_index in range(nx - 1):
        for y_index in range(ny - 1):
            corners = (
                int(index_map[x_index, y_index]),
                int(index_map[x_index + 1, y_index]),
                int(index_map[x_index + 1, y_index + 1]),
                int(index_map[x_index, y_index + 1]),
            )
            layouts = (
                ((corners[0], corners[1], corners[2]),
                 (corners[0], corners[2], corners[3])),
                ((corners[0], corners[1], corners[3]),
                 (corners[1], corners[2], corners[3])),
            )
            best_faces: list[tuple[tuple[int, int, int], int, float, bool]] = []
            best_score = (-1, -np.inf)
            for layout in layouts:
                candidate_faces: list[
                    tuple[tuple[int, int, int], int, float, bool]
                ] = []
                candidate_max_slope = 0.0
                for triangle in layout:
                    if min(triangle) < 0:
                        continue
                    triangle_array = np.asarray(triangle)
                    triangle_points = points[triangle_array]
                    slope_ratio = _maximum_edge_slope(triangle_points)
                    if not np.isfinite(slope_ratio):
                        continue
                    face_source = int(np.max(vertex_source[triangle_array]))
                    prediction_only = bool(
                        np.all(
                            vertex_source[triangle_array]
                            == ACCEPTED_PREDICTION
                        )
                    )
                    if not prediction_only and slope_ratio > max_slope:
                        continue
                    candidate_faces.append(
                        (triangle, face_source, slope_ratio, prediction_only)
                    )
                    candidate_max_slope = max(candidate_max_slope, slope_ratio)
                score = (len(candidate_faces), -candidate_max_slope)
                if score > best_score:
                    best_score = score
                    best_faces = candidate_faces

            possible_faces = sum(
                1 for triangle in layouts[0] if min(triangle) >= 0
            )
            slope_rejected_faces += max(possible_faces - len(best_faces), 0)
            for triangle, face_source, slope_ratio, prediction_only in best_faces:
                triangle_indices.append(triangle)
                face_sources.append(face_source)
                face_slopes.append(float(np.rad2deg(np.arctan(slope_ratio))))
                accepted_prediction_faces += int(prediction_only)

    if not triangle_indices:
        raise ValueError(
            "没有生成任何三角面。请扩大范围、接受合适的预测区域，或提高最大坡度。"
        )

    triangles = np.asarray(triangle_indices, dtype=np.int64)
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    mesh = pv.PolyData(points, faces)
    mesh.point_data["VertexSource"] = vertex_source
    mesh.point_data["Confidence"] = confidence
    mesh.point_data["SourceRGB"] = source_rgb
    fill_strategy = np.zeros(len(points), dtype=np.uint8)
    for vertex_index, (x_index, y_index) in enumerate(zip(ix, iy)):
        if vertex_source[vertex_index] != ACCEPTED_PREDICTION:
            continue
        region_id = int(effective_labels[x_index, y_index])
        fill_strategy[vertex_index] = FILL_STRATEGY_CODES[
            grid.fill_strategy_for(region_id)
        ]
    mesh.point_data["FillStrategy"] = fill_strategy
    face_source_array = np.asarray(face_sources, dtype=np.uint8)
    mesh.cell_data["FaceSource"] = face_source_array
    mesh.cell_data["FaceSlopeDegrees"] = np.asarray(
        face_slopes, dtype=np.float32
    )
    mesh.cell_data["FaceRGB"] = np.where(
        face_source_array[:, None] == ACCEPTED_PREDICTION,
        PREDICTED_RGB,
        OBSERVED_RGB,
    ).astype(np.uint8)
    mesh.field_data["MaxSlopeDegrees"] = np.array([max_slope_degrees])
    mesh.field_data["SlopeRejectedFaces"] = np.array(
        [slope_rejected_faces], dtype=np.int64
    )
    mesh.field_data["AcceptedPredictionFaces"] = np.array(
        [accepted_prediction_faces], dtype=np.int64
    )
    return mesh.compute_normals(
        cell_normals=True,
        point_normals=True,
        split_vertices=False,
        inplace=False,
    )


def _within_slope_limit(points: np.ndarray, max_slope: float) -> bool:
    return _maximum_edge_slope(points) <= max_slope


def _maximum_edge_slope(points: np.ndarray) -> float:
    maximum = 0.0
    for start, end in ((0, 1), (1, 2), (2, 0)):
        delta = points[end] - points[start]
        horizontal = float(np.hypot(delta[0], delta[1]))
        if horizontal <= 1e-12:
            return np.inf
        maximum = max(maximum, abs(float(delta[2])) / horizontal)
    return maximum
