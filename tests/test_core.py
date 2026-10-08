import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt

from rescuemap.core.calibration import (
    ground_alignment_transform,
    uniform_scale_transform,
)
from rescuemap.core.colmap_runner import build_colmap_stages
from rescuemap.core.dynamic_mask import (
    _clustered_dynamic_cells,
    _point_level_dynamic_mask,
)
from rescuemap.core.terrain import (
    ACCEPTED,
    FILL_AUTO,
    FILL_CONCAVE,
    FILL_CONVEX,
    FILL_FLAT,
    FILL_PLANE,
    FILL_QUADRATIC,
    TerrainGrid,
    build_terrain_grid,
)
from rescuemap.core.scene import SceneModel
from rescuemap.core.path_planning import plan_safe_path
from rescuemap.core.terrain_mesh import build_terrain_mesh


class DynamicSuppressionTests(unittest.TestCase):
    def test_only_clustered_rejections_expand_into_mask_cells(self):
        rejected = np.zeros((7, 7), dtype=np.int32)
        observed = np.zeros_like(rejected)
        rejected[3, 3] = observed[3, 3] = 10
        rejected[3, 4] = observed[3, 4] = 10
        rejected[0, 0] = observed[0, 0] = 20

        dynamic = _clustered_dynamic_cells(rejected, observed)

        self.assertTrue(dynamic[3, 3])
        self.assertFalse(dynamic[2, 3])
        self.assertFalse(dynamic[0, 0])

    def test_point_mask_does_not_remove_the_complete_grid_cell(self):
        cells = np.zeros((4, 4), dtype=bool)
        cells[1, 1] = True
        evidence = [(np.array([15.0, 15.0]), True)]

        mask = _point_level_dynamic_mask(
            40, 40, 10, cells, evidence, radius_scale=0.4
        )

        self.assertTrue(mask[15, 15])
        self.assertFalse(mask[10, 10])
        self.assertLess(int(mask.sum()), 10 * 10)

    def test_two_pass_pipeline_uses_generated_static_images_and_masks(self):
        with TemporaryDirectory(dir=Path(__file__).parent) as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            workspace.mkdir()
            stages = build_colmap_stages(
                root / "colmap.exe",
                root / "images",
                workspace,
                "standard",
                True,
            )

            self.assertEqual(len(stages), 11)
            self.assertEqual(stages[3].kind, "generate_masks")
            formal_feature = stages[4]
            self.assertIn("--ImageReader.mask_path", formal_feature.arguments)
            self.assertIn(str(workspace / "static_images"), formal_feature.arguments)


class CalibrationTests(unittest.TestCase):
    def test_selected_plane_maps_to_zero_height(self):
        points = np.array(
            [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 3.0, 3.2]]
        )
        transform = ground_alignment_transform(points)
        transformed = np.c_[points, np.ones(3)] @ transform.T
        np.testing.assert_allclose(transformed[:, 2], 0.0, atol=1e-10)

    def test_uniform_scale(self):
        point = np.array([1.0, 2.0, 3.0, 1.0])
        transformed = uniform_scale_transform(2.0) @ point
        np.testing.assert_allclose(transformed[:3], [2.0, 4.0, 6.0])


class SceneTests(unittest.TestCase):
    def test_coordinate_transform_preserves_rgb(self):
        cloud = pv.PolyData(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        normals = np.array([[1.0, 0.0, 0.0]] * 3, dtype=np.float32)
        rgb = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        cloud.point_data["Normals"] = normals
        cloud.point_data.active_normals_name = "Normals"
        cloud.point_data["RGB"] = rgb
        scene = SceneModel(Path("test.ply"), cloud, cloud, False, np.eye(4))

        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        scene.apply_transform(transform)

        np.testing.assert_array_equal(scene.original.point_data["RGB"], rgb)
        self.assertEqual(scene.original.point_data["RGB"].dtype, np.uint8)
        np.testing.assert_allclose(
            scene.original.point_data["Normals"],
            np.array([[0.0, 1.0, 0.0]] * 3),
        )
        self.assertEqual(scene.rgb_array_name(), "RGB")

    def test_normals_are_not_mistaken_for_rgb(self):
        cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
        cloud.point_data["Normals"] = np.array([[0.0, 0.0, 1.0]])
        scene = SceneModel(Path("test.ply"), cloud, cloud, False, np.eye(4))

        self.assertIsNone(scene.rgb_array_name())


class TerrainTests(unittest.TestCase):
    def test_manual_flat_and_plane_strategies_use_surrounding_observations(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 51),
            np.linspace(0.0, 10.0, 51),
            indexing="ij",
        )
        z = 0.1 * x + 0.05 * y + 2.0
        keep = ~((x > 3.0) & (x < 7.0) & (y > 3.0) & (y < 7.0))
        grid = build_terrain_grid(
            np.c_[x[keep], y[keep], z[keep]],
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=0.2,
            min_points=1,
            algorithm="harmonic",
        )
        region_id = 1
        mask = grid.region_labels == region_id

        grid.apply_fill_strategy(region_id, FILL_FLAT)
        self.assertLess(float(np.ptp(grid.predicted_height[mask])), 1e-10)

        grid.apply_fill_strategy(region_id, FILL_PLANE)
        ix, iy = np.where(mask)
        expected = 0.1 * grid.x_centers[ix] + 0.05 * grid.y_centers[iy] + 2.0
        self.assertLess(
            float(np.max(np.abs(grid.predicted_height[mask] - expected))), 0.03
        )

    def test_manual_convex_and_concave_have_opposite_shape(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 51),
            np.linspace(0.0, 10.0, 51),
            indexing="ij",
        )
        z = np.full_like(x, 3.0)
        keep = ~((x > 2.0) & (x < 8.0) & (y > 2.0) & (y < 8.0))
        grid = build_terrain_grid(
            np.c_[x[keep], y[keep], z[keep]],
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=0.2,
            min_points=1,
            algorithm="harmonic",
        )
        mask = grid.region_labels == 1

        grid.apply_fill_strategy(1, FILL_CONVEX, 0.4)
        convex_max = float(np.max(grid.predicted_height[mask]))
        grid.apply_fill_strategy(1, FILL_CONCAVE, 0.4)
        concave_min = float(np.min(grid.predicted_height[mask]))

        self.assertGreater(convex_max, 3.35)
        self.assertLess(concave_min, 2.65)

    def test_guide_point_majority_selects_the_intended_surface_layer(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 31),
            np.linspace(0.0, 10.0, 31),
            indexing="ij",
        )
        z = np.full_like(x, 2.0)
        keep = ~((x > 3.0) & (x < 7.0) & (y > 3.0) & (y < 7.0))
        grid = build_terrain_grid(
            np.c_[x[keep], y[keep], z[keep]],
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=1.0 / 3.0,
            min_points=1,
            algorithm="harmonic",
        )
        guide_points = np.array(
            [
                [2.0, 2.0, 2.0], [8.0, 2.0, 2.0],
                [8.0, 8.0, 2.0], [2.0, 8.0, 2.0],
                [5.0, 2.0, 2.0], [5.0, 8.0, 2.0],
                [4.0, 4.0, 0.2],
            ]
        )
        grid.set_guide_points(1, guide_points)
        grid.apply_fill_strategy(1, FILL_PLANE)
        mask = grid.region_labels == 1

        self.assertEqual(grid.region_summary(1)["guide_points"], 7)
        np.testing.assert_allclose(
            grid.predicted_height[mask], 2.0, atol=1e-6
        )

    def test_quadratic_fill_recovers_a_gently_curved_occluded_surface(self):
        coordinates = np.arange(11, dtype=float)
        x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
        expected = (
            2.0
            + 0.02 * (x - 5.0) ** 2
            + 0.01 * (y - 5.0) ** 2
            + 0.005 * (x - 5.0) * (y - 5.0)
            + 0.1 * (x - 5.0)
        )
        predicted = np.zeros((11, 11), dtype=bool)
        predicted[3:8, 3:8] = True
        observed = ~predicted
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[predicted] = 1
        grid = TerrainGrid(
            x_centers=coordinates,
            y_centers=coordinates,
            observed_height=np.where(observed, expected, np.nan),
            predicted_height=np.where(predicted, 0.0, expected),
            observed=observed,
            predicted=predicted,
            distance_to_observation=distance_transform_edt(~observed),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="harmonic",
        )
        guide_xy = np.array(
            [
                [2, 2], [5, 2], [8, 2], [8, 5], [8, 8], [5, 8],
                [2, 8], [2, 5], [1, 3], [9, 3], [9, 7], [1, 7],
            ],
            dtype=int,
        )
        guide_z = expected[guide_xy[:, 0], guide_xy[:, 1]]
        guides = np.column_stack((guide_xy, guide_z))
        grid.set_guide_points(1, guides)

        grid.apply_fill_strategy(1, FILL_QUADRATIC)

        np.testing.assert_allclose(
            grid.predicted_height[predicted], expected[predicted], atol=1e-6
        )
        self.assertEqual(grid.fill_strategy_for(1), FILL_QUADRATIC)

    def test_guide_points_do_not_expand_the_original_prediction_region(self):
        observed = np.zeros((9, 9), dtype=bool)
        observed[:, 0] = True
        observed[:, -1] = True
        observed[0, :] = True
        observed[-1, :] = True
        # A stray observed row splits what a user sees as one empty region.
        observed[4, 2:7] = True
        predicted = ~observed
        labels = np.zeros((9, 9), dtype=np.int32)
        labels[1:4, 1:8] = 1
        labels[5:8, 1:8] = 2
        grid = TerrainGrid(
            x_centers=np.arange(9, dtype=float),
            y_centers=np.arange(9, dtype=float),
            observed_height=np.where(observed, 1.0, np.nan),
            predicted_height=np.ones((9, 9), dtype=float),
            observed=observed,
            predicted=predicted,
            distance_to_observation=distance_transform_edt(~observed),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="harmonic",
        )
        guides = np.array(
            [[1.0, 1.0, 2.0], [7.0, 1.0, 2.0],
             [7.0, 7.0, 2.0], [1.0, 7.0, 2.0]]
        )
        grid.set_guide_points(1, guides)
        grid.apply_fill_strategy(1, FILL_FLAT)

        expected_region = labels == 1
        self.assertTrue(np.array_equal(grid.region_mask(1), expected_region))
        self.assertTrue(np.array_equal(grid.prediction_mask(), predicted))
        self.assertTrue(np.array_equal(grid.observation_mask(), observed))
        np.testing.assert_allclose(grid.predicted_height[expected_region], 2.0)

    def test_explicit_rectangle_adds_regular_cells_and_overrides_wrong_layer(self):
        observed = np.ones((9, 9), dtype=bool)
        predicted = np.zeros((9, 9), dtype=bool)
        predicted[2:4, 1:8] = True
        observed[predicted] = False
        labels = np.zeros((9, 9), dtype=np.int32)
        labels[predicted] = 1
        grid = TerrainGrid(
            x_centers=np.arange(9, dtype=float),
            y_centers=np.arange(9, dtype=float),
            observed_height=np.where(observed, 0.5, np.nan),
            predicted_height=np.where(predicted, 2.0, 0.5),
            observed=observed,
            predicted=predicted,
            distance_to_observation=distance_transform_edt(~observed),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="harmonic",
        )
        guides = np.array(
            [[1.0, 1.0, 2.0], [7.0, 1.0, 2.0],
             [7.0, 7.0, 2.0], [1.0, 7.0, 2.0]]
        )
        grid.set_guide_points(1, guides)

        count = grid.add_manual_fill_rectangle(1, (2.0, 6.0, 1.0, 7.0))
        grid.apply_fill_strategy(1, FILL_FLAT)

        rectangle = np.zeros((9, 9), dtype=bool)
        rectangle[2:7, 1:8] = True
        self.assertEqual(count, int(rectangle.sum()))
        self.assertTrue(np.all(grid.region_mask(1)[rectangle]))
        self.assertFalse(np.any(grid.observation_mask()[rectangle]))
        np.testing.assert_allclose(grid.predicted_height[rectangle], 2.0)
        self.assertEqual(grid.manual_fill_bounds[1], [[2.0, 6.0, 1.0, 7.0]])
        prediction_xy = np.asarray(grid.prediction_polydata().points)[:, :2]
        self.assertEqual(len(prediction_xy), len(np.unique(prediction_xy, axis=0)))

        # Returning to automatic mode keeps explicitly drawn cells on the
        # guide-point surface instead of exposing the lower observed layer.
        grid.apply_fill_strategy(1, FILL_AUTO)
        np.testing.assert_allclose(grid.predicted_height[rectangle], 2.0)
        grid.set_decision(1, ACCEPTED)
        mesh = build_terrain_mesh(grid, max_slope_degrees=89.0)
        self.assertTrue(np.any(np.asarray(mesh.point_data["Confidence"]) <= 0.35))

    def test_collinear_guides_fit_without_expanding_the_region(self):
        observed = np.ones((7, 7), dtype=bool)
        observed[2:5, 1:6] = False
        predicted = ~observed
        labels = np.zeros((7, 7), dtype=np.int32)
        labels[predicted] = 1
        grid = TerrainGrid(
            x_centers=np.arange(7, dtype=float),
            y_centers=np.arange(7, dtype=float),
            observed_height=np.where(observed, 2.0, np.nan),
            predicted_height=np.full((7, 7), 2.0),
            observed=observed,
            predicted=predicted,
            distance_to_observation=distance_transform_edt(~observed),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="harmonic",
        )
        # These points are sufficient for the old surface-trend selection but
        # cannot enclose an XY polygon.  The operation should still succeed.
        guides = np.array(
            [[1.0, 1.0, 2.0], [2.0, 1.0, 2.0],
             [3.0, 1.0, 2.0], [4.0, 1.0, 2.0]]
        )
        grid.set_guide_points(1, guides)
        grid.apply_fill_strategy(1, FILL_FLAT)

        np.testing.assert_allclose(grid.predicted_height[predicted], 2.0)
        prediction_points = grid.prediction_polydata().points
        self.assertEqual(len(prediction_points), int(predicted.sum()))
        self.assertEqual(len(np.unique(prediction_points[:, :2], axis=0)), len(prediction_points))

    def test_region_surface_preview_uses_regular_grid_adjacency(self):
        predicted = np.zeros((5, 5), dtype=bool)
        predicted[1:4, 1:4] = True
        observed = ~predicted
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[predicted] = 1
        grid = TerrainGrid(
            x_centers=np.arange(5, dtype=float),
            y_centers=np.arange(5, dtype=float),
            observed_height=np.zeros((5, 5)),
            predicted_height=np.zeros((5, 5)),
            observed=observed,
            predicted=predicted,
            distance_to_observation=np.where(predicted, 1.0, 0.0),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="harmonic",
        )

        preview = grid.region_surface_polydata(1)

        self.assertEqual(preview.n_points, 9)
        self.assertEqual(preview.n_faces, 8)

    def test_harmonic_completion_fills_entire_flat_hole(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 51),
            np.linspace(0.0, 10.0, 51),
            indexing="ij",
        )
        z = np.full_like(x, 2.5)
        keep = ~((x > 2.0) & (x < 8.0) & (y > 2.0) & (y < 8.0))
        grid = build_terrain_grid(
            np.c_[x[keep], y[keep], z[keep]],
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=0.2,
            min_points=1,
            max_prediction_distance_cells=1.0,
            algorithm="harmonic",
        )

        self.assertTrue(np.all(grid.observed | grid.predicted))
        np.testing.assert_allclose(
            grid.predicted_height[grid.predicted], 2.5, atol=1e-5
        )

    def test_harmonic_completion_uses_2d_boundary_for_sloped_plane(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 51),
            np.linspace(0.0, 10.0, 51),
            indexing="ij",
        )
        z = 0.12 * x - 0.07 * y + 1.3
        keep = ~((x > 2.0) & (x < 8.0) & (y > 2.0) & (y < 8.0))
        grid = build_terrain_grid(
            np.c_[x[keep], y[keep], z[keep]],
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=0.2,
            min_points=1,
            algorithm="harmonic",
        )
        ix, iy = np.where(grid.predicted)
        expected = 0.12 * grid.x_centers[ix] - 0.07 * grid.y_centers[iy] + 1.3
        error = np.abs(grid.predicted_height[ix, iy] - expected)

        self.assertTrue(np.all(grid.observed | grid.predicted))
        self.assertLess(float(np.max(error)), 0.02)

    def test_short_gap_is_predicted_and_human_decision_is_saved(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 101),
            np.linspace(0.0, 10.0, 101),
            indexing="ij",
        )
        z = 0.1 * x + 0.05 * y
        keep = ~((x > 4.0) & (x < 6.0) & (y > 4.0) & (y < 6.0))
        points = np.c_[x[keep], y[keep], z[keep]]
        grid = build_terrain_grid(
            points,
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=0.1,
            min_points=1,
            max_prediction_distance_cells=12.0,
        )
        self.assertGreater(grid.predicted.sum(), 0)
        self.assertEqual(grid.region_count, 1)
        grid.set_decision(1, ACCEPTED)
        self.assertEqual(grid.decision_for(1), ACCEPTED)

    def test_multidirectional_kalman_predicts_planar_gap(self):
        x, y = np.meshgrid(
            np.linspace(0.0, 10.0, 101),
            np.linspace(0.0, 10.0, 101),
            indexing="ij",
        )
        z = 0.1 * x + 0.05 * y
        keep = ~((x > 4.0) & (x < 6.0) & (y > 4.0) & (y < 6.0))
        points = np.c_[x[keep], y[keep], z[keep]]
        grid = build_terrain_grid(
            points,
            bounds_xy=(0.0, 10.0, 0.0, 10.0),
            cell_size=0.1,
            min_points=1,
            max_prediction_distance_cells=12.0,
            algorithm="kalman",
        )
        ix, iy = np.where(grid.predicted)
        expected = 0.1 * grid.x_centers[ix] + 0.05 * grid.y_centers[iy]
        error = np.abs(grid.predicted_height[ix, iy] - expected)
        self.assertLess(float(np.mean(error)), 0.08)

    def test_mesh_uses_only_accepted_predictions(self):
        observed_height = np.zeros((3, 3), dtype=float)
        observed = np.ones((3, 3), dtype=bool)
        observed[1, 1] = False
        predicted = np.zeros((3, 3), dtype=bool)
        predicted[1, 1] = True
        predicted_height = np.zeros((3, 3), dtype=float)
        labels = np.zeros((3, 3), dtype=np.int32)
        labels[1, 1] = 1
        grid = TerrainGrid(
            x_centers=np.arange(3, dtype=float),
            y_centers=np.arange(3, dtype=float),
            observed_height=observed_height,
            predicted_height=predicted_height,
            observed=observed,
            predicted=predicted,
            distance_to_observation=np.where(predicted, 1.0, 0.0),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="kalman",
        )

        pending_mesh = build_terrain_mesh(grid)
        self.assertEqual(pending_mesh.n_points, 8)
        self.assertTrue(np.all(pending_mesh.point_data["VertexSource"] == 0))

        grid.set_decision(1, ACCEPTED)
        accepted_mesh = build_terrain_mesh(grid)
        self.assertEqual(accepted_mesh.n_points, 9)
        self.assertEqual(
            int(np.count_nonzero(accepted_mesh.point_data["VertexSource"] == 1)), 1
        )
        self.assertGreater(
            int(np.count_nonzero(accepted_mesh.cell_data["FaceSource"] == 1)), 0
        )

    def test_mesh_rejects_edges_above_slope_limit(self):
        grid = TerrainGrid(
            x_centers=np.arange(2, dtype=float),
            y_centers=np.arange(2, dtype=float),
            observed_height=np.array([[0.0, 0.0], [10.0, 10.0]]),
            predicted_height=np.full((2, 2), np.nan),
            observed=np.ones((2, 2), dtype=bool),
            predicted=np.zeros((2, 2), dtype=bool),
            distance_to_observation=np.zeros((2, 2)),
            region_labels=np.zeros((2, 2), dtype=np.int32),
            cell_size=1.0,
            min_points=1,
            algorithm="linear",
        )

        with self.assertRaisesRegex(ValueError, "没有生成任何三角面"):
            build_terrain_mesh(grid, max_slope_degrees=45.0)

    def test_mesh_keeps_steep_faces_inside_accepted_prediction(self):
        predicted = np.ones((3, 3), dtype=bool)
        labels = np.ones((3, 3), dtype=np.int32)
        grid = TerrainGrid(
            x_centers=np.arange(3, dtype=float),
            y_centers=np.arange(3, dtype=float),
            observed_height=np.full((3, 3), np.nan),
            predicted_height=np.array(
                [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]]
            ),
            observed=np.zeros((3, 3), dtype=bool),
            predicted=predicted,
            distance_to_observation=np.ones((3, 3)),
            region_labels=labels,
            cell_size=1.0,
            min_points=1,
            algorithm="harmonic",
        )
        grid.set_decision(1, ACCEPTED)

        mesh = build_terrain_mesh(grid, max_slope_degrees=10.0)

        self.assertEqual(mesh.n_cells, 8)
        self.assertTrue(np.all(mesh.cell_data["FaceSource"] == 1))
        self.assertEqual(
            int(mesh.field_data["AcceptedPredictionFaces"].ravel()[0]), 8
        )


class PathPlanningTests(unittest.TestCase):
    def test_dijkstra_finds_route_on_flat_grid(self):
        x, y = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(9)))
        faces = []
        for ix in range(2):
            for iy in range(2):
                a = ix * 3 + iy
                b = (ix + 1) * 3 + iy
                c = (ix + 1) * 3 + iy + 1
                d = ix * 3 + iy + 1
                faces.extend(([3, a, b, c], [3, a, c, d]))
        mesh = pv.PolyData(points, np.asarray(faces).ravel())

        result = plan_safe_path(mesh, points[0], points[-1])

        np.testing.assert_allclose(result.points[0], points[0])
        np.testing.assert_allclose(result.points[-1], points[-1])
        self.assertAlmostEqual(result.length_3d, 2.0 * np.sqrt(2.0), places=6)
        self.assertEqual(len(result.traversable_vertices), 9)
        self.assertEqual(result.max_slope_degrees, 0.0)

    def test_slope_limit_can_disconnect_route(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [1.0, 0.0, 10.0], [1.0, 1.0, 10.0]]
        )
        mesh = pv.PolyData(
            points,
            np.array([3, 0, 2, 3, 3, 0, 3, 1], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "没有可通行路径"):
            plan_safe_path(
                mesh, points[0], points[2], max_slope_degrees=35.0
            )


if __name__ == "__main__":
    unittest.main()
