from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


@dataclass(slots=True)
class SceneModel:
    path: Path
    original: pv.DataSet
    display: pv.DataSet
    simplified: bool
    calibration_matrix: np.ndarray
    unit_name: str = "COLMAP单位"

    @classmethod
    def load(cls, path: Path, max_display_faces: int = 500_000) -> "SceneModel":
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到模型文件：{path}")
        if path.suffix.lower() != ".ply":
            raise ValueError("Viewer 0.1 暂时只支持 PLY 文件。")

        data = pv.read(path)
        display = data
        simplified = False

        if isinstance(data, pv.PolyData) and data.n_faces > max_display_faces:
            reduction = 1.0 - max_display_faces / data.n_faces
            display = data.decimate_pro(
                reduction,
                preserve_topology=True,
            )
            simplified = True

        return cls(
            path=path,
            original=data,
            display=display,
            simplified=simplified,
            calibration_matrix=np.eye(4),
        )

    def apply_transform(self, transform: np.ndarray) -> None:
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("坐标变换矩阵必须为4×4。")
        same_data = self.display is self.original
        self.original = self.original.transform(
            transform,
            # Only transform points and the active geometric vectors (for
            # example Normals).  RGB is also stored as a three-component
            # point array, but it is data rather than a spatial vector.
            # Transforming every three-component array rotates the colours
            # and changes uint8 RGB values into invalid floating-point data.
            transform_all_input_vectors=False,
            inplace=False,
        )
        if same_data:
            self.display = self.original
        else:
            self.display = self.display.transform(
                transform,
                transform_all_input_vectors=False,
                inplace=False,
            )
        self.calibration_matrix = transform @ self.calibration_matrix

    @property
    def is_point_cloud(self) -> bool:
        return self.original.n_cells == 0 or (
            isinstance(self.original, pv.PolyData)
            and self.original.n_faces == 0
        )

    def summary(self) -> dict[str, str]:
        xmin, xmax, ymin, ymax, zmin, zmax = self.original.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        size_mb = self.path.stat().st_size / (1024 * 1024)

        if isinstance(self.original, pv.PolyData):
            faces = self.original.n_faces
        else:
            faces = self.original.n_cells

        return {
            "文件": str(self.path),
            "类型": "点云" if self.is_point_cloud else "网格",
            "文件大小": f"{size_mb:.2f} MB",
            "顶点数": f"{self.original.n_points:,}",
            "三角面/单元数": f"{faces:,}",
            "X 范围": f"{xmin:.4f} ～ {xmax:.4f}",
            "Y 范围": f"{ymin:.4f} ～ {ymax:.4f}",
            "Z 范围": f"{zmin:.4f} ～ {zmax:.4f}",
            "模型尺寸": f"{dx:.4f} × {dy:.4f} × {dz:.4f}",
            "坐标单位": self.unit_name,
            "预览简化": "是" if self.simplified else "否",
        }

    def rgb_array_name(self) -> str | None:
        # Do not treat arbitrary 3-component arrays such as Normals as RGB.
        # COLMAP PLY files normally use "RGB", but accept a few conventional
        # spellings for imported models.
        preferred_names = ("RGB", "RGBA", "rgb", "rgba", "Colors", "colors")
        for name in preferred_names:
            values = self.display.point_data.get(name)
            if values is not None and getattr(values, "ndim", 0) == 2:
                expected_components = 4 if "A" in name.upper() else 3
                if values.shape[1] == expected_components:
                    return name
        return None
